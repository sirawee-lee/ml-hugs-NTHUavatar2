"""
Benchmark: GaussianRasterizer time vs Gaussian count.
Uses one existing frame PLY from the HUGS output.
Run with: conda activate hugs && python benchmark_gaussians.py
"""

import math, time
import numpy as np
import torch
import torch.nn.functional as F

# ── parse PLY without plyfile ──────────────────────────────────────────────
def load_ply(path):
    with open(path, 'rb') as f:
        # read header
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        n_verts = int([l for l in header if l.startswith('element vertex')][0].split()[-1])
        props = [l.split()[-1] for l in header if l.startswith('property float')]
        n_props = len(props)
        data = np.frombuffer(f.read(n_verts * n_props * 4),
                             dtype=np.float32).reshape(n_verts, n_props)
    prop_idx = {name: i for i, name in enumerate(props)}
    return data, prop_idx, n_verts

# ── dummy camera ──────────────────────────────────────────────────────────
def make_camera(H=540, W=960):
    fovx = fovy = math.radians(60)

    # View matrix: camera sitting at (0,0,5) looking toward -Z
    view = torch.eye(4, device='cuda')
    view[2, 3] = -5.0          # translate world so scene is in front of camera

    # Projection matrix built from FOV (OpenGL-style perspective)
    near, far = 0.01, 100.0
    f = 1.0 / math.tan(fovx * 0.5)
    proj = torch.zeros(4, 4, device='cuda')
    proj[0, 0] = f
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = 1.0
    full_proj = proj @ view

    return {
        'image_height': H, 'image_width': W,
        'fovx': fovx, 'fovy': fovy,
        'world_view_transform': view,
        'full_proj_transform':  full_proj,
        'camera_center':        torch.tensor([0., 0., 5.], device='cuda'),
    }

# ── time one rasterizer call ───────────────────────────────────────────────
def time_render(means3D, colors, opacity, scales, rotations, data, n_warmup=5, n_runs=20):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    bg = torch.zeros(3, device='cuda')
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)
    settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg,
        scale_modifier=1.0,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=0, campos=data['camera_center'],
        prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(settings)

    def one_pass():
        means2D = torch.zeros_like(means3D, requires_grad=True)
        rasterizer(means3D=means3D, means2D=means2D,
                   colors_precomp=colors,   # skip SH evaluation entirely
                   opacities=opacity, scales=scales, rotations=rotations)

    # warmup
    for _ in range(n_warmup):
        one_pass()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        one_pass()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed_ms

# ── main ───────────────────────────────────────────────────────────────────
PLY = ('output/human_scene/neuman/bike/hugs_trimlp/'
       'demo-dataset.seq=bike/2026-04-21_23-34-02/anim_ply/00091_splat.ply')

data_np, pidx, total = load_ply(PLY)
print(f"Loaded {total:,} Gaussians from PLY")

# Print value ranges so we can verify activations are applied correctly
for col in ['scale_0', 'opacity', 'x', 'z']:
    v = data_np[:, pidx[col]]
    print(f"  {col:10s}: min={v.min():.3f}  max={v.max():.3f}  mean={v.mean():.3f}")

# PLY stores log(scale) and inverse_sigmoid(opacity) — apply activations
# (see hugs/utils/vis.py save_posed_ply lines 72-75)
raw_opac  = torch.tensor(data_np[:, pidx['opacity']], dtype=torch.float32)
raw_scale = torch.tensor(data_np[:, [pidx['scale_0'], pidx['scale_1'], pidx['scale_2']]],
                         dtype=torch.float32)
opac_act  = torch.sigmoid(raw_opac).unsqueeze(1).cuda()      # (N,1)
scale_act = torch.exp(raw_scale).cuda()                       # (N,3)

print(f"\nAfter activations:")
print(f"  opacity : min={opac_act.min():.3f}  max={opac_act.max():.3f}")
print(f"  scale_0 : min={scale_act[:,0].min():.5f}  max={scale_act[:,0].max():.5f}")

xyz_all = torch.tensor(data_np[:, [pidx['x'], pidx['y'], pidx['z']]],
                       dtype=torch.float32).cuda()
rot_all = F.normalize(
    torch.tensor(data_np[:, [pidx['rot_0'], pidx['rot_1'],
                              pidx['rot_2'], pidx['rot_3']]],
                 dtype=torch.float32), dim=-1).cuda()

# Sort by opacity descending so pruned subsets keep the most visible Gaussians
order = torch.argsort(opac_act.squeeze(), descending=True)
xyz_all   = xyz_all[order]
opac_act  = opac_act[order]
scale_act = scale_act[order]
rot_all   = rot_all[order]

# Centre the scene: put camera above the centroid looking toward -Z
centroid = xyz_all.mean(dim=0)
cam = make_camera()
offset = centroid.clone()
offset[2] += 3.0   # camera 3 units behind the scene centroid
cam['camera_center'] = offset
cam['world_view_transform'][0, 3] = -centroid[0]
cam['world_view_transform'][1, 3] = -centroid[1]
cam['world_view_transform'][2, 3] = -(centroid[2] + 3.0)

# target counts — adjust to what the scene actually has
targets = [50_000, 100_000, 200_000, total]
targets = sorted(set(t for t in targets if t <= total))

print(f"\n{'Gaussians':>12}  {'ms/frame':>10}  {'FPS':>8}")
print('-' * 34)
for n in targets:
    xyz   = xyz_all[:n].contiguous()
    opac  = opac_act[:n].contiguous()
    sc    = scale_act[:n].contiguous()
    rot   = rot_all[:n].contiguous()
    colors = torch.full((n, 3), 0.5, device='cuda')   # flat grey — no SH needed

    ms = time_render(xyz, colors, opac, sc, rot, cam)
    print(f"{n:>12,}  {ms:>10.2f}  {1000/ms:>8.1f}")
