#!/usr/bin/env python3
"""
Master pipeline: Audio → EMAGE → HUGS → PLY frames → GaussianSplats3D viewer.

Replaces MDM with EMAGE as the motion source.

Workflow
--------
  1. EMAGE inference  : audio (.wav) → EMAGE beat-format .npz  (SMPLX 55-joint)
  2. Format convert   : SMPLX .npz   → HUGS SMPL .npz          (24-joint)
  3. Coord rotate     : SMPL .npz    → rotated .npz             (EMAGE→HUGS world)
  4. HUGS render      : rotated .npz → per-frame *_splat.ply
  5. (optional) serve : node util/render-server.js --ply-dir …  → browser viewer

Usage
-----
  python scripts/run_emage2hugs.py \\
      --audio        /path/to/speech.wav \\
      --out_dir      ./output_emage2hugs \\
      --scene        bike              \\
      [--betas       /path/to/betas.npy] \\
      [--center] [--tx 0] [--ty 0] [--tz 0] [--ground 0.1] \\
      [--rx 90] [--rz 180]

Scenes available: bike, citron, jogging  (add more in SCENE_CONFIGS below)

Environment notes
-----------------
  EMAGE inference  : EMAGE_PYTHON  (needs torch≥2.0, smplx, librosa, transformers,
                                    huggingface_hub, einops, omegaconf, yacs)
                     Default: hugs_py39 conda env — install missing deps with:
                       conda run -n hugs_py39 pip install smplx librosa transformers \\
                           huggingface_hub einops yacs easydict omegaconf
  HUGS rendering   : HUGS_PYTHON   (needs torch 1.13, pytorch3d, smplx)
                     Default: hugs conda env (already configured)
  Converter/rotate : HUGS_PYTHON   (numpy only, no GPU needed)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Environment binaries ────────────────────────────────────────────────────

EMAGE_PYTHON = "/home/sigma/anaconda3/envs/hugs_py39/bin/python"
HUGS_PYTHON  = "/home/sigma/anaconda3/envs/hugs/bin/python"

# ─── Repo roots ──────────────────────────────────────────────────────────────

PANTOMATRIX_REPO = "/home/sigma/PantoMatrix"
HUGS_REPO        = "/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar"

# ─── HUGS config / checkpoints ───────────────────────────────────────────────

HUGS_CONFIG   = f"{HUGS_REPO}/cfg_files/release/neuman/hugs_human_scene.yaml"
PRETRAINED    = f"{HUGS_REPO}/output/pretrained_models"
SCENE_CONFIGS = {
    "bike": {
        "human_ckpt": f"{PRETRAINED}/bike/human_final.pth",
        "scene_ckpt": f"{PRETRAINED}/bike/scene_final.pth",
    },
    "citron": {
        "human_ckpt": f"{PRETRAINED}/citron/human_final.pth",
        "scene_ckpt": f"{PRETRAINED}/citron/scene_final.pth",
    },
    "jogging": {
        "human_ckpt": f"{PRETRAINED}/jogging/human_final.pth",
        "scene_ckpt": f"{PRETRAINED}/jogging/scene_final.pth",
    },
}

# ─── Helper ──────────────────────────────────────────────────────────────────

def run(cmd, description, cwd=None):
    print(f"\n{'='*72}")
    print(f"  {description}")
    print(f"{'='*72}")
    print("  " + " ".join(str(c) for c in cmd))
    print()
    result = subprocess.run([str(c) for c in cmd], cwd=cwd or HUGS_REPO)
    if result.returncode != 0:
        print(f"\n✗ Step failed (exit {result.returncode}): {description}")
        sys.exit(result.returncode)
    print(f"\n✓ Done: {description}")
    return True


# ─── Benchmark ───────────────────────────────────────────────────────────────

class _BenchmarkStep:
    def __init__(self, benchmark: "Benchmark", step_number: int, step_name: str,
                 input_path: str, output_path: str):
        self._benchmark = benchmark
        self._record: dict = {
            "step":       step_number,
            "name":       step_name,
            "input":      input_path,
            "output":     output_path,
            "start_time": "",
            "end_time":   "",
            "duration_s": None,
            "status":     "running",
        }

    def set_output(self, path: str) -> None:
        self._record["output"] = path

    def __enter__(self) -> "_BenchmarkStep":
        self._t0 = time.perf_counter()
        self._record["start_time"] = datetime.now().isoformat()
        self._benchmark.steps.append(self._record)
        return self

    def __exit__(self, exc_type, exc_val, _exc_tb):
        self._record["duration_s"] = round(time.perf_counter() - self._t0, 3)
        self._record["end_time"]   = datetime.now().isoformat()
        if exc_type is None:
            self._record["status"] = "ok"
        else:
            self._record["status"] = "failed"
            self._record["error"]  = str(exc_val)
            self._benchmark.save()  # persist partial results before pipeline exits
        return False  # never suppress the exception


class Benchmark:
    def __init__(self, out_dir: Path):
        self.steps: list = []
        self._path = out_dir / "benchmark.json"

    def step(self, step_number: int, step_name: str,
             input_path: str = "", output_path: str = "") -> _BenchmarkStep:
        return _BenchmarkStep(self, step_number, step_name, input_path, output_path)

    def record_skipped(self, step_number: int, step_name: str, note: str = "") -> None:
        self.steps.append({
            "step":       step_number,
            "name":       step_name,
            "status":     "skipped",
            "note":       note,
            "duration_s": 0.0,
        })

    def save(self) -> None:
        self._path.write_text(json.dumps(
            {"saved_at": datetime.now().isoformat(), "steps": self.steps}, indent=2
        ))

    def print_summary(self) -> None:
        total = sum(s.get("duration_s") or 0.0 for s in self.steps)
        print(f"\n{'='*72}")
        print("  Benchmark Summary")
        print(f"  {'─'*68}")
        for s in self.steps:
            icon = {"ok": "✓", "failed": "✗", "skipped": "–"}.get(s["status"], "?")
            dur  = s.get("duration_s") or 0.0
            note = f"  ({s['note']})" if s.get("note") else ""
            print(f"  [{icon}] Step {s['step']}: {s['name']:<38} {dur:>8.1f}s{note}")
        print(f"  {'─'*68}")
        print(f"  {'Total':.<44} {total:>8.1f}s")
        print(f"  Saved → {self._path}")
        print(f"{'='*72}\n")


# ─── Pipeline steps ──────────────────────────────────────────────────────────

def step_emage_inference(audio_path: Path, emage_out_dir: Path) -> Path:
    """Run EMAGE on audio_path, return path to the output .npz."""
    emage_out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            EMAGE_PYTHON,
            f"{PANTOMATRIX_REPO}/test_emage_audio.py",
            "--audio_folder", str(audio_path.parent),
            "--save_folder",  str(emage_out_dir),
            # no --visualization: keeps it headless for pipeline use
        ],
        "EMAGE inference (audio → SMPLX motion)",
        cwd=PANTOMATRIX_REPO,
    )

    # test_emage_audio.py saves  <stem>_output.npz
    npz_name = audio_path.stem + "_output.npz"
    npz_path = emage_out_dir / npz_name
    if not npz_path.exists():
        # fallback: pick any .npz in the folder
        candidates = list(emage_out_dir.glob("*.npz"))
        if not candidates:
            print(f"✗ No .npz found in {emage_out_dir} after EMAGE inference.")
            sys.exit(1)
        npz_path = candidates[0]
        print(f"  (using fallback output: {npz_path.name})")

    print(f"  EMAGE output: {npz_path}")
    return npz_path


def step_convert(emage_npz: Path, out_dir: Path, betas_path=None) -> Path:
    """Convert EMAGE SMPLX .npz → HUGS SMPL .npz."""
    converted = out_dir / (emage_npz.stem + "_hugs.npz")
    cmd = [
        HUGS_PYTHON,
        f"{HUGS_REPO}/scripts/convert_emage_to_hugs.py",
        "--input",  str(emage_npz),
        "--output", str(converted),
    ]
    if betas_path:
        cmd += ["--betas", str(betas_path)]
    run(cmd, "Convert EMAGE SMPLX → HUGS SMPL format", cwd=HUGS_REPO)
    return converted


def step_rotate(hugs_npz: Path, out_dir: Path, rx, ry, rz,
                center, tx, ty, tz, ground) -> Path:
    """Apply coordinate-frame rotation so EMAGE world aligns with HUGS world."""
    rotated = out_dir / (hugs_npz.stem + "_rotated.npz")
    cmd = [
        HUGS_PYTHON,
        f"{HUGS_REPO}/scripts/rotate_hugs_motion_v2.py",
        "--input",  str(hugs_npz),
        "--output", str(rotated),
        "--rx", str(rx),
        "--ry", str(ry),
        "--rz", str(rz),
        "--tx", str(tx),
        "--ty", str(ty),
        "--tz", str(tz),
    ]
    if center:
        cmd.append("--center")
    if ground is not None:
        cmd += ["--ground", str(ground)]
    run(cmd, f"Rotate motion (rx={rx}° ry={ry}° rz={rz}°)", cwd=HUGS_REPO)
    return rotated


def step_hugs_render(rotated_npz: Path, scene: str) -> bool:
    """Run HUGS with custom motion to produce PLY frames."""
    cfg = SCENE_CONFIGS[scene]
    run(
        [
            HUGS_PYTHON,
            "main.py",
            "--cfg_file", HUGS_CONFIG,
            f"dataset.seq={scene}",
            "eval=true",
            f"human.ckpt={cfg['human_ckpt']}",
            f"scene.ckpt={cfg['scene_ckpt']}",
            f"custom_motion_path={rotated_npz}",
        ],
        f"HUGS render (scene={scene})",
        cwd=HUGS_REPO,
    )
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audio → EMAGE → HUGS → PLY pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal — use defaults (bike scene, rx=90 rz=180)
  python scripts/run_emage2hugs.py \\
      --audio /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/tts_readback.wav \\
      --out_dir ./output_emage2hugs

  # With avatar body shape and ground snap
  python scripts/run_emage2hugs.py \\
      --audio   speech.wav \\
      --out_dir ./output_emage2hugs \\
      --scene   citron \\
      --betas   ./data/custom_motions/my_betas.npy \\
      --center  --ground 0.1

After rendering, view the PLY frames in the browser:
  node /home/sigma/GaussianSplats3D/util/render-server.js \\
      --ply-dir <HUGS output anim_ply directory>
""",
    )

    parser.add_argument("--audio",   required=True,
                        help="Input speech audio (.wav)")
    parser.add_argument("--out_dir", required=True,
                        help="Directory for all intermediate and final outputs")
    parser.add_argument("--scene",   default="bike",
                        choices=list(SCENE_CONFIGS.keys()),
                        help="HUGS scene/checkpoint to use (default: bike)")
    parser.add_argument("--betas",   default=None,
                        help="Optional .npy/.npz with SMPL betas (10-dim) for body shape")

    # Coordinate rotation — default matches MDM→HUGS empirically validated values
    parser.add_argument("--rx", type=float, default=90.0,
                        help="Root rotation around X in degrees (default: 90)")
    parser.add_argument("--ry", type=float, default=0.0,
                        help="Root rotation around Y in degrees (default: 0)")
    parser.add_argument("--rz", type=float, default=180.0,
                        help="Root rotation around Z in degrees (default: 180)")

    # Translation adjustments
    parser.add_argument("--center", action="store_true",
                        help="Center translation to mean 0 before rendering")
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=0.0)
    parser.add_argument("--tz", type=float, default=0.0,
                        help="Forward Z offset after centering")
    parser.add_argument("--ground", type=float, default=None,
                        help="Snap avatar's lowest Z frame to this value (e.g. 0.1)")

    # Skip flags for resuming a partial run
    parser.add_argument("--skip_emage",   action="store_true",
                        help="Skip EMAGE inference (reuse existing .npz in out_dir/emage/)")
    parser.add_argument("--emage_npz",    default=None,
                        help="Path to pre-computed EMAGE .npz (implies --skip_emage)")

    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"✗ Audio file not found: {audio_path}")
        sys.exit(1)

    if args.scene not in SCENE_CONFIGS:
        print(f"✗ Unknown scene '{args.scene}'. Available: {', '.join(SCENE_CONFIGS)}")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    emage_dir    = out_dir / "emage"
    convert_dir  = out_dir / "convert"
    rotate_dir   = out_dir / "rotate"

    benchmark = Benchmark(out_dir)

    print(f"\n{'='*72}")
    print("  EMAGE → HUGS Pipeline")
    print(f"{'='*72}")
    print(f"  Audio    : {audio_path}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Scene    : {args.scene}")
    print(f"  Rotation : rx={args.rx}° ry={args.ry}° rz={args.rz}°")
    print(f"  Center   : {args.center}  |  tz={args.tz}  |  ground={args.ground}")
    print(f"{'='*72}\n")

    start = datetime.now()

    # ── Step 1: EMAGE inference ──────────────────────────────────────────────
    if args.emage_npz:
        emage_npz = Path(args.emage_npz).resolve()
        print(f"[1/4] Skipping EMAGE — using provided .npz: {emage_npz}")
        benchmark.record_skipped(1, "EMAGE inference", f"--emage_npz {emage_npz.name}")
    elif args.skip_emage:
        candidates = list(emage_dir.glob("*.npz"))
        if not candidates:
            print(f"✗ --skip_emage set but no .npz found in {emage_dir}")
            sys.exit(1)
        emage_npz = candidates[0]
        print(f"[1/4] Skipping EMAGE — reusing: {emage_npz}")
        benchmark.record_skipped(1, "EMAGE inference", f"--skip_emage → {emage_npz.name}")
    else:
        print("[1/4] Running EMAGE inference…")
        with benchmark.step(1, "EMAGE inference", str(audio_path), str(emage_dir)) as bstep:
            emage_npz = step_emage_inference(audio_path, emage_dir)
            bstep.set_output(str(emage_npz))

    # ── Step 2: Convert SMPLX → SMPL ────────────────────────────────────────
    print("[2/4] Converting EMAGE SMPLX → HUGS SMPL format…")
    with benchmark.step(2, "Convert SMPLX → SMPL", str(emage_npz), str(convert_dir)) as bstep:
        hugs_npz = step_convert(emage_npz, convert_dir, args.betas)
        bstep.set_output(str(hugs_npz))

    # ── Step 3: Rotate to HUGS world space ──────────────────────────────────
    print("[3/4] Rotating motion to HUGS world space…")
    with benchmark.step(3, "Rotate to HUGS world space", str(hugs_npz), str(rotate_dir)) as bstep:
        rotated_npz = step_rotate(
            hugs_npz, rotate_dir,
            rx=args.rx, ry=args.ry, rz=args.rz,
            center=args.center,
            tx=args.tx, ty=args.ty, tz=args.tz,
            ground=args.ground,
        )
        bstep.set_output(str(rotated_npz))

    # ── Step 4: HUGS render ──────────────────────────────────────────────────
    print(f"[4/4] Rendering HUGS (scene={args.scene})…")
    with benchmark.step(4, "HUGS render", str(rotated_npz), f"output/{args.scene}/anim_ply"):
        step_hugs_render(rotated_npz, args.scene)

    # ── Record ───────────────────────────────────────────────────────────────
    end = datetime.now()
    record = {
        "start_time":    start.isoformat(),
        "end_time":      end.isoformat(),
        "duration_s":    (end - start).total_seconds(),
        "audio":         str(audio_path),
        "scene":         args.scene,
        "emage_npz":     str(emage_npz),
        "hugs_npz":      str(hugs_npz),
        "rotated_npz":   str(rotated_npz),
        "rotation":      {"rx": args.rx, "ry": args.ry, "rz": args.rz},
        "translation":   {"center": args.center, "tx": args.tx,
                          "ty": args.ty, "tz": args.tz, "ground": args.ground},
    }
    record_path = out_dir / "run_record.json"
    record_path.write_text(json.dumps(record, indent=2))

    benchmark.save()
    benchmark.print_summary()

    print(f"\n{'='*72}")
    print("  ✓ Pipeline complete!")
    print(f"{'='*72}")
    print(f"  Duration   : {record['duration_s']:.1f}s")
    print(f"  EMAGE npz  : {emage_npz}")
    print(f"  Rotated npz: {rotated_npz}")
    print(f"  Run record : {record_path}")
    print(f"  Benchmark  : {out_dir / 'benchmark.json'}")
    print()
    print("  To view in the browser:")
    print("    1. Find the HUGS 'anim_ply' output directory in the HUGS logdir")
    print("    2. Run:")
    print("       node /home/sigma/GaussianSplats3D/util/render-server.js \\")
    print("           --ply-dir <path/to/anim_ply>")
    print("    3. Open http://localhost:8765/ in your browser")


if __name__ == "__main__":
    main()
