#!/usr/bin/env python3
"""
Visualize SMPL mesh and local coordinate frame at pelvis.

This script:
1. Loads SMPL parameters from a .npz file
2. Constructs the SMPL mesh using smplx
3. Extracts the pelvis joint position and global_orient rotation
4. Draws the SMPL mesh in 3D
5. Draws local coordinate axes at the pelvis (X=red, Y=green, Z=blue)
   reflecting the SMPL root joint's local frame

The axes show the actual SMPL root rotation, not world/camera axes.

Usage:
  python visualize_smpl_orientation.py \
    --smpl_npz /path/to/motion.npz \
    --frame 0 \
    --smpl_model_path /path/to/SMPLX_MALE.npz
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Try to import smplx; if not available, provide instructions
try:
    import smplx
except ImportError:
    print("smplx not found. Install with: pip install smplx")
    exit(1)


def axis_angle_to_rotmat(aa, eps=1e-8):
    """
    Convert axis-angle to rotation matrix.
    
    Args:
        aa: axis-angle, shape (..., 3)
    
    Returns:
        rotation matrix, shape (..., 3, 3)
    """
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.clip(angle, eps, None)
    ax = axis[..., 0]
    ay = axis[..., 1]
    az = axis[..., 2]

    zero = np.zeros_like(ax)
    K = np.stack(
        [
            zero, -az, ay,
            az, zero, -ax,
            -ay, ax, zero,
        ],
        axis=-1,
    ).reshape(*ax.shape, 3, 3)

    I = np.eye(3, dtype=np.float32)
    I = np.broadcast_to(I, K.shape)

    sin = np.sin(angle)[..., 0][..., None, None]
    cos = np.cos(angle)[..., 0][..., None, None]
    R = I + sin * K + (1.0 - cos) * (K @ K)

    small = (angle[..., 0] < eps)
    if np.any(small):
        R[small] = I[small]
    return R


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SMPL mesh and root joint local coordinate frame"
    )
    parser.add_argument(
        "--smpl_npz",
        required=True,
        help="Path to SMPL motion .npz file (containing global_orient, body_pose, transl, betas)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Which frame to visualize (default: 0)",
    )
    parser.add_argument(
        "--smpl_model_path",
        default=None,
        help="Path to SMPLX model file (e.g., SMPLX_MALE.npz). If not provided, will auto-download.",
    )
    parser.add_argument(
        "--gender",
        default="male",
        choices=["male", "female", "neutral"],
        help="SMPL model gender (default: male)",
    )
    parser.add_argument(
        "--axis_length",
        type=float,
        default=0.2,
        help="Length of coordinate axes (default: 0.2)",
    )
    args = parser.parse_args()

    # Load SMPL motion data
    print(f"Loading SMPL motion from: {args.smpl_npz}")
    data = np.load(args.smpl_npz)
    
    global_orient = data["global_orient"].astype(np.float32)  # Shape: (T, 3)
    body_pose = data["body_pose"].astype(np.float32)          # Shape: (T, 69)
    transl = data["transl"].astype(np.float32)                # Shape: (T, 3)
    betas = data["betas"].astype(np.float32)                  # Shape: (10,)
    
    # Validate frame index
    n_frames = global_orient.shape[0]
    if args.frame < 0 or args.frame >= n_frames:
        print(f"Frame {args.frame} out of range [0, {n_frames-1}]")
        exit(1)
    
    print(f"Motion has {n_frames} frames, visualizing frame {args.frame}")
    print(f"  global_orient shape: {global_orient.shape}")
    print(f"  body_pose shape: {body_pose.shape}")
    print(f"  transl shape: {transl.shape}")
    print(f"  betas shape: {betas.shape}")
    
    # Extract parameters for this frame
    frame_global_orient = global_orient[args.frame:args.frame+1]  # Shape: (1, 3)
    frame_body_pose = body_pose[args.frame:args.frame+1]          # Shape: (1, 69)
    frame_transl = transl[args.frame:args.frame+1]                # Shape: (1, 3)
    
    print(f"\nFrame {args.frame} parameters:")
    print(f"  global_orient (root rotation, axis-angle): {frame_global_orient[0]}")
    print(f"  transl (root position): {frame_transl[0]}")
    
    # Load SMPL model
    print(f"\nLoading SMPL model (gender: {args.gender})...")
    
    # Determine model path
    if args.smpl_model_path:
        model_path = args.smpl_model_path
    else:
        # Try common locations
        possible_paths = [
            "./smpl_models",
            "/home/sigma/skibidi/motion-diffusion-model/visualize/joints2smpl/smpl_models",
        ]
        model_path = None
        for p in possible_paths:
            if Path(p).exists():
                model_path = p
                break
        if not model_path:
            print(f"SMPL model not found in: {possible_paths}")
            print(f"Provide path with --smpl_model_path")
            exit(1)
    
    print(f"Using SMPL model from: {model_path}")
    
    model = smplx.create(
        model_path=model_path,
        model_type="smpl",
        gender=args.gender,
        use_pca=False,
        batch_size=1,
    )
    
    # Create SMPL output
    print("Computing SMPL mesh...")
    with open("/dev/null", "w") as devnull:
        # Suppress warnings
        import sys
        old_stderr = sys.stderr
        sys.stderr = devnull
        
        smpl_output = model(
            betas=betas[None],
            body_pose=frame_body_pose,
            global_orient=frame_global_orient,
            transl=frame_transl,
        )
        
        sys.stderr = old_stderr
    
    vertices = smpl_output.vertices.detach().cpu().numpy()[0]  # Shape: (6890, 3)
    joints = smpl_output.joints.detach().cpu().numpy()[0]      # Shape: (24, 3) or (25, 3)
    
    print(f"Mesh vertices shape: {vertices.shape}")
    print(f"Joints shape: {joints.shape}")
    
    # Get pelvis position and root rotation
    pelvis_position = joints[0]  # Joint 0 is pelvis
    global_orient_aa = frame_global_orient[0]  # Axis-angle for root
    
    # Convert axis-angle to rotation matrix
    root_rotation_mat = axis_angle_to_rotmat(global_orient_aa[None])[0]  # Shape: (3, 3)
    
    print(f"\nPelvis position (world): {pelvis_position}")
    print(f"Root rotation (axis-angle): {global_orient_aa}")
    print(f"Root rotation (rotation matrix):\n{root_rotation_mat}")
    
    # Verify rotation matrix properties
    det = np.linalg.det(root_rotation_mat)
    print(f"Rotation matrix determinant (should be 1.0): {det}")
    orthogonal_check = np.linalg.norm(root_rotation_mat @ root_rotation_mat.T - np.eye(3))
    print(f"Orthogonality check (should be ~0): {orthogonal_check}")
    
    # Create coordinate axes in root's local frame
    # Axes length
    axis_len = args.axis_length
    
    # Unit vectors in local frame
    local_x = np.array([axis_len, 0, 0])
    local_y = np.array([0, axis_len, 0])
    local_z = np.array([0, 0, axis_len])
    
    # Transform to world frame using root rotation
    world_x = root_rotation_mat @ local_x + pelvis_position
    world_y = root_rotation_mat @ local_y + pelvis_position
    world_z = root_rotation_mat @ local_z + pelvis_position
    
    print(f"\nLocal coordinate frame at pelvis:")
    print(f"  Pelvis: {pelvis_position}")
    print(f"  X-axis endpoint (local X rotated): {world_x}")
    print(f"  Y-axis endpoint (local Y rotated): {world_y}")
    print(f"  Z-axis endpoint (local Z rotated): {world_z}")
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh vertices
    ax.scatter(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        c='cyan', s=1, alpha=0.3, label='Mesh vertices'
    )
    
    # Plot joints
    ax.scatter(
        joints[:, 0], joints[:, 1], joints[:, 2],
        c='purple', s=50, alpha=0.8, label='Joints'
    )
    
    # Highlight pelvis
    ax.scatter(
        [pelvis_position[0]], [pelvis_position[1]], [pelvis_position[2]],
        c='black', s=200, marker='o', label='Pelvis (root)', zorder=10
    )
    
    # Draw coordinate axes
    # X-axis (red)
    ax.plot(
        [pelvis_position[0], world_x[0]],
        [pelvis_position[1], world_x[1]],
        [pelvis_position[2], world_x[2]],
        'r-', linewidth=3, label='X (local frame)'
    )
    
    # Y-axis (green)
    ax.plot(
        [pelvis_position[0], world_y[0]],
        [pelvis_position[1], world_y[1]],
        [pelvis_position[2], world_y[2]],
        'g-', linewidth=3, label='Y (local frame)'
    )
    
    # Z-axis (blue)
    ax.plot(
        [pelvis_position[0], world_z[0]],
        [pelvis_position[1], world_z[1]],
        [pelvis_position[2], world_z[2]],
        'b-', linewidth=3, label='Z (local frame)'
    )
    
    # Draw world axes for reference (dashed, lighter)
    world_axis_len = axis_len * 0.5
    origin = pelvis_position
    ax.plot(
        [origin[0], origin[0] + world_axis_len],
        [origin[1], origin[1]],
        [origin[2], origin[2]],
        'r--', linewidth=1, alpha=0.3, label='X (world)'
    )
    ax.plot(
        [origin[0], origin[0]],
        [origin[1], origin[1] + world_axis_len],
        [origin[2], origin[2]],
        'g--', linewidth=1, alpha=0.3, label='Y (world)'
    )
    ax.plot(
        [origin[0], origin[0]],
        [origin[1], origin[1]],
        [origin[2], origin[2] + world_axis_len],
        'b--', linewidth=1, alpha=0.3, label='Z (world)'
    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        f'SMPL Mesh with Root Joint Local Frame\n'
        f'Frame {args.frame}, '
        f'Root rotation: {global_orient_aa}'
    )
    ax.legend(loc='upper right', fontsize=8)
    
    # Set equal aspect ratio
    all_points = np.vstack([vertices, pelvis_position])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min(),
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Show plot
    plt.tight_layout()
    print("\nDisplaying 3D plot...")
    plt.show()


if __name__ == "__main__":
    main()
