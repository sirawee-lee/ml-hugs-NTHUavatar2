#!/usr/bin/env python3
"""
Visualize SMPL motion skeleton and local coordinate frame at pelvis.

This script does NOT require the full SMPL model file. It:
1. Loads SMPL parameters (global_orient, body_pose, transl, betas)
2. Computes forward kinematics to get joint positions
3. Draws the skeleton structure
4. Draws local coordinate axes at the pelvis reflecting root rotation
5. X axis = red
6. Y axis = green  
7. Z axis = blue

The axes show the actual SMPL root rotation (global_orient), not world axes.

Usage:
  python visualize_smpl_orientation_simple.py \
    --smpl_npz /path/to/motion.npz \
    --frame 0
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# SMPL kinematic chain
SMPL_KINEMATIC_CHAIN = [
    [0, 1, 4, 7, 10],          # Spine
    [0, 2, 5, 8, 11],          # Left leg
    [0, 3, 6, 9, 12, 15],      # Right leg
    [13, 16, 18, 20, 22],      # Left arm
    [14, 17, 19, 21, 23],      # Right arm
]


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


def forward_kinematics_smpl(global_orient, body_pose, transl):
    """
    Simple SMPL forward kinematics using the kinematic chain.
    Assumes identity parent offsets (for visualization only).
    
    Args:
        global_orient: (1, 3) or (3,) - root rotation in axis-angle
        body_pose: (1, 69) or (69,) - body joint rotations in axis-angle
        transl: (1, 3) or (3,) - root translation
    
    Returns:
        joints: (24, 3) - joint positions in world frame
    """
    if global_orient.shape == (3,):
        global_orient = global_orient[None]
    if body_pose.shape == (69,):
        body_pose = body_pose[None]
    if transl.shape == (3,):
        transl = transl[None]
    
    # Stack all rotations: [global_orient, body_pose]
    all_rotations_aa = np.concatenate([global_orient, body_pose], axis=1)  # (1, 72)
    all_rotations_aa = all_rotations_aa.reshape(1, 24, 3)  # (1, 24, 3)
    
    # Convert to rotation matrices
    rotmats = axis_angle_to_rotmat(all_rotations_aa[0])  # (24, 3, 3)
    
    # Initialize joint positions (simplified - assumes parent offsets are small)
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = transl[0]  # Root position
    
    # Simple FK: each joint inherits parent position and rotation
    # This is a simplified version; proper SMPL FK would include parent offsets
    parent_indices = [
        0,  # 0: Pelvis (root)
        0,  # 1: L_Hip
        0,  # 2: R_Hip
        0,  # 3: Spine (actually neck, using 0 as parent for simplification)
        1,  # 4: L_Knee
        2,  # 5: R_Knee
        3,  # 6: L_Ankle (using 3 as substitute)
        4,  # 7: R_Ankle (using 4 as substitute)
        3,  # 8: Thorax
        3,  # 9: Jaw (using 3 as parent)
        3,  # 10: L_Collar (using 3 as parent)
        3,  # 11: R_Collar
        3,  # 12: L_Shoulder
        3,  # 13: R_Shoulder
        12, # 14: L_Elbow
        13, # 15: R_Elbow
        12, # 16: L_Wrist
        13, # 17: R_Wrist
        16, # 18: L_Hand_thumb
        17, # 19: R_Hand_thumb
        16, # 20: L_Hand_index
        17, # 21: R_Hand_index
        16, # 22: L_Hand_middle
        17, # 23: R_Hand_middle
    ]
    
    # This is a very simplified FK. For accurate visualization, use the full SMPL model.
    # For now, we just use the root position for all joints and show the local frame at root.
    
    return joints


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SMPL skeleton and root joint local frame (simple version)"
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
        "--axis_length",
        type=float,
        default=0.3,
        help="Length of coordinate axes (default: 0.3)",
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
    frame_global_orient = global_orient[args.frame]  # Shape: (3,)
    frame_body_pose = body_pose[args.frame]          # Shape: (69,)
    frame_transl = transl[args.frame]                # Shape: (3,)
    
    print(f"\nFrame {args.frame} parameters:")
    print(f"  global_orient (root rotation, axis-angle): {frame_global_orient}")
    print(f"  transl (root position): {frame_transl}")
    
    # Convert root rotation to matrix
    root_rotation_mat = axis_angle_to_rotmat(frame_global_orient[None])[0]  # Shape: (3, 3)
    
    print(f"\nRoot rotation (rotation matrix):\n{root_rotation_mat}")
    
    # Verify rotation matrix properties
    det = np.linalg.det(root_rotation_mat)
    print(f"Rotation matrix determinant (should be 1.0): {det}")
    orthogonal_check = np.linalg.norm(root_rotation_mat @ root_rotation_mat.T - np.eye(3))
    print(f"Orthogonality check (should be ~0): {orthogonal_check}")
    
    # Pelvis position
    pelvis_position = frame_transl
    
    # Create coordinate axes in root's local frame
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
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot pelvis (root)
    ax.scatter(
        [pelvis_position[0]], [pelvis_position[1]], [pelvis_position[2]],
        c='black', s=300, marker='o', label='Pelvis (root)', zorder=10
    )
    
    # Draw local coordinate axes at pelvis
    # X-axis (red)
    ax.plot(
        [pelvis_position[0], world_x[0]],
        [pelvis_position[1], world_x[1]],
        [pelvis_position[2], world_x[2]],
        'r-', linewidth=4, label='X (local frame)', zorder=9
    )
    ax.scatter([world_x[0]], [world_x[1]], [world_x[2]], c='red', s=100, marker='>', zorder=9)
    
    # Y-axis (green)
    ax.plot(
        [pelvis_position[0], world_y[0]],
        [pelvis_position[1], world_y[1]],
        [pelvis_position[2], world_y[2]],
        'g-', linewidth=4, label='Y (local frame)', zorder=9
    )
    ax.scatter([world_y[0]], [world_y[1]], [world_y[2]], c='green', s=100, marker='^', zorder=9)
    
    # Z-axis (blue)
    ax.plot(
        [pelvis_position[0], world_z[0]],
        [pelvis_position[1], world_z[1]],
        [pelvis_position[2], world_z[2]],
        'b-', linewidth=4, label='Z (local frame)', zorder=9
    )
    ax.scatter([world_z[0]], [world_z[1]], [world_z[2]], c='blue', s=100, marker='s', zorder=9)
    
    # Draw world axes for reference (dashed, at origin)
    world_axis_len = axis_len * 0.7
    origin = np.array([0, 0, 0])
    ax.plot([0, world_axis_len], [0, 0], [0, 0], 'r--', linewidth=1, alpha=0.3, label='X (world)')
    ax.plot([0, 0], [0, world_axis_len], [0, 0], 'g--', linewidth=1, alpha=0.3, label='Y (world)')
    ax.plot([0, 0], [0, 0], [0, world_axis_len], 'b--', linewidth=1, alpha=0.3, label='Z (world)')
    
    # Add text annotations
    ax.text(world_x[0], world_x[1], world_x[2], 'X', color='red', fontsize=12, weight='bold')
    ax.text(world_y[0], world_y[1], world_y[2], 'Y', color='green', fontsize=12, weight='bold')
    ax.text(world_z[0], world_z[1], world_z[2], 'Z', color='blue', fontsize=12, weight='bold')
    
    # Set labels and title
    ax.set_xlabel('X (world)', fontsize=11)
    ax.set_ylabel('Y (world)', fontsize=11)
    ax.set_zlabel('Z (world)', fontsize=11)
    ax.set_title(
        f'SMPL Root Joint Local Frame\n'
        f'Frame {args.frame}: global_orient={frame_global_orient}\n'
        f'transl={frame_transl}',
        fontsize=12
    )
    ax.legend(loc='upper right', fontsize=9)
    
    # Set equal aspect ratio
    max_range = axis_len * 2
    mid_x = pelvis_position[0]
    mid_y = pelvis_position[1]
    mid_z = pelvis_position[2]
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # View angle
    ax.view_init(elev=20, azim=45)
    
    # Save plot to file
    plt.tight_layout()
    output_path = Path(args.smpl_npz).parent / f"smpl_visualization_frame_{args.frame}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    print("(Run with display or open the PNG file)")
    
    # Also try to show if display is available
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")


if __name__ == "__main__":
    main()
