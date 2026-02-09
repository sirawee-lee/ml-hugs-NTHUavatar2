#!/usr/bin/env python3
"""
Flip motion direction by inverting Z axis and global orientation.
Simple approach: just flip Z coordinate and rotate global_orient 180° around Y.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def flip_motion_direction(input_path, output_path, center_y=1.0):
    """
    Flip motion to face opposite direction
    
    Args:
        input_path: Input motion .npz file
        output_path: Output motion .npz file  
        center_y: Y center position (default 1.0 for bike scene)
    """
    print(f"Loading motion from: {input_path}")
    data = np.load(input_path)
    
    global_orient = data['global_orient'].copy()
    body_pose = data['body_pose'].copy()
    transl = data['transl'].copy()
    betas = data['betas'].copy()
    
    print(f"\nOriginal translation:")
    print(f"  X: {transl[:, 0].min():.3f} to {transl[:, 0].max():.3f}")
    print(f"  Y: {transl[:, 1].min():.3f} to {transl[:, 1].max():.3f}")
    print(f"  Z: {transl[:, 2].min():.3f} to {transl[:, 2].max():.3f}")
    
    # Step 1: Flip Z direction
    print("\nFlipping Z direction...")
    transl[:, 2] = -transl[:, 2]
    
    # Step 2: Rotate global orientation 180° around Y axis
    print("Rotating global orientation 180° around Y...")
    rot_y_180 = R.from_euler('y', 180, degrees=True)
    
    for i in range(len(global_orient)):
        # Current orientation
        current_rot = R.from_rotvec(global_orient[i])
        # Combine with 180° Y rotation
        flipped_rot = rot_y_180 * current_rot
        # Store back
        global_orient[i] = flipped_rot.as_rotvec()
    
    # Step 3: Center Y at desired height
    current_y_mean = transl[:, 1].mean()
    y_offset = center_y - current_y_mean
    transl[:, 1] += y_offset
    
    # Step 4: Center X and Z at origin
    transl[:, 0] -= transl[:, 0].mean()
    transl[:, 2] -= transl[:, 2].mean()
    
    print(f"\nAdjusted translation:")
    print(f"  X: {transl[:, 0].min():.3f} to {transl[:, 0].max():.3f}")
    print(f"  Y: {transl[:, 1].min():.3f} to {transl[:, 1].max():.3f}")
    print(f"  Z: {transl[:, 2].min():.3f} to {transl[:, 2].max():.3f}")
    
    # Save adjusted motion
    np.savez(output_path,
             global_orient=global_orient,
             body_pose=body_pose,
             transl=transl,
             betas=betas)
    
    print(f"\nSaved flipped motion to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flip motion direction')
    parser.add_argument('--input', required=True, help='Input motion .npz file')
    parser.add_argument('--output', required=True, help='Output motion .npz file')
    parser.add_argument('--center-y', type=float, default=1.0, help='Y center position')
    
    args = parser.parse_args()
    
    flip_motion_direction(args.input, args.output, args.center_y)
