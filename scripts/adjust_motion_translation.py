#!/usr/bin/env python3
"""
Adjust translation of motion to fit bike scene view.
"""

import numpy as np
import argparse

def adjust_translation(input_path, output_path, center_y=0.0, scale_factor=1.0, rotate_y=0.0):
    """
    Adjust translation and rotation to fit the camera view.
    
    Args:
        input_path: Path to input motion .npz file
        output_path: Path to save adjusted motion
        center_y: Target Y center position (height)
        scale_factor: Scale factor for translation
        rotate_y: Rotation around Y axis in degrees (e.g., 180 to flip)
    """
    # Load motion data
    print(f"Loading motion from: {input_path}")
    data = np.load(input_path)
    
    global_orient = data['global_orient'].copy()  # Make a copy
    body_pose = data['body_pose']
    transl = data['transl'].copy()  # Make a copy
    betas = data['betas']
    
    print(f"\nOriginal translation range:")
    print(f"  X: {transl[:, 0].min():.3f} to {transl[:, 0].max():.3f}")
    print(f"  Y: {transl[:, 1].min():.3f} to {transl[:, 1].max():.3f}")
    print(f"  Z: {transl[:, 2].min():.3f} to {transl[:, 2].max():.3f}")
    
    # Rotate around Y axis if requested
    if rotate_y != 0.0:
        print(f"\nRotating {rotate_y} degrees around Y axis...")
        angle_rad = np.radians(rotate_y)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Rotation matrix around Y axis
        # [cos  0  sin]   [x]
        # [ 0   1   0 ] * [y]
        # [-sin 0  cos]   [z]
        
        # Rotate translation
        x_new = transl[:, 0] * cos_angle + transl[:, 2] * sin_angle
        z_new = -transl[:, 0] * sin_angle + transl[:, 2] * cos_angle
        transl[:, 0] = x_new
        transl[:, 2] = z_new
        
        # Rotate global_orient (root orientation)
        # Convert axis-angle to rotation, apply Y rotation, convert back
        from scipy.spatial.transform import Rotation as R
        for i in range(len(global_orient)):
            # Current orientation as rotation
            current_rot = R.from_rotvec(global_orient[i])
            # Y-axis rotation
            y_rot = R.from_euler('y', rotate_y, degrees=True)
            # Combine rotations
            new_rot = y_rot * current_rot
            # Back to axis-angle
            global_orient[i] = new_rot.as_rotvec()
    
    # Adjust translation
    # 1. Scale the motion
    transl = transl * scale_factor
    
    # 2. Center Y around target (keep relative motion but shift center)
    current_y_mean = transl[:, 1].mean()
    y_offset = center_y - current_y_mean
    transl[:, 1] += y_offset
    
    # 3. Center X and Z at origin (bike is typically centered)
    transl[:, 0] -= transl[:, 0].mean()
    transl[:, 2] -= transl[:, 2].mean()
    
    print(f"\nAdjusted translation range:")
    print(f"  X: {transl[:, 0].min():.3f} to {transl[:, 0].max():.3f}")
    print(f"  Y: {transl[:, 1].min():.3f} to {transl[:, 1].max():.3f}")
    print(f"  Z: {transl[:, 2].min():.3f} to {transl[:, 2].max():.3f}")
    
    # Save adjusted motion
    output_data = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'transl': transl.astype(np.float32),
        'betas': betas,
    }
    
    print(f"\nSaving adjusted motion to: {output_path}")
    np.savez(output_path, **output_data)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input motion .npz file')
    parser.add_argument('--output', required=True, help='Output motion .npz file')
    parser.add_argument('--center-y', type=float, default=0.0, 
                        help='Target Y center (default: 0.0 = ground level)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for translation (default: 1.0)')
    parser.add_argument('--rotate-y', type=float, default=0.0,
                        help='Rotation around Y axis in degrees (default: 0.0, use 180 to flip)')
    
    args = parser.parse_args()
    adjust_translation(args.input, args.output, args.center_y, args.scale, args.rotate_y)
