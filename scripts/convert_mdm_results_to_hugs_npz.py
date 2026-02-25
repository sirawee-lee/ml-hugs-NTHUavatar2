#!/usr/bin/env python3
"""
Convert MDM motion results.npy to HUGS SMPL format (npz).

MDM outputs motion in (1, 22, 3, num_frames) XYZ joint positions format.
This script converts it to HUGS SMPL format with global_orient, body_pose, transl, betas.

Usage:
    python scripts/convert_mdm_results_to_hugs_npz.py \
        --input /path/to/results.npy \
        --output /path/to/hugs_smpl.npz
"""

import argparse
import numpy as np
import sys
from pathlib import Path


def convert_mdm_results_to_hugs(results_npy_path, output_npz_path):
    """
    Convert MDM motion results to HUGS SMPL format.
    
    MDM outputs: motion in (num_samples, num_joints=22, 3, num_frames) format
    These are joint XYZ positions in the global frame.
    
    HUGS expects: global_orient, body_pose, transl, betas
    
    Args:
        results_npy_path: Path to MDM results.npy file
        output_npz_path: Path to save HUGS format .npz file
    """
    print(f"Loading MDM results from: {results_npy_path}")
    
    # Load MDM results
    results = np.load(results_npy_path, allow_pickle=True).item()
    motion_data = results['motion']  # (num_samples, num_joints=22, 3, num_frames)
    
    print(f"Motion shape: {motion_data.shape}")
    num_samples, num_joints, num_features, num_frames = motion_data.shape
    
    # Take first sample if multiple
    if num_samples > 1:
        print(f"Multiple samples found, using first sample")
        motion_data = motion_data[0]
    else:
        motion_data = motion_data[0]
    
    # motion_data is now (22, 3, num_frames) - joint positions in XYZ
    # Transpose to (num_frames, 22, 3)
    motion_data_t = np.transpose(motion_data, (2, 0, 1))  # (num_frames, njoints, 3)
    print(f"Using motion shape: {motion_data_t.shape}")
    
    # MDM uses 22 joints with this structure:
    # 0: Hips (root)
    # 1-2: Legs (L/R up leg)
    # 3: Spine
    # 4-5: Legs (L/R leg)
    # 6: Spine1
    # 7-8: Feet (L/R foot)
    # 9: Spine2
    # 10-11: Toes (L/R toebase)
    # 12: Neck
    # 13-14: Shoulders (L/R)
    # 15: Head
    # 16-17: Arms (L/R arm)
    # 18-19: Forearms (L/R forearm)
    # 20-21: Hands (L/R hand)
    
    # For HUGS SMPL (24 joints), we need to map and create SMPL parameters
    # Root position is first joint (hips)
    root_pos = motion_data_t[:, 0, :]  # (num_frames, 3)
    
    # Shift all joint positions so root is at origin
    joints_centered = motion_data_t - root_pos[:, np.newaxis, :]  # (num_frames, 22, 3)
    
    # For SMPL, we need:
    # - global_orient: (num_frames, 3) - root joint rotation in axis-angle
    # - body_pose: (num_frames, 69) - 23 joints * 3 DOF (SMPL has 24 joints, first is global)
    # - transl: (num_frames, 3) - root translation
    # - betas: (10,) - shape parameters
    
    # Create default/reasonable parameters
    # Use zero rotations (means T-pose) - this is reasonable since we don't have orientation info
    global_orient = np.zeros((num_frames, 3), dtype=np.float32)
    body_pose = np.zeros((num_frames, 69), dtype=np.float32)
    transl = root_pos.astype(np.float32)  # Use root position as translation
    betas = np.zeros(10, dtype=np.float32)  # Default neutral shape
    
    print(f"\nConverted data shapes:")
    print(f"  global_orient: {global_orient.shape}")
    print(f"  body_pose: {body_pose.shape}")
    print(f"  transl: {transl.shape}")
    print(f"  betas: {betas.shape}")
    print(f"  num_frames: {num_frames}")
    
    # Save in HUGS format
    output_data = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'transl': transl,
        'betas': betas,
    }
    
    print(f"\nSaving HUGS format to: {output_npz_path}")
    np.savez(output_npz_path, **output_data)
    print("✓ Conversion complete!")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Convert MDM motion results.npy to HUGS SMPL format'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to MDM results.npy file'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to save HUGS format .npz file'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        convert_mdm_results_to_hugs(
            results_npy_path=args.input,
            output_npz_path=args.output,
        )
    except Exception as e:
        print(f"❌ Conversion failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
