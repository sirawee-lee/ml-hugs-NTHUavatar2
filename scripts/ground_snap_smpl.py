#!/usr/bin/env python3
"""
Ground-snap a HUGS motion .npz using SMPL forward kinematics.

Problem: naive ground-snap using min(root_Z) gives inconsistent ground levels
across motions because the pelvis height above the feet varies per motion.
E.g. a squat has lower root Z than dancing, so root-based snapping places
the ground at different heights relative to the actual feet.

Fix: run SMPL FK to get real foot joint positions, snap those to a target
ground height instead.

Usage:
  python scripts/ground_snap_smpl.py \
    --input  motion.npz \
    --output motion_grounded.npz \
    [--ground 0.0]   # target ground height (default 0.0)
    [--smpl_model /path/to/SMPL_NEUTRAL.pkl]
    [--device cpu]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import smplx


# smplx.create(model_path, model_type="smpl") looks for <model_path>/smpl/SMPL_NEUTRAL.pkl
SMPL_MODEL_DIR = Path(__file__).resolve().parent.parent / "data"  # contains smpl/ subdir

# SMPL 24-joint indices for feet
LEFT_FOOT  = 10
RIGHT_FOOT = 11


def ground_snap(input_path: Path, output_path: Path, ground: float,
                smpl_model_path: Path, device: str):
    data = np.load(str(input_path))
    global_orient = data["global_orient"].astype(np.float32)  # (T, 3)
    body_pose     = data["body_pose"].astype(np.float32)       # (T, 69)
    transl        = data["transl"].astype(np.float32)          # (T, 3)
    betas         = data["betas"].astype(np.float32)            # (10,)

    T = global_orient.shape[0]
    dev = torch.device(device)

    smpl = smplx.create(
        str(smpl_model_path),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=T,
    ).to(dev)

    with torch.no_grad():
        output = smpl(
            global_orient=torch.from_numpy(global_orient).to(dev),
            body_pose=torch.from_numpy(body_pose).to(dev),
            transl=torch.from_numpy(transl).to(dev),
            betas=torch.from_numpy(betas).unsqueeze(0).expand(T, -1).to(dev),
        )

    joints = output.joints.cpu().numpy()  # (T, 45, 3)

    # Z is the vertical axis in HUGS world (after rx=90 rotation).
    foot_z = joints[:, [LEFT_FOOT, RIGHT_FOOT], 2]   # (T, 2)
    min_foot_z = float(foot_z.min())

    shift = ground - min_foot_z
    print(f"Min foot Z = {min_foot_z:.4f}  →  shift Z by {shift:+.4f} to reach ground={ground}")

    transl_new = transl.copy()
    transl_new[:, 2] += shift

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl_new,
        betas=betas,
    )
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--ground", type=float, default=0.0,
                        help="Target Z for lowest foot point (default 0.0)")
    parser.add_argument("--smpl_model", default=str(SMPL_MODEL_DIR))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix("") \
                  .parent / (input_path.stem + "_grounded.npz")
    ground_snap(input_path, output_path, args.ground,
                Path(args.smpl_model), args.device)


if __name__ == "__main__":
    main()
