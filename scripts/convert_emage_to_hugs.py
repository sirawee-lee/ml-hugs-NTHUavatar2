#!/usr/bin/env python3
"""
Convert EMAGE (PantoMatrix) beat-format .npz to HUGS SMPL .npz.

EMAGE outputs SMPL-X (55 joints × 3 = 165 axis-angle params):
  poses      : (T, 165)  joint order: [global(0), body(1-21), jaw(22), eyes(23-24), LH(25-39), RH(40-54)]
  betas      : (300,)    SMPL-X shape coefficients
  expressions: (T, 100)  FLAME expression (not used by HUGS)
  trans      : (T, 3)    root translation

HUGS expects SMPL (24 joints):
  global_orient: (T, 3)   root axis-angle
  body_pose    : (T, 69)  23 body joints × 3 axis-angle
  transl       : (T, 3)   root translation
  betas        : (10,)    SMPL shape coefficients

Mapping:
  global_orient = poses[:, 0:3]
  body_pose     = [poses[:, 3:66]  ← 21 SMPLX body joints]
                  + [zeros(T, 6)   ← 2 dummy joints to reach SMPL's 23]
  transl        = trans
  betas         = betas[:10]

Usage:
  python scripts/convert_emage_to_hugs.py \\
      --input  /path/to/emage_output.npz \\
      --output /path/to/hugs_motion.npz  \\
      [--betas /path/to/betas.npy]
"""

import argparse
import numpy as np
from pathlib import Path


def convert_emage_to_hugs(
    emage_path: str,
    output_path: str,
    betas_override: str = None,
) -> dict:
    print(f"Loading EMAGE output: {emage_path}")
    data = np.load(emage_path, allow_pickle=True)

    poses = data["poses"].astype(np.float32)       # (T, 165)
    trans = data["trans"].astype(np.float32)        # (T, 3)
    raw_betas = data["betas"].astype(np.float32)    # (300,) or (T, 300)

    T = poses.shape[0]
    assert poses.shape[1] == 165, (
        f"Expected poses shape (T, 165) for SMPLX-55 joints, got {poses.shape}"
    )

    # ── global orientation (joint 0) ────────────────────────────────────────
    global_orient = poses[:, 0:3]                   # (T, 3)

    # ── body pose: SMPLX joints 1-21 (63 dims) padded to SMPL 23 joints (69) ─
    smplx_body = poses[:, 3:66]                     # (T, 63)  joints 1-21
    padding    = np.zeros((T, 6), dtype=np.float32) # (T,  6)  dummy joints 22-23
    body_pose  = np.concatenate([smplx_body, padding], axis=-1)  # (T, 69)

    # ── translation ─────────────────────────────────────────────────────────
    transl = trans                                  # (T, 3)

    # ── betas: SMPLX has 300 coefficients, SMPL uses 10 ────────────────────
    if betas_override is not None:
        print(f"Using betas from: {betas_override}")
        betas_data = np.load(betas_override)
        if isinstance(betas_data, np.lib.npyio.NpzFile):
            betas_data = betas_data["betas"]
        betas = betas_data.flatten()[:10].astype(np.float32)
    else:
        if raw_betas.ndim == 2:
            raw_betas = raw_betas[0]   # take first frame if (T, 300)
        betas = raw_betas[:10]        # first 10 coefficients

    print(f"Conversion summary:")
    print(f"  T (frames)    : {T}")
    print(f"  global_orient : {global_orient.shape}")
    print(f"  body_pose     : {body_pose.shape}")
    print(f"  transl        : {transl.shape}")
    print(f"  betas         : {betas.shape}")

    out = {
        "global_orient": global_orient,
        "body_pose":      body_pose,
        "transl":         transl,
        "betas":          betas,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **out)
    print(f"Saved HUGS-format motion: {output_path}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert EMAGE beat-format .npz to HUGS SMPL .npz"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="EMAGE output .npz (beat format)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output HUGS SMPL .npz path")
    parser.add_argument("--betas",  "-b", default=None,
                        help="Optional .npy/.npz with SMPL betas to override")
    args = parser.parse_args()
    convert_emage_to_hugs(args.input, args.output, args.betas)


if __name__ == "__main__":
    main()
