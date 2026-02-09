#!/usr/bin/env python3
"""
Master pipeline script: Convert MDM-generated SMPL motion to HUGS-rendered video.

Workflow:
  1. Load MDM SMPL motion (.npz)
  2. Rotate to HUGS coordinates (RX +90°, RZ +180°)
  3. Optionally center and offset translation
  4. Run HUGS rendering with custom motion
  5. Save run record with metadata

Usage:
  python scripts/run_mdm2hugs.py \
    --input_npz /path/to/smpl_motion.npz \
    --out_dir ./output_mdm2hugs \
    --scene bike \
    --center \
    --tz 0.5
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Configuration
PYTHON_BIN = "/home/sigma/anaconda3/envs/hugs/bin/python"
HUGS_REPO = "/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar"
ROTATE_SCRIPT = f"{HUGS_REPO}/scripts/rotate_hugs_motion_v2.py"

# HUGS config and checkpoints
HUGS_CONFIG = f"{HUGS_REPO}/cfg_files/release/neuman/hugs_human_scene.yaml"
PRETRAINED_DIR = f"{HUGS_REPO}/output/pretrained_models"

# Default scene config (can be extended for other scenes)
SCENE_CONFIGS = {
    "bike": {
        "human_ckpt": f"{PRETRAINED_DIR}/bike/human_final.pth",
        "scene_ckpt": f"{PRETRAINED_DIR}/bike/scene_final.pth",
    },
    "citron": {
        "human_ckpt": f"{PRETRAINED_DIR}/citron/human_final.pth",
        "scene_ckpt": f"{PRETRAINED_DIR}/citron/scene_final.pth",
    },
    "jogging": {
        "human_ckpt": f"{PRETRAINED_DIR}/jogging/human_final.pth",
        "scene_ckpt": f"{PRETRAINED_DIR}/jogging/scene_final.pth",
    },
}


def run_command(cmd, description=""):
    """Execute a shell command and return exit code."""
    print(f"\n{'='*80}")
    if description:
        print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=HUGS_REPO)
    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        return False
    print(f"\n✓ Command succeeded")
    return True


def rotate_motion(input_npz, output_npz, center=False, tz=0.0):
    """Rotate SMPL motion from MDM to HUGS coordinates."""
    cmd = [
        PYTHON_BIN,
        ROTATE_SCRIPT,
        "--input", str(input_npz),
        "--output", str(output_npz),
        "--rx", "90",    # MDM→HUGS: rotate X +90°
        "--rz", "180",   # MDM→HUGS: rotate Z +180°
    ]
    
    if center:
        cmd.append("--center")
    
    if tz != 0.0:
        cmd.extend(["--tz", str(tz)])
    
    return run_command(cmd, "Rotate SMPL motion (MDM → HUGS coordinates)")


def render_hugs(rotated_npz, scene, out_dir):
    """Render HUGS animation with rotated motion."""
    if scene not in SCENE_CONFIGS:
        print(f"❌ Unknown scene: {scene}")
        print(f"   Available scenes: {', '.join(SCENE_CONFIGS.keys())}")
        return False
    
    config = SCENE_CONFIGS[scene]
    
    cmd = [
        PYTHON_BIN,
        "main.py",
        "--cfg_file", HUGS_CONFIG,
        f"dataset.seq={scene}",
        "eval=true",
        f"human.ckpt={config['human_ckpt']}",
        f"scene.ckpt={config['scene_ckpt']}",
        f"custom_motion_path={rotated_npz}",
    ]
    
    return run_command(cmd, f"Render HUGS animation (scene={scene})")


def save_record(record_path, record_data):
    """Save execution record as JSON."""
    with open(record_path, "w") as f:
        json.dump(record_data, f, indent=2, default=str)
    print(f"\n✓ Saved run record to: {record_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline: MDM SMPL → HUGS render",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic rendering with default settings
  python scripts/run_mdm2hugs.py \\
    --input_npz /path/to/smpl.npz \\
    --out_dir ./output_mdm2hugs

  # With translation centering and forward offset
  python scripts/run_mdm2hugs.py \\
    --input_npz /path/to/smpl.npz \\
    --out_dir ./output_mdm2hugs \\
    --scene citron \\
    --center \\
    --tz 0.5
        """,
    )
    
    parser.add_argument(
        "--input_npz",
        required=True,
        help="Path to MDM-generated SMPL motion (.npz)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for rotated motion and record",
    )
    parser.add_argument(
        "--scene",
        default="bike",
        choices=list(SCENE_CONFIGS.keys()),
        help="Scene to render (default: bike)",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center translation to mean 0 before rendering",
    )
    parser.add_argument(
        "--tz",
        type=float,
        default=0.0,
        help="Forward offset (Z-axis) after centering (default: 0.0)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_npz = Path(args.input_npz)
    if not input_npz.exists():
        print(f"❌ Input file not found: {input_npz}")
        sys.exit(1)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file name
    stem = input_npz.stem
    rotated_npz = out_dir / f"{stem}_mdm2hugs.npz"
    record_path = out_dir / "run_record.json"
    
    print(f"\n{'='*80}")
    print("MDM → HUGS Pipeline")
    print(f"{'='*80}")
    print(f"Input:      {input_npz}")
    print(f"Output dir: {out_dir}")
    print(f"Scene:      {args.scene}")
    print(f"Center:     {args.center}")
    print(f"TZ offset:  {args.tz}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    # Step 1: Rotate motion
    print("\n[1/2] Rotating SMPL motion from MDM to HUGS coordinates...")
    if not rotate_motion(input_npz, rotated_npz, center=args.center, tz=args.tz):
        sys.exit(1)
    
    # Step 2: Render with HUGS
    print("\n[2/2] Rendering HUGS animation...")
    if not render_hugs(rotated_npz, args.scene, out_dir):
        sys.exit(1)
    
    end_time = datetime.now()
    
    # Save record
    record_data = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "input_npz": str(input_npz),
        "output_npz": str(rotated_npz),
        "output_dir": str(out_dir),
        "scene": args.scene,
        "rotation": {
            "rx_degrees": 90,
            "rz_degrees": 180,
        },
        "translation": {
            "center": args.center,
            "tz_offset": args.tz,
        },
        "hugs_repo": HUGS_REPO,
        "hugs_config": HUGS_CONFIG,
        "scene_checkpoints": SCENE_CONFIGS[args.scene],
    }
    
    save_record(record_path, record_data)
    
    print(f"\n{'='*80}")
    print("✓ Pipeline completed successfully!")
    print(f"{'='*80}")
    print(f"Duration: {record_data['duration_seconds']:.1f} seconds")
    print(f"Rotated motion: {rotated_npz}")
    print(f"Run record:     {record_path}")


if __name__ == "__main__":
    main()
