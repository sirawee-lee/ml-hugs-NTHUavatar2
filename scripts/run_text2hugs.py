#!/usr/bin/env python3
"""
End-to-end pipeline: TEXT PROMPT → MDM → HUGS SMPL → Rotate → HUGS Render → Video

Workflow:
  1. Run MDM sampling from text prompt
  2. Find/convert MDM output to HUGS SMPL npz format
  3. Rotate SMPL motion to HUGS coordinate system (RX=+90°, RZ=+180°)
  4. Run HUGS rendering with custom motion
  5. Extract final video and save to output directory

Usage:
  python scripts/run_text2hugs.py \
    --prompt "a person jumps" \
    --scene bike \
    --out_root ./output_text2hugs \
    --center \
    --tz 1.0

  # Dry run (prints commands without executing)
  python scripts/run_text2hugs.py \
    --prompt "a person walks" \
    --out_root ./test \
    --dry_run
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# Default paths
DEFAULT_MDM_REPO = Path("/home/sigma/skibidi/motion-diffusion-model")
DEFAULT_HUGS_REPO = Path("/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar")
DEFAULT_MDM_PY = Path("/home/sigma/anaconda3/envs/mdm/bin/python")
DEFAULT_HUGS_PY = Path("/home/sigma/anaconda3/envs/hugs/bin/python")

# HUGS scene configurations
SCENE_CONFIGS = {
    "bike": {
        "human_ckpt": "output/pretrained_models/bike/human_final.pth",
        "scene_ckpt": "output/pretrained_models/bike/scene_final.pth",
    },
    "citron": {
        "human_ckpt": "output/pretrained_models/citron/human_final.pth",
        "scene_ckpt": "output/pretrained_models/citron/scene_final.pth",
    },
    "jogging": {
        "human_ckpt": "output/pretrained_models/jogging/human_final.pth",
        "scene_ckpt": "output/pretrained_models/jogging/scene_final.pth",
    },
}


def slugify(text: str, max_len: int = 30) -> str:
    """Convert text to filesystem-safe slug."""
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '_', slug)
    return slug[:max_len].strip('_')


def run_command(
    cmd: List[str],
    description: str,
    log_file: Optional[Path] = None,
    cwd: Optional[Path] = None,
    dry_run: bool = False,
) -> bool:
    """Execute a command and log output."""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    if cwd:
        print(f"Working dir: {cwd}")
    if log_file:
        print(f"Log: {log_file}")
    print(f"{'='*80}")
    
    if dry_run:
        print("[DRY RUN] Command not executed")
        return True
    
    try:
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        else:
            result = subprocess.run(cmd, cwd=cwd, check=True)
        
        print(f"✓ Command succeeded")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if log_file and log_file.exists():
            print(f"See log file: {log_file}")
        return False


def find_file(root: Path, filename: str) -> Optional[Path]:
    """Recursively find a file by name under root."""
    matches = list(root.rglob(filename))
    if matches:
        return matches[0]  # Return first match
    return None


def find_newest_mp4(search_root: Path, after_time: Optional[float] = None) -> Optional[Path]:
    """Find the most recent .mp4 file (optionally created after a timestamp)."""
    mp4_files = list(search_root.rglob("*.mp4"))
    
    if after_time:
        mp4_files = [f for f in mp4_files if f.stat().st_mtime > after_time]
    
    if not mp4_files:
        return None
    
    # Return the newest by modification time
    newest = max(mp4_files, key=lambda p: p.stat().st_mtime)
    return newest


def build_mdm_cmd(
    prompt: str,
    out_dir: Path,
    mdm_repo: Path,
    mdm_py: Path,
    seed: int,
    steps: int,
) -> List[str]:
    """
    Build MDM sampling command.
    
    MDM automatically creates output in:
    save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed{seed}_{prompt}/
    
    The output includes hugs_smpl_original.npz which is ready for HUGS rendering.
    """
    cmd = [
        str(mdm_py),
        "-m", "sample.generate",
        "--model_path", str(mdm_repo / "save/humanml_enc_512_50steps/model000750000.pt"),
        "--text_prompt", prompt,
        "--num_samples", "1",
        "--num_repetitions", "1",
        "--guidance_param", "2.5",
        "--seed", str(seed),
    ]
    
    return cmd



def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: Text → MDM → HUGS render",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/run_text2hugs.py \\
    --prompt "a person jumps" \\
    --out_root ./output_text2hugs

  # With custom scene and translation
  python scripts/run_text2hugs.py \\
    --prompt "a person walks forward" \\
    --scene citron \\
    --out_root ./output \\
    --center \\
    --tz 1.5

  # Dry run to test commands
  python scripts/run_text2hugs.py \\
    --prompt "test" \\
    --out_root ./test \\
    --dry_run
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for MDM motion generation",
    )
    parser.add_argument(
        "--out_root",
        required=True,
        type=Path,
        help="Root output directory for all pipeline artifacts",
    )
    
    # Scene and rendering
    parser.add_argument(
        "--scene",
        default="bike",
        choices=list(SCENE_CONFIGS.keys()),
        help="HUGS scene to render (default: bike)",
    )
    
    # Repository and Python paths
    parser.add_argument(
        "--mdm_repo",
        type=Path,
        default=DEFAULT_MDM_REPO,
        help=f"Path to MDM repository (default: {DEFAULT_MDM_REPO})",
    )
    parser.add_argument(
        "--hugs_repo",
        type=Path,
        default=DEFAULT_HUGS_REPO,
        help=f"Path to HUGS repository (default: {DEFAULT_HUGS_REPO})",
    )
    parser.add_argument(
        "--mdm_py",
        type=Path,
        default=DEFAULT_MDM_PY,
        help=f"Path to MDM Python executable (default: {DEFAULT_MDM_PY})",
    )
    parser.add_argument(
        "--hugs_py",
        type=Path,
        default=DEFAULT_HUGS_PY,
        help=f"Path to HUGS Python executable (default: {DEFAULT_HUGS_PY})",
    )
    
    # MDM parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for MDM (default: 10)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps for MDM (default: 50)",
    )
    
    # Rotation and translation
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center translation to mean 0 before rendering",
    )
    parser.add_argument(
        "--tz",
        type=float,
        default=1.0,
        help="Forward offset (Z-axis) after centering (default: 1.0)",
    )
    
    # Execution control
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.dry_run:
        if not args.mdm_repo.exists():
            print(f"❌ MDM repository not found: {args.mdm_repo}")
            sys.exit(1)
        if not args.hugs_repo.exists():
            print(f"❌ HUGS repository not found: {args.hugs_repo}")
            sys.exit(1)
        if not args.mdm_py.exists():
            print(f"❌ MDM Python not found: {args.mdm_py}")
            sys.exit(1)
        if not args.hugs_py.exists():
            print(f"❌ HUGS Python not found: {args.hugs_py}")
            sys.exit(1)
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(args.prompt)
    run_dir = args.out_root / f"{timestamp}_{slug}"
    
    # Create subdirectories
    mdm_out_dir = run_dir / "mdm_out"
    smpl_npz_dir = run_dir / "smpl_npz"
    rotated_npz_dir = run_dir / "rotated_npz"
    hugs_logs_dir = run_dir / "hugs_logs"
    final_dir = run_dir / "final"
    
    for d in [mdm_out_dir, smpl_npz_dir, rotated_npz_dir, hugs_logs_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("TEXT → MDM → HUGS Pipeline")
    print(f"{'='*80}")
    print(f"Prompt:     {args.prompt}")
    print(f"Scene:      {args.scene}")
    print(f"Run dir:    {run_dir}")
    print(f"Seed:       {args.seed}")
    print(f"Steps:      {args.steps}")
    print(f"Center:     {args.center}")
    print(f"TZ offset:  {args.tz}")
    print(f"Dry run:    {args.dry_run}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    executed_commands = []
    
    # ====================
    # Stage 1: Run MDM
    # ====================
    print(f"\n[1/5] Running MDM motion generation...")
    mdm_cmd = build_mdm_cmd(
        prompt=args.prompt,
        out_dir=mdm_out_dir,
        mdm_repo=args.mdm_repo,
        mdm_py=args.mdm_py,
        seed=args.seed,
        steps=args.steps,
    )
    
    mdm_log = mdm_out_dir / "mdm.log"
    if not run_command(
        mdm_cmd,
        "Generate motion with MDM",
        log_file=mdm_log,
        cwd=args.mdm_repo,
        dry_run=args.dry_run,
    ):
        print("❌ MDM generation failed")
        sys.exit(1)
    
    executed_commands.append({
        "stage": "mdm",
        "cmd": " ".join(str(c) for c in mdm_cmd),
        "cwd": str(args.mdm_repo),
    })
    
    # ====================
    # Stage 2: results.npy → hugs_smpl_original.npz via extract_smpl_params.py
    # ====================
    print(f"\n[2/5] Extracting SMPL parameters from MDM output...")

    target_npz = smpl_npz_dir / "hugs_smpl_original.npz"
    extract_log = mdm_out_dir / "extract_smpl.log"

    if not args.dry_run:
        # Locate the latest MDM samples directory
        mdm_save_dir = args.mdm_repo / "save" / "humanml_enc_512_50steps"
        latest_sample_dir = None

        if mdm_save_dir.exists():
            samples_dirs = sorted(
                mdm_save_dir.glob("samples_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if samples_dirs:
                latest_sample_dir = samples_dirs[0]
                print(f"Found latest MDM output dir: {latest_sample_dir.name}")

        # Find results.npy
        results_npy = None
        if latest_sample_dir and (latest_sample_dir / "results.npy").exists():
            results_npy = latest_sample_dir / "results.npy"
        else:
            results_npy = find_file(args.mdm_repo / "save", "results.npy")

        if results_npy is None or not results_npy.exists():
            print("❌ results.npy not found in MDM save directory")
            print(f"Searched in: {mdm_save_dir}")
            print(f"MDM log: {mdm_log}")
            sys.exit(1)

        print(f"✓ Found MDM results: {results_npy}")

        # Run extract_smpl_params.py (MDM env, MDM cwd) to produce hugs_smpl_original.npz
        # Use absolute path for --output so it is not resolved relative to mdm_repo cwd
        extract_script = args.mdm_repo / "sample" / "extract_smpl_params.py"
        extract_cmd = [
            str(args.mdm_py),
            str(extract_script),
            "--motion_data", str(results_npy),
            "--output", str(target_npz.resolve()),
        ]

        if not run_command(
            extract_cmd,
            "Extract SMPL params (results.npy → hugs_smpl_original.npz)",
            log_file=extract_log,
            cwd=args.mdm_repo,   # must run from MDM root for relative model paths
            dry_run=args.dry_run,
        ):
            print("❌ SMPL extraction failed")
            print(f"See log: {extract_log}")
            sys.exit(1)

        executed_commands.append({
            "stage": "extract_smpl",
            "cmd": " ".join(str(c) for c in extract_cmd),
            "cwd": str(args.mdm_repo),
        })

        if not target_npz.exists():
            raise FileNotFoundError(
                f"hugs_smpl_original.npz not produced by extract_smpl_params.py\n"
                f"Expected: {target_npz}\n"
                f"Check log: {extract_log}"
            )

        print(f"✓ SMPL npz ready: {target_npz}")

    else:
        print(f"[DRY RUN] Would run extract_smpl_params.py on MDM results.npy")
        print(f"[DRY RUN] Target: {target_npz}")
    
    # ====================
    # Stage 3: Rotate to HUGS coordinates
    # ====================
    print(f"\n[3/5] Rotating SMPL motion to HUGS coordinates...")
    
    rotate_script = args.hugs_repo / "scripts/rotate_hugs_motion_v2.py"
    rotated_npz = rotated_npz_dir / "hugs_smpl_upright.npz"
    
    rotate_cmd = [
        str(args.hugs_py),
        str(rotate_script),
        "--input", str(target_npz),
        "--output", str(rotated_npz),
        "--rx", "90",
        "--rz", "180",
    ]
    
    if args.center:
        rotate_cmd.append("--center")
    
    rotate_cmd.extend(["--tz", str(args.tz)])
    
    rotate_log = rotated_npz_dir / "rotate.log"
    if not run_command(
        rotate_cmd,
        "Rotate SMPL motion (MDM → HUGS coords: RX=+90°, RZ=+180°)",
        log_file=rotate_log,
        cwd=args.hugs_repo,
        dry_run=args.dry_run,
    ):
        print("❌ Rotation failed")
        sys.exit(1)
    
    executed_commands.append({
        "stage": "rotate",
        "cmd": " ".join(str(c) for c in rotate_cmd),
        "cwd": str(args.hugs_repo),
    })
    
    # ====================
    # Stage 4: Run HUGS rendering
    # ====================
    print(f"\n[4/5] Running HUGS rendering...")
    
    scene_cfg = SCENE_CONFIGS[args.scene]
    hugs_config = args.hugs_repo / "cfg_files/release/neuman/hugs_human_scene.yaml"
    
    hugs_cmd = [
        str(args.hugs_py),
        "main.py",
        "--cfg_file", str(hugs_config),
        f"dataset.seq={args.scene}",
        "eval=true",
        f"human.ckpt={scene_cfg['human_ckpt']}",
        f"scene.ckpt={scene_cfg['scene_ckpt']}",
        f"custom_motion_path={rotated_npz}",
    ]
    
    hugs_log = hugs_logs_dir / "hugs.log"
    
    # Record timestamp before HUGS run to find new mp4 files
    before_hugs = datetime.now().timestamp()
    
    if not run_command(
        hugs_cmd,
        f"Render HUGS animation (scene={args.scene})",
        log_file=hugs_log,
        cwd=args.hugs_repo,
        dry_run=args.dry_run,
    ):
        print("❌ HUGS rendering failed")
        sys.exit(1)
    
    executed_commands.append({
        "stage": "hugs",
        "cmd": " ".join(str(c) for c in hugs_cmd),
        "cwd": str(args.hugs_repo),
    })
    
    # ====================
    # Stage 5: Extract final video
    # ====================
    print(f"\n[5/5] Extracting final video...")
    
    if not args.dry_run:
        hugs_output_dir = args.hugs_repo / "output"
        newest_mp4 = find_newest_mp4(hugs_output_dir, after_time=before_hugs)
        
        if newest_mp4:
            final_mp4 = final_dir / "result.mp4"
            shutil.copy2(newest_mp4, final_mp4)
            print(f"✓ Video saved to: {final_mp4}")
            print(f"  Source: {newest_mp4}")
        else:
            print("⚠ No new mp4 files found after HUGS rendering")
            print(f"Searched in: {hugs_output_dir}")
            final_mp4 = None
    else:
        print(f"[DRY RUN] Would find newest mp4 in {args.hugs_repo / 'output'}")
        final_mp4 = final_dir / "result.mp4"
    
    end_time = datetime.now()
    
    # ====================
    # Save run record
    # ====================
    record_data = {
        "pipeline_version": "1.0",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "prompt": args.prompt,
        "scene": args.scene,
        "seed": args.seed,
        "steps": args.steps,
        "paths": {
            "run_dir": str(run_dir),
            "mdm_out": str(mdm_out_dir),
            "original_npz": str(target_npz) if 'target_npz' in locals() else None,
            "rotated_npz": str(rotated_npz),
            "final_mp4": str(final_mp4) if final_mp4 else None,
        },
        "rotation": {
            "rx_degrees": 90,
            "rz_degrees": 180,
        },
        "translation": {
            "center": args.center,
            "tz_offset": args.tz,
        },
        "repositories": {
            "mdm": str(args.mdm_repo),
            "hugs": str(args.hugs_repo),
        },
        "python_executables": {
            "mdm": str(args.mdm_py),
            "hugs": str(args.hugs_py),
        },
        "executed_commands": executed_commands,
        "scene_checkpoints": scene_cfg,
    }
    
    record_path = run_dir / "run_record.json"
    with open(record_path, 'w') as f:
        json.dump(record_data, f, indent=2)
    print(f"\n✓ Saved run record to: {record_path}")
    
    # ====================
    # Summary
    # ====================
    print(f"\n{'='*80}")
    print("✓ Pipeline completed successfully!")
    print(f"{'='*80}")
    print(f"Duration:       {record_data['duration_seconds']:.1f} seconds")
    print(f"Run directory:  {run_dir}")
    print(f"Original npz:   {target_npz if 'target_npz' in locals() else 'N/A'}")
    print(f"Rotated npz:    {rotated_npz}")
    print(f"Final video:    {final_mp4 if final_mp4 else 'N/A'}")
    print(f"Run record:     {record_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
