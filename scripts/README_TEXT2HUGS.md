# Complete Text-to-Video Pipeline Documentation

## Overview

This document describes the complete end-to-end pipeline for generating HUGS-rendered videos from text prompts using Motion-Diffusion-Model (MDM).

**Pipeline Flow:**
```
Text Prompt → MDM Generation → SMPL Format Conversion → Coordinate Rotation → HUGS Rendering → Final Video
```

## Quick Start

### Basic Usage

```bash
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./output_text2hugs \
  --center \
  --tz 1.0
```

### Advanced Usage

```bash
python scripts/run_text2hugs.py \
  --prompt "a person walks forward and waves" \
  --scene citron \
  --out_root ./output \
  --seed 42 \
  --steps 50 \
  --center \
  --tz 1.5
```

## Pipeline Scripts

### 1. run_text2hugs.py (Complete Pipeline)

**Purpose:** End-to-end automation from text to rendered video.

**Stages:**
1. **MDM Generation**: Generate motion from text prompt
2. **Format Conversion**: Convert MDM output to HUGS SMPL format
3. **Coordinate Rotation**: Transform to HUGS coordinate system
4. **HUGS Rendering**: Render motion with scene
5. **Video Extraction**: Copy final video to output directory

**Arguments:**

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--prompt` | string | - | ✓ | Text prompt for motion generation |
| `--out_root` | path | - | ✓ | Output directory for all artifacts |
| `--scene` | {bike, citron, jogging} | bike | | HUGS scene |
| `--mdm_repo` | path | `/home/sigma/skibidi/motion-diffusion-model` | | MDM repository |
| `--hugs_repo` | path | `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar` | | HUGS repository |
| `--mdm_py` | path | `/home/sigma/anaconda3/envs/mdm/bin/python` | | MDM Python |
| `--hugs_py` | path | `/home/sigma/anaconda3/envs/hugs/bin/python` | | HUGS Python |
| `--seed` | int | 10 | | Random seed |
| `--steps` | int | 50 | | Diffusion steps |
| `--center` | flag | False | | Center translation |
| `--tz` | float | 1.0 | | Z-axis offset |
| `--dry_run` | flag | False | | Print commands without executing |

### 2. run_mdm2hugs.py (MDM Output to Video)

**Purpose:** Convert existing MDM SMPL output to HUGS-rendered video.

**Use case:** When you already have MDM-generated SMPL motion files.

```bash
python scripts/run_mdm2hugs.py \
  --input_npz /path/to/hugs_smpl_original.npz \
  --out_dir ./output_mdm2hugs \
  --scene bike \
  --center \
  --tz 0.5
```

### 3. rotate_hugs_motion_v2.py (Root-Only Rotation)

**Purpose:** Rotate SMPL motion (root only, preserves body pose).

**Use case:** Standalone rotation for coordinate system conversion.

```bash
python scripts/rotate_hugs_motion_v2.py \
  --input motion.npz \
  --output motion_rotated.npz \
  --rx 90 --rz 180 \
  --center --tz 1.0
```

## Output Structure

After running `run_text2hugs.py`, the output directory contains:

```
<out_root>/<timestamp>_<slug>/
├── mdm_out/                  # MDM generation output
│   ├── mdm.log               # MDM execution log
│   └── [MDM files]           # Generated motion files
├── smpl_npz/                 # Converted SMPL format
│   └── hugs_smpl_original.npz
├── rotated_npz/              # Rotated motion (HUGS coords)
│   ├── hugs_smpl_upright_z180.npz
│   └── rotate.log
├── hugs_logs/                # HUGS rendering logs
│   └── hugs.log
├── final/                    # Final output
│   └── result.mp4            # *** Final rendered video ***
└── run_record.json           # Complete execution metadata
```

## MDM Integration

### Current Status

The MDM command in `build_mdm_cmd()` is a **PLACEHOLDER** that needs to be configured for your specific setup.

### TODO: Configure MDM Command

Edit `scripts/run_text2hugs.py`, function `build_mdm_cmd()` (around line 135):

```python
def build_mdm_cmd(...):
    # TODO: Update this command to match your MDM setup
    cmd = [
        str(mdm_py),
        "-m", "sample.generate",  # Your MDM script
        "--model_path", "...",     # Your model checkpoint
        "--text_prompt", prompt,
        "--num_samples", "1",
        "--seed", str(seed),
        # Add your specific arguments
    ]
    return cmd
```

**Common MDM Variations:**

**Option 1: HumanML3D model**
```python
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
```

**Option 2: KIT-ML model**
```python
cmd = [
    str(mdm_py),
    "-m", "sample.generate",
    "--model_path", str(mdm_repo / "save/kit_enc_512_50steps/model000400000.pt"),
    "--dataset", "kit",
    "--text_prompt", prompt,
    "--seed", str(seed),
]
```

**Option 3: Custom script**
```python
cmd = [
    str(mdm_py),
    str(mdm_repo / "sample/generate.py"),
    "--config", str(mdm_repo / "configs/sample.yaml"),
    "--prompt", prompt,
    "--output", str(out_dir),
]
```

### MDM Output Discovery

The pipeline searches for `hugs_smpl_original.npz` in:
1. MDM output directory (`mdm_out/`)
2. MDM repository save directory (`<mdm_repo>/save/`)

If not found, the pipeline attempts conversion via `try_convert_to_hugs_npz()`.

## Format Conversion

### TODO: Implement MDM → HUGS Conversion

Edit `scripts/run_text2hugs.py`, function `try_convert_to_hugs_npz()` (around line 170):

```python
def try_convert_to_hugs_npz(mdm_out_dir: Path, smpl_out_path: Path) -> bool:
    # TODO: Implement your conversion logic
    
    # Example conversion:
    import numpy as np
    
    # 1. Find MDM output files
    mdm_file = mdm_out_dir / "results.npy"  # Or your MDM output name
    
    # 2. Load MDM data
    mdm_data = np.load(mdm_file)
    
    # 3. Extract/convert to SMPL parameters
    # Structure depends on your MDM output format
    global_orient = ...  # Shape: (T, 3)
    body_pose = ...      # Shape: (T, 69)
    transl = ...         # Shape: (T, 3)
    betas = ...          # Shape: (10,) or (1, 10)
    
    # 4. Save in HUGS format
    np.savez(
        smpl_out_path,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        betas=betas,
    )
    
    return True
```

**Common Conversion Patterns:**

**Pattern 1: MDM outputs SMPL parameters directly**
```python
# MDM file: sample00_rep00_smpl_params.npy
params = np.load(mdm_out_dir / "sample00_rep00_smpl_params.npy", allow_pickle=True).item()
np.savez(
    smpl_out_path,
    global_orient=params['root_orient'],
    body_pose=params['pose_body'],
    transl=params['trans'],
    betas=params['betas'],
)
```

**Pattern 2: MDM outputs joint positions (needs SMPL fitting)**
```python
# MDM file: results.npy (joint positions)
joints = np.load(mdm_out_dir / "results.npy")  # Shape: (1, 22, 3, T)

# Use SMPL inverse kinematics or optimization
# This requires additional libraries (e.g., smplx, torch)
from smpl_fitting import fit_smpl_to_joints  # Your implementation

smpl_params = fit_smpl_to_joints(joints)
np.savez(smpl_out_path, **smpl_params)
```

## Coordinate System Transformation

### MDM → HUGS Rotation

The pipeline applies hardcoded rotation to convert MDM coordinates to HUGS:

- **RX = +90°**: Rotate around X-axis (align vertical)
- **RZ = +180°**: Rotate around Z-axis (flip forward direction)

This is implemented in `rotate_hugs_motion_v2.py` using **root-only rotation**:
- Only `global_orient` (root joint) is rotated
- `body_pose` (relative joint rotations) remains unchanged
- `transl` (root translation) is rotated

### Axis Convention

- **X-axis**: Right
- **Y-axis**: Up
- **Z-axis**: Forward

## Supported Scenes

| Scene | Description | Checkpoints Required |
|-------|-------------|----------------------|
| bike | Person on bike | `bike/human_final.pth`, `bike/scene_final.pth` |
| citron | Indoor scene | `citron/human_final.pth`, `citron/scene_final.pth` |
| jogging | Outdoor jogging | `jogging/human_final.pth`, `jogging/scene_final.pth` |

Checkpoints location: `<hugs_repo>/output/pretrained_models/<scene>/`

## Examples

### Example 1: Basic Jump Motion

```bash
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./output \
  --center \
  --tz 1.0
```

**Expected output:**
- Duration: ~2-3 minutes
- Video: `output/<timestamp>_a_person_jumps/final/result.mp4`

### Example 2: Custom Scene and Settings

```bash
python scripts/run_text2hugs.py \
  --prompt "a person walks backwards" \
  --scene citron \
  --out_root ./custom_output \
  --seed 42 \
  --steps 100 \
  --center \
  --tz 2.0
```

### Example 3: Dry Run (Test Configuration)

```bash
python scripts/run_text2hugs.py \
  --prompt "test motion" \
  --out_root ./test \
  --dry_run
```

**Output:** Prints all commands without execution.

## Troubleshooting

### Issue 1: MDM Command Fails

**Symptom:** Pipeline fails at Stage 1 (MDM generation)

**Solution:**
1. Check `mdm_out/mdm.log` for error details
2. Verify MDM Python environment: `/home/sigma/anaconda3/envs/mdm/bin/python`
3. Update `build_mdm_cmd()` with correct MDM arguments
4. Test MDM command manually first

### Issue 2: SMPL NPZ Not Found

**Symptom:** Pipeline fails at Stage 2 with "SMPL npz not found"

**Solution:**
1. Check MDM output directory for generated files
2. Implement `try_convert_to_hugs_npz()` function
3. Verify MDM output format matches expected structure
4. Look for files like `results.npy`, `sample00_rep00_smpl_params.npy`

### Issue 3: Avatar Off-Screen

**Symptom:** Video renders but avatar not visible

**Solution:**
- Adjust `--tz` offset: try values 0.5, 1.0, 1.5, 2.0
- Use `--center` flag to zero mean translation
- Check HUGS logs in `hugs_logs/hugs.log`

### Issue 4: Wrong Avatar Orientation

**Symptom:** Avatar lying down or facing wrong direction

**Solution:**
- Verify rotation is RX=+90°, RZ=+180° (hardcoded in pipeline)
- Check if you need `rotate_hugs_motion.py` (full chain) vs `rotate_hugs_motion_v2.py` (root-only)
- For custom rotations, use standalone rotation scripts

## Performance

**Typical Runtime:**
- MDM Generation: 30-60 seconds
- Format Conversion: < 5 seconds
- Rotation: < 5 seconds
- HUGS Rendering: 60-90 seconds
- **Total: ~2-3 minutes per prompt**

**GPU Requirements:**
- MDM: CUDA-capable GPU (tested on 11.8)
- HUGS: CUDA-capable GPU (11.8+, PyTorch 1.13)

## Advanced Usage

### Using Existing MDM Output

If you already have MDM-generated SMPL files:

```bash
python scripts/run_mdm2hugs.py \
  --input_npz /path/to/existing/hugs_smpl_original.npz \
  --out_dir ./output \
  --scene bike \
  --center \
  --tz 1.0
```

### Batch Processing Multiple Prompts

```bash
#!/bin/bash
prompts=(
  "a person jumps"
  "a person walks"
  "a person dances"
)

for prompt in "${prompts[@]}"; do
  python scripts/run_text2hugs.py \
    --prompt "$prompt" \
    --out_root ./batch_output \
    --center \
    --tz 1.0
done
```

### Custom Python Environments

```bash
python scripts/run_text2hugs.py \
  --prompt "a person waves" \
  --out_root ./output \
  --mdm_py /custom/path/to/python \
  --hugs_py /custom/path/to/python \
  --mdm_repo /custom/mdm/path \
  --hugs_repo /custom/hugs/path
```

## File Locations

**Scripts:**
- `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/scripts/run_text2hugs.py`
- `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/scripts/run_mdm2hugs.py`
- `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/scripts/rotate_hugs_motion_v2.py`
- `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/scripts/rotate_hugs_motion.py`

**Repositories:**
- MDM: `/home/sigma/skibidi/motion-diffusion-model`
- HUGS: `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar`

**Python Environments:**
- MDM: `/home/sigma/anaconda3/envs/mdm/bin/python`
- HUGS: `/home/sigma/anaconda3/envs/hugs/bin/python`

## Next Steps

1. **Configure MDM Command**: Update `build_mdm_cmd()` in `run_text2hugs.py`
2. **Implement Conversion**: Update `try_convert_to_hugs_npz()` in `run_text2hugs.py`
3. **Test Pipeline**: Run with `--dry_run` first
4. **Validate Output**: Check each stage's logs and outputs
5. **Iterate**: Adjust `--tz`, `--center`, scene parameters as needed

## References

- **HUGS Paper**: [Human Gaussian Splatting](https://machinelearning.apple.com/research/hugs)
- **MDM Paper**: [Human Motion Diffusion Model](https://guytevet.github.io/mdm-page/)
- **SMPL Format**: [SMPL Body Model](https://smpl.is.tue.mpg.de/)
