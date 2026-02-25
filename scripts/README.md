# HUGS Pipeline Scripts - Master Reference

Complete toolkit for generating and rendering human motion with HUGS (Human Gaussian Splatting).

## 📋 Table of Contents

- [Quick Reference](#quick-reference)
- [Pipeline Scripts](#pipeline-scripts)
- [Workflow Diagrams](#workflow-diagrams)
- [Common Use Cases](#common-use-cases)
- [Documentation](#documentation)

## 🚀 Quick Reference

### Text → Video (Complete Pipeline)

```bash
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./output \
  --center --tz 1.0
```

### Existing Motion → Video

```bash
python scripts/run_mdm2hugs.py \
  --input_npz motion.npz \
  --out_dir ./output \
  --scene bike --center --tz 1.0
```

### Standalone Rotation

```bash
python scripts/rotate_hugs_motion_v2.py \
  --input motion.npz \
  --output motion_rotated.npz \
  --rx 90 --rz 180 --center --tz 1.0
```

## 📦 Pipeline Scripts

### 1. run_text2hugs.py ⭐ (COMPLETE PIPELINE)

**End-to-end automation: Text → MDM → HUGS → Video**

**Stages:**
1. Generate motion from text (MDM)
2. Convert to HUGS SMPL format
3. Rotate to HUGS coordinates
4. Render with HUGS
5. Extract final video

**Quick Start:**
```bash
python scripts/run_text2hugs.py \
  --prompt "a person walks" \
  --out_root ./output_text2hugs \
  --center --tz 1.0
```

**Status:** ⚠️ Requires MDM configuration (see [README_TEXT2HUGS.md](README_TEXT2HUGS.md))

**Output:**
```
<timestamp>_<slug>/
├── mdm_out/              # MDM generation
├── smpl_npz/             # Converted SMPL
├── rotated_npz/          # Rotated motion
├── hugs_logs/            # HUGS logs
├── final/
│   └── result.mp4        # *** Final video ***
└── run_record.json
```

---

### 2. run_mdm2hugs.py (MDM Output → Video)

**Convert existing MDM SMPL motion to HUGS video**

**Use case:** You already have `hugs_smpl_original.npz` from MDM

**Quick Start:**
```bash
python scripts/run_mdm2hugs.py \
  --input_npz /path/to/hugs_smpl_original.npz \
  --out_dir ./output \
  --scene bike --center --tz 0.5
```

**Status:** ✅ Ready to use

**Output:**
```
out_dir/
├── hugs_smpl_original_mdm2hugs.npz  # Rotated motion
├── run_record.json                   # Metadata
└── [HUGS videos in HUGS output/]
```

---

### 3. rotate_hugs_motion_v2.py (Root-Only Rotation)

**Rotate SMPL motion (global_orient only, keeps body_pose relative)**

**Use case:** Coordinate system conversion, avatar orientation fix

**Quick Start:**
```bash
python scripts/rotate_hugs_motion_v2.py \
  --input motion.npz \
  --output motion_upright.npz \
  --rx 90 --center --tz 1.0
```

**Status:** ✅ Ready to use

**Key Feature:** Only rotates root joint, preserves body pose relationships

**Best for:**
- MDM → HUGS coordinate conversion (RX=+90°, RZ=+180°)
- Fixing avatar orientation when arms are correct but body is wrong

---

### 4. rotate_hugs_motion.py (Full Chain Rotation)

**Rotate entire SMPL pose hierarchy (all joints)**

**Use case:** When you need to rotate both body AND joint orientations

**Quick Start:**
```bash
python scripts/rotate_hugs_motion.py \
  --input motion.npz \
  --output motion_rotated.npz \
  --rx 90 --ry 45 --rz 180 \
  --center --tz 0.5
```

**Status:** ✅ Ready to use

**Key Feature:** Applies conjugation rotation to all joint rotations

**Best for:**
- Complete pose reorientation
- When both body and joints need rotation

---

## 🔄 Workflow Diagrams

### Complete Pipeline (run_text2hugs.py)

```
┌─────────────────────┐
│   Text Prompt       │
│  "a person jumps"   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   MDM Generation    │
│  (Motion Diffusion) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Format Conversion  │
│  MDM → HUGS SMPL    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Coordinate Rotate  │
│  RX=+90°, RZ=+180°  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  HUGS Rendering     │
│  (Gaussian Splat)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Final Video       │
│   result.mp4        │
└─────────────────────┘
```

### MDM-to-HUGS Pipeline (run_mdm2hugs.py)

```
┌─────────────────────┐
│  hugs_smpl_*.npz    │
│  (MDM SMPL output)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Coordinate Rotate  │
│  RX=+90°, RZ=+180°  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  HUGS Rendering     │
│  (scene + human)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Video Output      │
│   anim_*.mp4        │
└─────────────────────┘
```

## 📚 Common Use Cases

### Use Case 1: Generate Video from Text

**Goal:** Create rendered video from text description

**Script:** `run_text2hugs.py`

```bash
python scripts/run_text2hugs.py \
  --prompt "a person performs a backflip" \
  --out_root ./my_videos \
  --scene bike \
  --center --tz 1.0
```

**Prerequisites:**
- ⚠️ MDM command configured (edit `build_mdm_cmd()`)
- ⚠️ Conversion implemented (edit `try_convert_to_hugs_npz()`)

---

### Use Case 2: Render Existing MDM Motion

**Goal:** You have MDM SMPL file, want HUGS video

**Script:** `run_mdm2hugs.py`

```bash
python scripts/run_mdm2hugs.py \
  --input_npz /path/to/hugs_smpl_original.npz \
  --out_dir ./output \
  --scene citron \
  --center --tz 1.5
```

**Prerequisites:**
- ✅ SMPL file exists in correct format
- ✅ HUGS checkpoints available

---

### Use Case 3: Fix Avatar Orientation

**Goal:** Avatar lying down, need it standing upright

**Script:** `rotate_hugs_motion_v2.py`

```bash
# Make avatar stand upright
python scripts/rotate_hugs_motion_v2.py \
  --input motion_horizontal.npz \
  --output motion_upright.npz \
  --rx 90 --center --tz 0.5

# Then render with HUGS
python main.py \
  --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml \
  dataset.seq=bike eval=true \
  human.ckpt=output/pretrained_models/bike/human_final.pth \
  scene.ckpt=output/pretrained_models/bike/scene_final.pth \
  custom_motion_path=motion_upright.npz
```

---

### Use Case 4: Batch Process Multiple Prompts

**Goal:** Generate videos for many text prompts

```bash
#!/bin/bash
prompts=(
  "a person jumps"
  "a person walks"
  "a person dances"
  "a person waves"
)

for prompt in "${prompts[@]}"; do
  echo "Processing: $prompt"
  python scripts/run_text2hugs.py \
    --prompt "$prompt" \
    --out_root ./batch_output \
    --center --tz 1.0
done
```

---

### Use Case 5: Custom Scene Rendering

**Goal:** Render motion in different HUGS scenes

```bash
# Bike scene
python scripts/run_mdm2hugs.py \
  --input_npz motion.npz \
  --out_dir ./output_bike \
  --scene bike

# Citron scene
python scripts/run_mdm2hugs.py \
  --input_npz motion.npz \
  --out_dir ./output_citron \
  --scene citron

# Jogging scene
python scripts/run_mdm2hugs.py \
  --input_npz motion.npz \
  --out_dir ./output_jogging \
  --scene jogging
```

---

## 📖 Documentation

### Detailed Guides

1. **[README_TEXT2HUGS.md](README_TEXT2HUGS.md)**
   - Complete text-to-video pipeline
   - MDM integration guide
   - Format conversion instructions
   - Troubleshooting

2. **[README_MDM2HUGS.md](README_MDM2HUGS.md)**
   - MDM output to HUGS video
   - Rotation coordinate systems
   - Scene configurations
   - Performance notes

### Script References

| Script | Lines | Purpose | Status |
|--------|-------|---------|--------|
| `run_text2hugs.py` | ~700 | Text → Video (end-to-end) | ⚠️ Needs MDM config |
| `run_mdm2hugs.py` | ~200 | MDM SMPL → Video | ✅ Ready |
| `rotate_hugs_motion_v2.py` | ~150 | Root-only rotation | ✅ Ready |
| `rotate_hugs_motion.py` | ~130 | Full chain rotation | ✅ Ready |

## 🎯 Parameter Guide

### Rotation Parameters

| Parameter | Type | Description | Common Values |
|-----------|------|-------------|---------------|
| `--rx` | float | Rotate around X-axis (degrees) | 90 (upright) |
| `--ry` | float | Rotate around Y-axis (degrees) | 180 (turn around) |
| `--rz` | float | Rotate around Z-axis (degrees) | 180 (flip forward) |
| `--center` | flag | Zero mean translation | Usually True |
| `--tx` | float | X-axis offset | 0.0 |
| `--ty` | float | Y-axis offset (height) | 0.0 |
| `--tz` | float | Z-axis offset (forward) | 0.5-2.0 |

### MDM → HUGS Standard Rotation

**Always use for MDM coordinate conversion:**
```bash
--rx 90 --rz 180 --center --tz 1.0
```

### Scene Parameters

| Scene | Description | Best For |
|-------|-------------|----------|
| `bike` | Person on bike (outdoor) | General motion, sports |
| `citron` | Indoor environment | Walking, gestures |
| `jogging` | Outdoor jogging path | Running, athletic motion |

## 🔧 Environment Setup

### Python Environments

```bash
# MDM environment
/home/sigma/anaconda3/envs/mdm/bin/python

# HUGS environment
/home/sigma/anaconda3/envs/hugs/bin/python
```

### Repository Locations

```bash
# MDM repository
/home/sigma/skibidi/motion-diffusion-model

# HUGS repository
/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar
```

## ⚡ Performance Tips

1. **Use `--dry_run` first**: Test command structure before execution
2. **Adjust `--tz` incrementally**: Try 0.5, 1.0, 1.5, 2.0 for positioning
3. **Always use `--center`**: Helps keep avatar in frame
4. **GPU memory**: HUGS needs ~8GB VRAM minimum
5. **Expected runtime**: 2-3 minutes per motion (MDM + HUGS)

## 🐛 Troubleshooting Quick Reference

| Problem | Solution | Script |
|---------|----------|--------|
| MDM fails | Configure `build_mdm_cmd()` | `run_text2hugs.py` |
| No SMPL file | Implement `try_convert_to_hugs_npz()` | `run_text2hugs.py` |
| Avatar off-screen | Adjust `--tz`, use `--center` | Any rotation script |
| Wrong orientation | Use `--rx 90 --rz 180` | `rotate_hugs_motion_v2.py` |
| Body/arms misaligned | Use `v2` (root-only) not `v1` (full) | `rotate_hugs_motion_v2.py` |

## 📝 TODO Checklist

Before using `run_text2hugs.py`:

- [ ] Configure MDM command in `build_mdm_cmd()` function
- [ ] Implement conversion in `try_convert_to_hugs_npz()` function
- [ ] Test MDM generation standalone
- [ ] Verify MDM output format and file locations
- [ ] Test with `--dry_run` first
- [ ] Validate HUGS checkpoints exist for chosen scene

## 🔗 Links & Resources

- **HUGS Repository:** [ml-hugs-NTHUavatar](https://github.com/...)
- **MDM Repository:** [motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)
- **SMPL Models:** [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)

---

## 📞 Support

For issues or questions:
1. Check detailed documentation in `README_TEXT2HUGS.md` or `README_MDM2HUGS.md`
2. Review logs in output directories
3. Test with `--dry_run` to debug commands
4. Verify environment setup and paths

---

**Last Updated:** February 9, 2026
**Version:** 1.0
**Maintainer:** HUGS Pipeline Team
