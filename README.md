# NTHU Avatar — Text-to-Animation Pipeline

Generate a rendered 3D human avatar video from a text prompt, using **MDM** (Motion Diffusion Model) for motion generation and **HUGS** (Human Gaussian Splats) for photorealistic rendering.

```
Text Prompt → MDM → SMPL Motion → HUGS Rendering → result.mp4
```

Based on [HUGS: Human Gaussian Splats](https://arxiv.org/abs/2311.17910) (CVPR 2024).

---

## Requirements

- CUDA-capable GPU (tested on CUDA 11.8)
- [Conda](https://docs.conda.io/en/latest/)
- The [MDM repo](https://github.com/GuyTevet/motion-diffusion-model) set up alongside this one

---

## Setup

### 1. Clone both repos

```bash
git clone <this-repo> ml-hugs-NTHUavatar
git clone https://github.com/GuyTevet/motion-diffusion-model
```

### 2. Set up the HUGS environment

```bash
cd ml-hugs-NTHUavatar
source scripts/conda_setup.sh
```

### 3. Set up the MDM environment

```bash
cd motion-diffusion-model
conda env create -f environment.yml
conda activate mdm
pip install -e .
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
```

Download the MDM 50-step checkpoint from [Google Drive](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view) and place it at:
```
motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt
```

### 4. Download HUGS data and pretrained models

```bash
source scripts/prepare_data_models.sh
```

This downloads:
- SMPL body model → `data/smpl/`
- NeuMan dataset → `data/neuman/`
- Pretrained HUGS checkpoints → `output/pretrained_models/`

---

## Usage

### Quick Start

```bash
conda activate hugs
python scripts/run_text2hugs.py \
  --prompt "a person does a latin dance" \
  --out_root ./outputs \
  --human_only \
  --center \
  --bg_color white
```

### Full Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | *(required)* | Text description of the motion |
| `--out_root` | *(required)* | Root output directory |
| `--scene` | `bike` | Background scene: `bike`, `citron`, `jogging` |
| `--human_only` | `False` | Render avatar only without background scene |
| `--bg_color` | `black` | Background color: `white` or `black` |
| `--center` | `False` | Zero-center the root translation |
| `--seed` | `10` | MDM random seed |
| `--steps` | `50` | Number of diffusion steps |
| `--mdm_repo` | auto | Path to MDM repository |
| `--mdm_py` | auto | Python executable for MDM env |
| `--hugs_py` | auto | Python executable for HUGS env |
| `--dry_run` | `False` | Print commands without executing |

### Examples

```bash
# Human only, white background
python scripts/run_text2hugs.py \
  --prompt "a person waves hello" \
  --out_root ./outputs \
  --human_only --center --bg_color white

# With background scene
python scripts/run_text2hugs.py \
  --prompt "a person walks forward" \
  --scene citron \
  --out_root ./outputs \
  --center --tz 1.5

# Dry run to test configuration
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./outputs \
  --dry_run
```

---

## Output Structure

```
outputs/<timestamp>_<prompt_slug>/
├── mdm_out/              # MDM generation artifacts
├── smpl_npz/
│   └── hugs_smpl_original.npz   # Raw SMPL parameters
├── rotated_npz/
│   └── hugs_smpl_upright.npz    # SMPL in HUGS coordinate system
├── hugs_logs/
│   └── hugs.log
├── final/
│   └── result.mp4        # Final rendered video
└── run_record.json       # Run metadata
```

---

## Available Scenes

| Scene | Description |
|-------|-------------|
| `bike` | Outdoor, person near a bicycle |
| `citron` | Indoor scene |
| `jogging` | Outdoor jogging path |

Pretrained checkpoints are stored in `output/pretrained_models/<scene>/`.

---

## Pipeline Details

| Stage | What happens |
|-------|-------------|
| 1. MDM Generation | Runs `sample.generate` in MDM env to produce SMPL motion |
| 2. SMPL Extraction | `sample/extract_smpl_params.py` converts MDM output to `.npz` |
| 3. Coordinate Rotation | `scripts/rotate_hugs_motion_v2.py` applies RX=+90°, RZ=+180° |
| 4. HUGS Rendering | `main.py` renders the 3D Gaussian Splat avatar |
| 5. Video Output | Copies `result.mp4` to `final/` |

---

## Citation

```bibtex
@inproceedings{kocabas2024hugs,
  title={{HUGS}: Human Gaussian Splatting},
  author={Kocabas, Muhammed and Chang, Jen-Hao Rick and Gabriel, James and Tuzel, Oncel and Ranjan, Anurag},
  booktitle={CVPR},
  year={2024},
  url={https://arxiv.org/abs/2311.17910}
}
```

## License

This project is released under the [LICENSE](LICENSE) terms from the original HUGS repository.
