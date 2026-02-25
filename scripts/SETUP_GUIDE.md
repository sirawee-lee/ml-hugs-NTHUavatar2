# Step-by-Step Guide: Complete the Text-to-HUGS Pipeline

## 🎯 Goal
Make `run_text2hugs.py` fully operational by configuring MDM integration.

## 📋 What You Have
✅ MDM installed at `/home/sigma/skibidi/motion-diffusion-model`
✅ Model checkpoint: `save/humanml_enc_512_50steps/model000750000.pt`
✅ Sample script: `sample/generate.py`
✅ Existing outputs show `hugs_smpl_original.npz` files are already created
✅ Extract script: `sample/extract_smpl_params.py`

## 🔍 What We Found
Your MDM **already outputs** `hugs_smpl_original.npz` files! 
Looking at: `/home/sigma/skibidi/motion-diffusion-model/save/humanml_enc_512_50steps/samples_*/`
- ✅ `hugs_smpl_original.npz` exists
- ✅ `results.npy` exists
- ✅ `sample00_rep00_smpl_params.npy` exists

**This means**: MDM is already configured to output HUGS-compatible format! You just need to update the pipeline script to use the correct command.

---

## 📝 Step 1: Find Your MDM Run Command

First, let's figure out the exact command you use to run MDM.

**Check your terminal history or scripts:**

```bash
# Search for how you ran MDM before
history | grep "python.*generate"
history | grep "sample/generate"

# Or check if you have a script
ls ~/skibidi/motion-diffusion-model/*.sh
cat ~/skibidi/motion-diffusion-model/run_generate.sh  # if exists
```

**Most likely command format:**
```bash
cd /home/sigma/skibidi/motion-diffusion-model
python -m sample.generate \
  --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
  --text_prompt "a person jumps" \
  --num_samples 1 \
  --num_repetitions 1 \
  --guidance_param 2.5 \
  --seed 10
```

---

## 📝 Step 2: Update `build_mdm_cmd()` Function

Edit: `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/scripts/run_text2hugs.py`

**Find this function (around line 135):**

```python
def build_mdm_cmd(
    prompt: str,
    out_dir: Path,
    mdm_repo: Path,
    mdm_py: Path,
    seed: int,
    steps: int,
) -> List[str]:
```

**Replace the TODO section with your actual command:**

```python
def build_mdm_cmd(
    prompt: str,
    out_dir: Path,
    mdm_repo: Path,
    mdm_py: Path,
    seed: int,
    steps: int,
) -> List[str]:
    """Build MDM sampling command."""
    
    # Use the command that works for you
    cmd = [
        str(mdm_py),
        "-m", "sample.generate",
        "--model_path", str(mdm_repo / "save/humanml_enc_512_50steps/model000750000.pt"),
        "--text_prompt", prompt,
        "--num_samples", "1",
        "--num_repetitions", "1",
        "--guidance_param", "2.5",
        "--seed", str(seed),
        # MDM will auto-create output in save/humanml_enc_512_50steps/samples_...
    ]
    
    return cmd
```

**Note:** MDM creates its own output directory automatically in the format:
`save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed{seed}_{prompt}/`

---

## 📝 Step 3: Update `try_convert_to_hugs_npz()` Function

**Good news:** You don't need complex conversion! Your MDM already outputs `hugs_smpl_original.npz`.

Edit the same file, find `try_convert_to_hugs_npz()` (around line 170):

**Replace with this simple implementation:**

```python
def try_convert_to_hugs_npz(mdm_out_dir: Path, smpl_out_path: Path) -> bool:
    """
    Convert MDM output to HUGS SMPL npz format.
    
    Since MDM already outputs hugs_smpl_original.npz, we just need to find it.
    """
    print(f"\n{'='*80}")
    print("Converting MDM output to HUGS SMPL format")
    print(f"{'='*80}")
    print(f"Searching in: {mdm_out_dir}")
    
    # Search for hugs_smpl_original.npz in MDM output
    # MDM creates: save/humanml_enc_512_50steps/samples_*/hugs_smpl_original.npz
    hugs_npz = find_file(mdm_out_dir.parent, "hugs_smpl_original.npz")
    
    if hugs_npz and hugs_npz.exists():
        print(f"✓ Found MDM output: {hugs_npz}")
        import shutil
        shutil.copy2(hugs_npz, smpl_out_path)
        print(f"✓ Copied to: {smpl_out_path}")
        return True
    
    # If not found, list what we have
    print("❌ hugs_smpl_original.npz not found")
    print("\nFiles found in MDM output:")
    for f in sorted(mdm_out_dir.parent.rglob("*")):
        if f.is_file() and f.suffix in ['.npy', '.npz']:
            print(f"  - {f.relative_to(mdm_out_dir.parent)}")
    print(f"{'='*80}\n")
    
    return False
```

---

## 📝 Step 4: Test with Dry Run

Before running the full pipeline, test with `--dry_run`:

```bash
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar

python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./test_pipeline \
  --dry_run
```

**What to check:**
- ✅ MDM command looks correct
- ✅ All paths are valid
- ✅ No errors in command construction

---

## 📝 Step 5: Test Full Pipeline (Small Run)

Run the actual pipeline with a simple prompt:

```bash
python scripts/run_text2hugs.py \
  --prompt "a person jumps" \
  --out_root ./output_text2hugs \
  --scene bike \
  --center \
  --tz 1.0
```

**Expected behavior:**
1. **Stage 1**: MDM runs, creates samples directory
2. **Stage 2**: Finds `hugs_smpl_original.npz` in MDM output
3. **Stage 3**: Rotates motion to HUGS coords
4. **Stage 4**: HUGS renders the motion
5. **Stage 5**: Copies final video to output

**Check logs:**
```bash
# Check each stage
cat output_text2hugs/<timestamp>_a_person_jumps/mdm_out/mdm.log
cat output_text2hugs/<timestamp>_a_person_jumps/rotated_npz/rotate.log
cat output_text2hugs/<timestamp>_a_person_jumps/hugs_logs/hugs.log

# Check final video
ls output_text2hugs/<timestamp>_a_person_jumps/final/
```

---

## 📝 Step 6: Fix MDM Output Path (If Needed)

If Stage 2 can't find `hugs_smpl_original.npz`, you need to update the search path.

**Problem:** MDM creates output in its own directory structure, not in the `mdm_out_dir` we specify.

**Solution:** Update the search in `run_text2hugs.py` Stage 2 (around line 420):

```python
# Stage 2: Find/convert SMPL npz
print(f"\n[2/5] Finding HUGS SMPL npz...")

if not args.dry_run:
    # MDM creates output in save/humanml_enc_512_50steps/samples_*/
    # Search there instead of mdm_out_dir
    mdm_save_dir = args.mdm_repo / "save/humanml_enc_512_50steps"
    
    # Find the most recent samples directory
    samples_dirs = sorted(mdm_save_dir.glob("samples_*"), key=lambda p: p.stat().st_mtime)
    if samples_dirs:
        latest_sample_dir = samples_dirs[-1]
        print(f"Found MDM output: {latest_sample_dir}")
        original_npz = latest_sample_dir / "hugs_smpl_original.npz"
        
        if original_npz.exists():
            print(f"✓ Found SMPL npz: {original_npz}")
            target_npz = smpl_npz_dir / "hugs_smpl_original.npz"
            shutil.copy2(original_npz, target_npz)
            print(f"✓ Copied to: {target_npz}")
        else:
            print("⚠ hugs_smpl_original.npz not found, attempting conversion...")
            # ... rest of conversion logic
```

---

## 🎉 Step 7: Verify Everything Works

Run a complete test:

```bash
python scripts/run_text2hugs.py \
  --prompt "a person walks forward" \
  --out_root ./final_test \
  --scene bike \
  --center \
  --tz 1.0
```

**Success indicators:**
- ✅ All 5 stages complete without errors
- ✅ `final/result.mp4` exists
- ✅ `run_record.json` has all metadata
- ✅ Video shows avatar in correct orientation

---

## 🐛 Troubleshooting

### Issue: MDM command fails

**Check:**
```bash
# Test MDM manually first
cd /home/sigma/skibidi/motion-diffusion-model
/home/sigma/anaconda3/envs/mdm/bin/python -m sample.generate \
  --model_path ./save/humanml_enc_512_50steps/model000750000.pt \
  --text_prompt "a person jumps" \
  --num_samples 1 \
  --seed 10
```

### Issue: Can't find hugs_smpl_original.npz

**Check:**
```bash
# See what MDM actually creates
ls -la /home/sigma/skibidi/motion-diffusion-model/save/humanml_enc_512_50steps/samples_*/
```

### Issue: Avatar off-screen in video

**Fix:** Adjust `--tz` parameter:
```bash
# Try different values
--tz 0.5   # Closer
--tz 1.0   # Medium (default)
--tz 1.5   # Further
--tz 2.0   # Far
```

---

## 📊 Summary of Changes Needed

### File: `run_text2hugs.py`

**Change 1: `build_mdm_cmd()` (line ~135)**
```python
# Replace TODO with actual MDM command
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

**Change 2: `try_convert_to_hugs_npz()` (line ~170)**
```python
# Simplified: MDM already outputs hugs_smpl_original.npz
def try_convert_to_hugs_npz(mdm_out_dir: Path, smpl_out_path: Path) -> bool:
    hugs_npz = find_file(mdm_out_dir.parent, "hugs_smpl_original.npz")
    if hugs_npz and hugs_npz.exists():
        import shutil
        shutil.copy2(hugs_npz, smpl_out_path)
        return True
    return False
```

**Change 3: Stage 2 search path (line ~420)** - if needed
```python
# Search in MDM's actual output location
mdm_save_dir = args.mdm_repo / "save/humanml_enc_512_50steps"
samples_dirs = sorted(mdm_save_dir.glob("samples_*"), key=lambda p: p.stat().st_mtime)
```

---

## ✅ Next Action

**I can make these changes for you right now!** Just confirm:

1. Do you want me to update `run_text2hugs.py` with the MDM command above?
2. Should I implement the simplified conversion function?
3. Do you want me to update the Stage 2 search path?

Type "yes" and I'll implement all three changes immediately!
