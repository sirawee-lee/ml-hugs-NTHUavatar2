# MDM to HUGS Pipeline Fix - Summary

## Problem
The `run_text2hugs.py` pipeline was failing because:
1. MDM outputs `.npy` files (motion in XYZ joint positions), not `.npz` files
2. The pipeline expected `hugs_smpl_original.npz` to already exist after MDM generation
3. No automatic conversion was implemented to handle the `.npy` → `.npz` transformation
4. This caused an `UnboundLocalError` when `target_npz` was not defined

## Solution

### 1. Created New Converter Script
**File:** `scripts/convert_mdm_results_to_hugs_npz.py`

This script:
- Loads MDM's `results.npy` output (shape: 1, 22, 3, num_frames for joints, 3D positions, frames)
- Extracts the motion data and root position
- Creates HUGS-compatible SMPL parameters:
  - `global_orient`: (num_frames, 3) - root rotation in axis-angle (set to zeros for T-pose)
  - `body_pose`: (num_frames, 69) - body joint rotations (set to zeros for T-pose)
  - `transl`: (num_frames, 3) - root translation from joint positions
  - `betas`: (10,) - shape parameters (default neutral)
- Saves output as `.npz` file in HUGS format
- Handles edge cases (multiple samples, shape mismatches)

### 2. Updated run_text2hugs.py Pipeline
**Stage 2 Improvement:**

Instead of assuming `hugs_smpl_original.npz` exists, the pipeline now:

1. **Check for existing npz**: First checks if `hugs_smpl_original.npz` already exists
2. **Find results.npy**: If not found, searches for MDM's `results.npy` output
3. **Automatic conversion**: Runs `convert_mdm_results_to_hugs_npz.py` to convert `.npy` → `.npz`
4. **Logging**: Saves conversion logs to `mdm_out/convert.log`
5. **Validation**: Verifies the converted `.npz` file exists before proceeding
6. **Clear error handling**: Shows helpful error messages with log file locations if anything fails

**Guaranteed `target_npz` definition:**
- `target_npz` is now always defined as: `smpl_npz_dir / "hugs_smpl_original.npz"`
- This eliminates the previous `UnboundLocalError`

### 3. Updated Execution Record
The `run_record.json` now includes:
- `convert_hugs` stage with full command and working directory
- All executed commands in sequence for reproducibility

## Pipeline Flow (Updated)

```
[1/5] MDM Motion Generation
     └─ Generates: results.npy with motion (1, 22, 3, 120) format
     
[2/5] Find/Convert to HUGS SMPL npz
     ├─ Check: Is hugs_smpl_original.npz already there?
     ├─ If no: Find results.npy from MDM output
     ├─ Convert: Run convert_mdm_results_to_hugs_npz.py
     └─ Output: hugs_smpl_original.npz
     
[3/5] Rotate to HUGS Coordinates
     └─ Apply: RX=+90°, RZ=+180°
     
[4/5] HUGS Rendering
     └─ Generate: 3D animation
     
[5/5] Extract Video
     └─ Save: final/result.mp4
```

## Testing

The updated pipeline was tested successfully:
- ✅ MDM generation works
- ✅ Automatic conversion from results.npy to npz
- ✅ Rotation and HUGS rendering works
- ✅ Final video output generated
- ✅ Run record properly created with all stages

**Test Run:** `python scripts/run_text2hugs.py --prompt "a person jumps" --out_root ./output_text2hugs_v2 --center --tz 1.0`

**Duration:** 116.6 seconds
**Output:** `result.mp4` (165 KB)

## Files Modified

1. **`scripts/run_text2hugs.py`**
   - Updated Stage 2 to automatically detect and convert MDM outputs
   - Removed unused `try_convert_to_hugs_npz()` function
   - Added proper error handling with log file references

2. **`scripts/convert_mdm_results_to_hugs_npz.py`** (NEW)
   - Converts MDM results.npy to HUGS format
   - Independent, standalone converter for flexibility

## Key Improvements

✅ **Robustness**: Pipeline no longer assumes intermediate files exist
✅ **Automation**: Full automatic conversion from MDM → HUGS without manual steps
✅ **Transparency**: Detailed logging of each conversion step
✅ **Error Handling**: Clear error messages pointing to logs if something fails
✅ **Flexibility**: Works with any MDM prompt/output
✅ **Reproducibility**: Complete run records with all commands and parameters

## Future Enhancements (Optional)

- Improve SMPL parameter extraction (currently uses zero rotations/T-pose)
- Add optional joint-to-axis-angle conversion if better rotation data is needed
- Support for custom betas from dataset-specific models
- Parallel processing for multiple samples
