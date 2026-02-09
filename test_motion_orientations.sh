#!/bin/bash

# Test script to render HUGS with different motion orientations
# Renders the same scene with 4 different rotation variants

BASE_CMD="/home/sigma/anaconda3/envs/hugs/bin/python main.py --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml dataset.seq=bike eval=true human.ckpt=output/pretrained_models/bike/human_final.pth scene.ckpt=output/pretrained_models/bike/scene_final.pth human.triplane_res=256 scene.triplane_res=256"

MOTION_DIR="/home/sigma/skibidi/motion-diffusion-model/save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps"

# Test variants
variants=(
    "no_rot:Original (no rotation)"
    "rotZ-90:Rotate Z -90°"
    "rotY-90:Rotate Y -90°"
    "rotX90:Rotate X +90°"
)

for variant in "${variants[@]}"; do
    IFS=':' read -r name label <<< "$variant"
    motion_file="$MOTION_DIR/hugs_smpl_${name}.npz"
    
    echo ""
    echo "========================================"
    echo "Testing: $label"
    echo "Motion: hugs_smpl_${name}.npz"
    echo "========================================"
    
    $BASE_CMD custom_motion_path="$motion_file"
done
