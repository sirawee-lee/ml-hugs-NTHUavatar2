# การสร้างแอนิเมชัน HUGS กระโดดจาก Motion Diffusion Model

## สิ่งที่ได้ทำสำเร็จแล้ว ✅

### 1. ค้นหาและวิเคราะห์ไฟล์ต้นทาง
- พบ `samples_00_to_00.mp4` (ท่ากระโดด) ที่: 
  `/home/sigma/skibidi/motion-diffusion-model/save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_a_person_jumps/`
- พบ SMPL parameters: `sample00_rep00_smpl_params.npy` (120 เฟรม)
- พบ checkpoint ของ HUGS bike model ที่: 
  `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/output/pretrained_models/bike/`

### 2. สร้างสคริปต์แปลงข้อมูล
สร้างไฟล์: `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/scripts/convert_mdm_to_hugs.py`
- แปลงข้อมูล SMPL จาก motion-diffusion-model format
- เปลี่ยนเป็น format ที่ HUGS ใช้ได้
- รองรับ betas จาก bike dataset เพื่อให้รูปร่างตรงกับโมเดล

### 3. แปลงข้อมูลกระโดดสำเร็จ
สร้างไฟล์: `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/data/custom_motions/jumping_motion_bike.npz`

**ข้อมูลที่ได้:**
- `global_orient`: (120, 3) - การหมุนลำตัวหลัก
- `body_pose`: (120, 69) - ท่าทางข้อต่อต่างๆ  
- `transl`: (120, 3) - ตำแหน่งการเคลื่อนที่
- `betas`: (10,) - พารามิเตอร์รูปร่างจาก bike dataset
- จำนวนเฟรม: 120 (ประมาณ 6 วินาที ที่ 20 FPS)

### 4. สร้าง Config File
สร้างไฟล์: `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/cfg_files/release/neuman/hugs_bike_jumping.yaml`
- ตั้งค่า `custom_motion_path` ให้ชี้ไปที่ไฟล์กระโดด
- ใช้ checkpoint จาก bike model
- พร้อมสำหรับการสร้างแอนิเมชัน

### 5. ตรวจสอบความถูกต้อง
สร้างสคริปต์: `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/scripts/test_custom_motion.py`
- ตรวจสอบว่าข้อมูลถูกโหลดได้
- ตรวจสอบว่า checkpoint มีอยู่และโหลดได้
- บันทึกข้อมูลที่: `/home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/output/custom_animations/jumping/motion_info.txt`

## ปัญหาที่พบ ⚠️

### ปัญหา CUDA Environment
การรัน HUGS ต้องการ:
1. **CUDA Runtime** - ปัจจุบันมี CUDA 11.5 แต่ PyTorch ถูก compile ด้วย CUDA 11.7
2. **simple-knn extension** - ต้องการ compile แต่มีปัญหาเรื่อง CUDA architecture

## วิธีแก้ไขและรันแอนิเมชัน 🎬

### ตัวเลือกที่ 1: แก้ไข CUDA Environment (แนะนำ)

```bash
# 1. ตรวจสอบ CUDA version
nvidia-smi
nvcc --version

# 2. ติดตั้ง simple-knn และ diff-gaussian-rasterization
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/submodules/simple-knn
pip install .

cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar/submodules/diff-gaussian-rasterization
pip install .

# 3. รันแอนิเมชัน
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar
python main.py --cfg_file cfg_files/release/neuman/hugs_bike_jumping.yaml
```

### ตัวเลือกที่ 2: ใช้ Docker (ถ้ามี)

```bash
# ใช้ Docker image ที่มี CUDA environment ที่ถูกต้อง
# ดูที่ README.md ของ HUGS project
```

### ตัวเลือกที่ 3: รันบน GPU ที่รองรับ

ถ้า GPU ปัจจุบันไม่รองรับ CUDA 11.7:
- ลองรันบนเครื่องอื่นที่มี GPU ที่รองรับ
- หรือใช้ Cloud GPU service (Google Colab, AWS, etc.)

## ไฟล์สำคัญที่สร้างขึ้น 📁

```
ml-hugs-NTHUavatar/
├── data/custom_motions/
│   └── jumping_motion_bike.npz          # ข้อมูลการกระโดด (120 เฟรม)
├── cfg_files/release/neuman/
│   └── hugs_bike_jumping.yaml           # Config file สำหรับรัน
├── scripts/
│   ├── convert_mdm_to_hugs.py          # สคริปต์แปลงข้อมูล
│   └── test_custom_motion.py           # สคริปต์ทดสอบ
└── output/custom_animations/jumping/
    └── motion_info.txt                 # ข้อมูลการเคลื่อนไหว
```

## คำสั่งที่ใช้ได้เมื่อแก้ CUDA แล้ว 🚀

```bash
# เข้าไปที่โฟลเดอร์โปรเจ็ค
cd /home/sigma/project_avatar2_hugs/ml-hugs-NTHUavatar

# Activate environment
conda activate hugs

# รันแอนิเมชันกระโดด
python main.py --cfg_file cfg_files/release/neuman/hugs_bike_jumping.yaml

# ผลลัพธ์จะถูกบันทึกที่:
# output/human/neuman/bike/hugs_trimlp/demo-jumping-motion/[timestamp]/
# - anim_neuman_bike_final_jumping.mp4  (วิดีโอแอนิเมชัน)
# - anim/                                 (เฟรมแต่ละภาพ)
```

## การใช้งานกับ Motion อื่นๆ 🎭

ถ้าต้องการใช้ท่าอื่นจาก motion-diffusion-model:

```bash
# 1. แปลงข้อมูล SMPL
python scripts/convert_mdm_to_hugs.py \
  --input /path/to/mdm/sample00_rep00_smpl_params.npy \
  --output data/custom_motions/my_custom_motion.npz \
  --betas data/neuman/dataset/bike/4d_humans/smpl_optimized_aligned_scale.npz

# 2. แก้ไข config file
# เปลี่ยน custom_motion_path ใน cfg_files/release/neuman/hugs_bike_jumping.yaml

# 3. รัน
python main.py --cfg_file cfg_files/release/neuman/hugs_bike_jumping.yaml
```

## หมายเหตุ 📝

- ข้อมูลกระโดดมี 120 เฟรม (~6 วินาที)
- ใช้ betas จาก bike dataset เพื่อความสอดคล้องกับโมเดล
- การเคลื่อนที่ (translation) ถูกแปลงจาก motion-diffusion-model แล้ว
- สามารถปรับ alignment (manual_trans, manual_rot, manual_scale) ได้ใน `hugs/datasets/neuman.py` ถ้าต้องการ

## การแก้ไข Alignment (ถ้าต้องการ) 🎯

ถ้าหุ่นกระโดดไม่ถูกตำแหน่งหรือมีขนาดไม่เหมาะสม สามารถปรับแต่งได้ที่:

```python
# File: hugs/datasets/neuman.py
# Function: alignment()

# สำหรับ bike scene:
elif os.path.basename(scene_name) == 'bike':
    manual_trans = np.array([0.0, 0.88, 3.89])    # ปรับตำแหน่ง X, Y, Z
    manual_rot = np.array([88.8, 180, 1.8]) / 180 * np.pi  # ปรับมุมหมุน
    manual_scale = 1.0  # ปรับขนาด
```

---

**สรุป:** ทุกอย่างพร้อมแล้ว! เพียงแค่แก้ไขปัญหา CUDA environment แล้วก็สามารถสร้างวิดีโอ HUGS กระโดดได้ทันที 🎉
