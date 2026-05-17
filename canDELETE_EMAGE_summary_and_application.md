# EMAGE Paper Summary & Application to This Project

> Paper: *EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling* (CVPR 2024)

---

## Summary in English (B1–B2 Level)

### What problem does EMAGE solve?

Most previous systems for animating a 3D human character from speech only control **one part of the body** — either the face, the upper body, or the hands — but not all of them together. EMAGE is a framework that generates **full-body gestures at the same time**: face expressions, upper body, hands, lower body, and global movement (walking, turning), all synchronized with speech audio.

### Key contributions

**1. BEAT2 Dataset**
The authors built a new dataset called BEAT2 (BEAT-SMPLX-FLAME). It contains 60 hours of motion-captured human speech and gestures. The body is stored in **SMPL-X** format and the face in **FLAME** format — both are standard 3D mesh formats widely used in research. This makes the dataset compatible with many other systems.

**2. Masked Gesture Modeling**
EMAGE is trained with a "masking" trick borrowed from language models like BERT. During training, some body joints are hidden (masked), and the model must fill them in. This forces the model to learn strong motion priors, which makes it better at inference time. It also means you can give the model **partial gestures as hints** (e.g., fix the lower body to a walking motion) and it will complete the rest coherently.

**3. Content and Rhythm Self-Attention (CRA)**
Speech has two kinds of useful information:
- **Rhythm**: the beat and timing of the audio
- **Content**: the meaning of the words (from text transcripts)

EMAGE uses a self-attention mechanism to *adaptively blend* these two signals per frame, producing gestures that are both beat-aware and semantically meaningful (e.g., raising both hands for "spare time").

**4. Compositional VQ-VAEs**
Instead of one big model for the whole body, EMAGE uses **four separate VQ-VAEs** — one each for face, upper body, hands, and lower body. This prevents the model from ignoring less-frequent body movements.

### How does it perform?

EMAGE achieves **state-of-the-art** results on the BEAT2 benchmark:
- Best FGD (Frechet Gesture Distance) → most realistic body gestures
- Competitive MSE/LVD on facial vertex accuracy
- 52.7% user preference win rate in perceptual study vs. prior methods

---

## สรุปเป็นภาษาไทย

### EMAGE คืออะไร?

EMAGE คือ framework สำหรับสร้าง **ท่าทางร่างกายเต็มตัว** ของมนุษย์ดิจิทัล โดยใช้เสียงพูดเป็น input ระบบนี้สามารถสร้างการเคลื่อนไหวพร้อมกันทั้ง ใบหน้า, ลำตัวส่วนบน, มือ, ขา และการเคลื่อนที่ระดับ global (เช่น การเดิน) — ทั้งหมดนี้ให้ตรงกับจังหวะและเนื้อหาของเสียงพูด

### จุดเด่นสำคัญ

**1. Dataset ใหม่ชื่อ BEAT2**
มีข้อมูล motion capture 60 ชั่วโมง เก็บ body ในรูปแบบ **SMPL-X** และใบหน้าในรูปแบบ **FLAME** ซึ่งเป็นมาตรฐานที่ใช้กันกว้างขวางในงานวิจัย 3D avatar ทำให้เชื่อมต่อกับระบบอื่นได้ง่าย

**2. การ Mask ท่าทาง (Masked Gesture Modeling)**
ระหว่าง training จะซ่อน joint บางส่วนแล้วให้โมเดลทายเติม คล้ายกับ BERT ในงาน NLP วิธีนี้ทำให้โมเดลเรียนรู้ motion prior ที่แข็งแกร่ง และยังช่วยให้ **ใส่ท่าทางบางส่วนเป็น hint** แล้วให้ระบบเติมส่วนที่เหลือเองได้

**3. ผสมสัญญาณ Rhythm และ Content อย่างฉลาด**
เสียงพูดมีทั้ง "จังหวะ" และ "ความหมาย" EMAGE ใช้ self-attention ผสมสองสัญญาณนี้แบบ adaptive ต่างเฟรม ทำให้ท่าทางมีทั้งความสอดคล้องกับจังหวะ และสื่อความหมายของคำพูดด้วย

**4. ใช้ VQ-VAE แยกส่วนร่างกาย**
แบ่ง VQ-VAE ออกเป็น 4 ตัว สำหรับ ใบหน้า / ลำตัวบน / มือ / ลำตัวล่าง แยกกัน เพื่อป้องกันไม่ให้โมเดลมองข้ามส่วนที่เคลื่อนไหวน้อยกว่า

### ผลลัพธ์

EMAGE ได้คะแนนดีที่สุดใน BEAT2 benchmark และผู้ใช้เลือก output ของ EMAGE มากกว่า method อื่น 52.7% ในการทดสอบ perceptual study

---

## How to Apply to This Project

**Current pipeline in this workspace:**

```
Text Prompt → MDM (motion diffusion) → SMPL Motion (.npz) → HUGS Rendering → video
```

### Where EMAGE fits in

EMAGE operates in the same SMPL-X motion space that this project already uses. Here are concrete integration opportunities:

---

### Option 1 — Replace MDM with EMAGE for speech-driven animation

Instead of using a text prompt to generate motion via MDM, you could use EMAGE to generate motion from **speech audio** (`.wav` / `.m4a` files — you already have `dance10secs.m4a`, `kicks.m4a`, `tts_readback.wav` in this repo):

```
Speech Audio (.wav) → EMAGE → SMPL-X Motion (.npz) → HUGS Rendering → video
```

The output of EMAGE is joint rotations in Rot6D + global translations, which can be converted to the `.npz` format already consumed by `scripts/convert_mdm_results_to_hugs_npz.py`.

**Benefit:** Avatars that gesture naturally while speaking — not just generic MDM motion.

---

### Option 2 — Use EMAGE's masked hints to anchor MDM-generated motion

EMAGE can accept **partially predefined gestures** and fill in the rest. You could:
1. Generate a rough motion with MDM (text prompt → SMPL)
2. Pass that as a body hint (mask everything except key frames)
3. Let EMAGE refine it with audio synchronization

This gives you **text + audio control** over the avatar simultaneously.

---

### Option 3 — BEAT2 dataset for fine-tuning HUGS

BEAT2 provides 60h of high-quality SMPL-X sequences paired with speech. You could use these sequences to fine-tune or augment the HUGS rendering model for more natural, speech-synchronized poses — especially for talking-head scenes.

---

### Practical steps to start

| Step | What to do |
|------|-----------|
| 1 | Clone the [EMAGE repo](https://github.com/PantoMatrix/EMAGE) |
| 2 | Download BEAT2 dataset + EMAGE pretrained weights |
| 3 | Run EMAGE inference on one of the `.wav` files already in this repo (e.g. `tts_readback.wav`) |
| 4 | Convert EMAGE output (Rot6D joints) → `.npz` using `scripts/convert_mdm_results_to_hugs_npz.py` as reference |
| 5 | Feed the `.npz` into `scripts/run_text2hugs.py` (replacing the MDM motion source) |

---

### Key format notes

- EMAGE output: `g ∈ R^{T×(55×6 + 100 + 4 + 3)}` (55 joints Rot6D + FLAME + foot contact + translation)
- This project's SMPL `.npz`: uses `poses` (T×72 or T×156 for SMPL-X) + `trans` (T×3)
- A converter script similar to `convert_mdm_results_to_hugs_npz.py` will be needed for the Rot6D → axis-angle conversion

---

*Summary written 2026-04-23 | Paper: arXiv:2401.00374v5*
