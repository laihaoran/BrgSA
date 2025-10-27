# BrgSA: Bridged Semantic Alignment for Zero-shot 3D Medical Image Diagnosis

BrgSA is an open-source framework designed for **zero-shot 3D medical image diagnosis** and **cross-modal retrieval**.  
It introduces **Semantic Summarization** and **Cross-Modal Knowledge Interaction (CMKI)**  to jointly optimize
explicit (contrastive) and implicit (reconstruction) alignment between 3D CT volumes and radiology reports.

This repository provides a **complete end-to-end pipeline** — from data downloading and preprocessing to model training, evaluation, and inference — along with **pretrained weights** and the **best-performing checkpoint**.

---

## 🧩 Key Features

- **Full Training & Inference Pipeline** — from dataset preparation to evaluation  
- **3D Vision-Text Alignment** via dynamic semantic bridging  
- **Joint Reconstruction & Contrastive Optimization**  
- **Support for Multi-label Diagnosis & Cross-modal Retrieval**  
- **Pretrained and Fine-tuned Weights** publicly released  

---

## 📁 Repository Structure


---

## ⚙️ Installation

### 1. Create Environment
```bash
conda create -n brgsa python=3.10 -y
conda activate brgsa
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🩻 Dataset Preparation

### 📦 Supported Datasets

- **CT-RATE**  
  A large-scale 3D chest CT dataset with structured diagnostic reports.  
  🔗 Download: [https://huggingface.co/datasets/ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)

- **RAD-ChestCT**  
  External benchmark dataset with mapped labels for zero-shot evaluation.  
  🔗 Download: [https://zenodo.org/records/6406114#.Ytl6OXbMLAQ](https://zenodo.org/records/6406114#.Ytl6OXbMLAQ)


### Preprocess
All CT scans are resampled to (1.5, 1.5, 3.0) mm, cropped to (256, 256, 128)


```bash
#CT-RATE Preprocess
python preprocess/data_fix_metadata_train.py \
  --img-root /data/CT-RATE \
  --save-root /data/CT-RATE_fixed_256_128_high \
  --metadata-csv /data/CT-RATE/Info/metadata/train_metadata.csv \
  --subset train \
  --part_num 1 \
  --total_parts 1 \
    --num-workers 32 \
  --missing-csv label_data.csv

python preprocess/data_fix_metadata_valid.py \
  --img-root /data/CT-RATE \
  --save-root /data/CT-RATE_valid_fixed_256_128_high \
  --metadata-csv /data/CT-RATE/Info/metadata/dataset_metadata_validation_metadata.csv \
  --subset valid \
  --part_num 1 \
  --total_parts 1 \
  --num-workers 32 \
  --missing-csv label_data.csv
```




```bash
#RADChestCT preprocess
python preprocess/data_process_radchestct.py \
  --convert-from-csv \
  --csv-file /data/RAD_ChestCT/CT_Scan_Metadata_Complete_35747.csv \
  --image-folder /data/RAD_ChestCT/RadChestCT \
  --out-root /data/RAD_ChestCT/image \
  --npz-key ct \
  --default-spacing 0.8,0.8,0.8 \
  --orientation-col orig_orientation \
  --target-spacing 1.5,1.5,3.0 \
  --target-shape 128,256,256 \
  --clamp-low -1000 \
  --clamp-high 1000 \
  --pad-value -1024 \
  --num-workers 18
```

### 📄 Prompt and Annotation Details for CT-RATE-LT
All information related to the summarization process — including the **full GPT-4o prompt template**,  
**operation procedure**, **radiologist verification protocol**, **correction examples**, and **agreement statistics** —  
is documented in the supplementary file:

➡️ **[`supplymentation.pdf`](https://github.com/laihaoran/BrgSA/blob/main/supplementary.pdf)**

This PDF contains:
- The exact GPT-4o prompt text and parameter settings.  
- Step-by-step description of the annotation workflow.  
- Number and qualifications of human reviewers, review protocol, and correction rate.  
- Common GPT-4o error categories with real correction examples.  
- Example triplets (Original report → GPT-4o output → Final verified label).  

We also release a representative CT-RATE-LT annotations at:

**`./Dataset/CT-RATE-LT-label.csv`**

### 💾 Semantic Summarization
We also release a representative portion of the GPT-4o–generated at:

**`./Dataset/ctrate_train_unique_report_gpt4modify.json`**

This dataset includes:
- Original de-identified reports.  
- GPT-4o summarized findings and impressions.  


### 🪶 Pretrained Weights


| Model                 | Description                                      | Download                   |
| --------------------- | ------------------------------------------------ | -------------------------- |
| `M3AE_CXRBERT` | Pretrained M3AE CXR-BERT | `https://pan.baidu.com/s/1pwGSI9GYRVSkzWUE73kWig access code: p4cd` |
| `M3AE_Vision_Encoder` | Pretrained M3AE 3D ViT | `https://pan.baidu.com/s/1DMUm6BlOGP_4WVbUVYRYxg access code: 1rck` |
| `BrgSA_best_model`     | BrgSA best checkpoint on CT-RATE            | `https://pan.baidu.com/s/1ixMZ-JFYkDfubQ-MoRbJAw access code: 55u1`     |


### 🚀 Training

```bash
bash run.sh
```

### 🧾 Evaluation

```bash
bash test.sh
```

⚙️ Default Settings
| Component      | Configuration                                     |
| -------------- | ------------------------------------------------- |
| Vision Encoder | 3D ViT-B/16 (`224×224×112`, patch size `16×16×8`) |
| Text Encoder   | CXRBERT                                        |
| Optimizer      | AdamW (`lr=5e-5`, batch size=64)                   |
| Hardware       | H20 GPUs (≥80GB VRAM)                      |
| Datasets       | CT-RATE (internal) + RAD-ChestCT (external)       |



### 🧩 Supplementary Visualizations

Additional analysis materials are included in the supplementary file:
➡️ **[`supplymentation.pdf`](https://github.com/laihaoran/BrgSA/blob/main/supplementary.pdf)**


Feature visualization maps that illustrate the learned alignment space between image, text, and knowledge representations.

Representative examples demonstrating how the proposed CMKB improves cross-modal separation and interpretability.

These visualizations provide an intuitive overview of model performance and support the quantitative findings reported in the main paper.


### 📖 Citation

If this project helps your research, please cite:

```
@article{Lai2025BrgSA,
  title   = {Bridged Semantic Alignment for Zero-shot 3D Medical Image Diagnosis},
  author  = {Haoran Lai and Zihang Jiang and Qingsong Yao and Rongsheng Wang and Zhiyang He and Xiaodong Tao and Wei Wei and Weifu Lv and S. Kevin Zhou},
  journal = {preprint/under review},
  year    = {2025}
}
```


### 📜 License

This project is licensed under the Apache 2.0 License.
Datasets follow their respective original licenses.
