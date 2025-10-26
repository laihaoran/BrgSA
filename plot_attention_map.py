import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from gloria import builder
from gloria import utils
from gloria import constants
from gloria import models

@torch.no_grad()
def save_nifti_like(arr_zyx: np.ndarray, ref_nii_path: str, out_path: str):
    ref = nib.load(ref_nii_path)
    img = nib.Nifti1Image(arr_zyx.astype(np.float32), affine=ref.affine, header=ref.header)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, out_path)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    ckpt = torch.load("/data2/haoranlai/Project/gloria/data/ckpt/3D_CLIP_text_5_16_1.0_image_CXRBertM3AE2_text_CXRBertM3AE2_proj_norm_bert_F_dictionary_Loss_1_0.5_1_with_GPT4Modify_bs64_1.0_1.0/2025_08_20_23_16_18/last.ckpt", map_location=device)
    cfg = ckpt["hyper_parameters"]
    
    ckpt_dict = ckpt["state_dict"]
    cfg.model.vision.model_name = 'make_3d_atten_model'
    # ===== 1) 构建模型（与你训练一致的 cfg）=====
    model = models.gloria_model_clip_proj_dict.CLIPProjDict(cfg).to(device)

    model_weights = model.state_dict()

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        if new_key in model_weights:
            fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    model.load_state_dict(ckpt_dict, strict=True)


    model.eval()

    # 可选：加载权重
    # ckpt = torch.load("path/to/ckpt.ckpt", map_location=device)
    # model.load_state_dict(ckpt["state_dict"], strict=False)

    # ===== 2) 准备单个病例 =====
    nii_path = "/data4/haoranlai/Dataset/CT-RATE/valid_fixed_256_128_high/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    text = "There is pulmonary embolism in the right lower lobe."

    # 用你类里现成的处理函数
    imgs = model.process_img(nii_path, device)    # Tensor [1, Z, Y, X] 或 [Z,Y,X] -> 你实现里应该会变成 [1,1,Z,Y,X]
    if imgs.dim() == 4:
        imgs = imgs.unsqueeze(0)  # [B=1, 1, Z, Y, X]
    batch = model.process_text(text, device)      # dict: caption_ids/attention_mask/token_type_ids
    batch["imgs"] = imgs
    
    # ===== 3) 计算 GWAR 热图 =====
    with torch.enable_grad():
        heatmaps, grid = model.compute_gwar_heatmap(batch)  # [1, Dz, Hy, Wx]
    heat = heatmaps[0]  # (Dz,Hy,Wx)

    # ===== 4) 上采样到体素并保存 =====
    full = model._upsample_to_image_like(heat, imgs[0:1])   # (Z,Y,X), 0~1
    full_u8 = (full.clamp(0,1) * 255).byte().cpu().numpy()
    out_path = "outputs/attention_gwar.nii.gz"
    save_nifti_like(full_u8, nii_path, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
