import argparse
from pathlib import Path
import re
import json
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from gloria import builder
from gloria import utils
from gloria import constants
from gloria import models

# 你的类别文本（可改成从 --texts_json 读取）
TEXTS = {
    "0":  ["There is Arterial wall calcification"],
    "1":  ["There is Cardiomegaly"],
    "2":  ["There is Pericardial effusion"],
    "3":  ["There is Coronary artery wall calcification"],
    "4":  ["There is Hiatal hernia"],
    "5":  ["There is Lymphadenopathy"],
    "6":  ["There is Emphysema"],
    "7":  ["There is Atelectasis"],
    "8":  ["There is Lung nodule"],
    "9":  ["There is Lung opacity"],
    "10": ["There is Pulmonary fibrotic sequela"],
    "11": ["There is Pleural effusion"],
    "12": ["There is Peribronchial thickening"],
    "13": ["There is Consolidation"],
    "14": ["There is Bronchiectasis"],
    "15": ["There is Interlobular septal thickening"],
}


def slugify(s: str, maxlen: int = 40) -> str:
    s = s.strip()
    # 取一句中的关键词作为简短名（句式 "There is XXX" -> XXX）
    m = re.search(r"There is (.+)", s, re.IGNORECASE)
    if m:
        s = m.group(1)
    s = re.sub(r"[^A-Za-z0-9\-_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] if s else "cls"

@torch.no_grad()
def save_nifti_like(arr_zyx: np.ndarray, ref_nii_path: str, out_path: str):
    ref = nib.load(ref_nii_path)
    img = nib.Nifti1Image(arr_zyx.astype(np.float32), affine=ref.affine, header=ref.header)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, out_path)

def build_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    # 你原本的设置
    cfg.model.vision.model_name = 'make_3d_atten_model'
    model = models.gloria_model_clip_proj_dict.CLIPProjDict(cfg).to(device)

    model_weights = model.state_dict()
    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        if new_key in model_weights:
            fixed_ckpt_dict[new_key] = v
    model.load_state_dict(fixed_ckpt_dict, strict=True)
    model.eval()
    return model

def run_one_case(model, nii_path: str, text: str, device: str):
    """
    返回：上采样到体素分辨率后的热图 (numpy uint8, ZYX)
    """
    # 注意：compute_gwar_heatmap 需要梯度，这里开启 enable_grad 再关闭
    imgs = model.process_img(nii_path, device)   # -> [1,1,Z,Y,X]（若为 4D，则补 batch 维）
    if imgs.dim() == 4:
        imgs = imgs.unsqueeze(0)

    batch = model.process_text(text, device)
    batch["imgs"] = imgs

    # 计算 GWAR 热图
    with torch.enable_grad():
        heatmaps, grid = model.compute_gwar_heatmap(batch)  # [B=1, Dz, Hy, Wx]
    heat = heatmaps[0]  # (Dz,Hy,Wx)

    # 上采样回原体素大小并保存
    full = model._upsample_to_image_like(heat, imgs[0:1])   # (Z,Y,X), 0~1
    full_u8 = (full.clamp(0, 1) * 255).byte().cpu().numpy()
    return full_u8

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="包含 Path,TotalSegPath 列的 CSV（至少需要 Path）")
    ap.add_argument("--outdir", default="outputs_heatmaps", help="输出根目录")
    ap.add_argument("--ckpt", required=True, help="训练好的 ckpt 路径（.ckpt）")
    ap.add_argument("--texts_json", default="", help="可选：疾病文本的 JSON 文件路径（不填则用内置 TEXTS）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # 可选范围处理（大批量时有用）
    ap.add_argument("--start", type=int, default=0, help="从第 start 行（含）开始")
    ap.add_argument("--end", type=int, default=-1, help="到第 end 行（不含），-1 表示到末尾")
    return ap.parse_args()

def read_csv_paths(csv_path: str):
    """
    只依赖标准库读取，兼容性更好。要求有表头，且至少包含 'Path' 列。
    """
    import csv
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "Path" not in reader.fieldnames:
            raise ValueError("CSV 缺少 'Path' 列")
        for r in reader:
            rows.append(r)
    return rows

def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # 类别文本
    if args.texts_json:
        with open(args.texts_json, "r") as f:
            texts_map = json.load(f)
    else:
        texts_map = TEXTS

    # 将 { "0": ["..."], ... } 规整为 [(idx_int, text_str, short_name)]
    class_prompts = []
    for k in sorted(texts_map.keys(), key=lambda x: int(x)):
        lst = texts_map[k]
        if not lst:
            continue
        t = str(lst[0])
        cls_idx = int(k)
        short = slugify(t)
        class_prompts.append((cls_idx, t, short))

    # 构建模型
    model = build_model(args.ckpt, device)

    # 读 CSV
    rows = read_csv_paths(args.csv)
    n = len(rows)
    start = max(0, args.start)
    end = n if args.end < 0 else min(args.end, n)

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Total cases in CSV: {n}, processing rows [{start}, {end}) ...")
    print(f"[INFO] Output root: {out_root.resolve()}")

    for i in range(start, end):
        r = rows[i]
        nii_path = r["Path"].strip()
        if not nii_path:
            print(f"[WARN] Row {i}: empty Path, skip.")
            continue
        if not Path(nii_path).exists():
            print(f"[WARN] Row {i}: file not found: {nii_path}, skip.")
            continue

        case_stem = Path(nii_path).stem  # e.g., valid_1_a_1
        case_dir = out_root / case_stem
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{end}] {nii_path} -> saving to {case_dir}")

        for cls_idx, text, short in class_prompts:
            try:
                heat_u8 = run_one_case(model, nii_path, text, device)
                out_name = f"{cls_idx:02d}_{short}.nii.gz"
                out_path = case_dir / out_name
                save_nifti_like(heat_u8, nii_path, str(out_path))
            except Exception as e:
                print(f"[ERROR] Case={case_stem}, cls={cls_idx} ({short}): {e}")

    print("All done.")

if __name__ == "__main__":
    main()
