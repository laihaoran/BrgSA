# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Union
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk


# ------------------ 基础工具 ------------------

def grid_from_img(img_size: Tuple[int,int,int], patch_size=(16,16,8)):
    """img_size=(X,Y,Z) -> grid=(Dz,Hy,Wx) ; Conv3d输出顺序对应 (D',H',W')，其中 D' 对 Z 维。"""
    X, Y, Z = img_size
    px, py, pz = patch_size
    assert X % px == 0 and Y % py == 0 and Z % pz == 0, \
        f"img_size {img_size} 必须能被 patch_size {patch_size} 整除"
    Dz = Z // pz
    Hy = Y // py
    Wx = X // px
    return Dz, Hy, Wx

def strip_cls(tokens_1_L_D: torch.Tensor, prod: int) -> torch.Tensor:
    """若 L==prod+1 则去 CLS，若 L==prod 直接返回，否则报错。"""
    assert tokens_1_L_D.dim() == 3 and tokens_1_L_D.size(0) == 1, \
        f"expect (1,L,D), got {list(tokens_1_L_D.shape)}"
    L = tokens_1_L_D.size(1)
    if L == prod:
        return tokens_1_L_D
    if L == prod + 1:
        return tokens_1_L_D[:, 1:, :]
    raise ValueError(f"L={L} 与 grid产出的patch数 {prod} 不匹配（或不含CLS）。")

def tokens_to_grid(tokens_1_L_D: torch.Tensor, grid: Tuple[int,int,int]) -> torch.Tensor:
    """(1,L,D) -> (Dz,Hy,Wx,D)；展平顺序与 PatchEmbed3D 一致（W 最快，H 次之，D 最慢）。"""
    Dz, Hy, Wx = grid
    t = strip_cls(tokens_1_L_D, Dz * Hy * Wx)   # (1, L_no_cls, D)
    t = t.squeeze(0).contiguous()               # (L, D)
    t = t.view(Dz, Hy, Wx, -1).contiguous()     # (Dz,Hy,Wx,D)
    return t

def upsample_to_image(heat_DzHyWx: torch.Tensor, img_size_xyz: Tuple[int,int,int]) -> torch.Tensor:
    """(Dz,Hy,Wx) -> (Z,Y,X) 使用 3D trilinear 上采样。"""
    Z, Y, X = img_size_xyz[2], img_size_xyz[1], img_size_xyz[0]
    x = heat_DzHyWx.unsqueeze(0).unsqueeze(0)  # (1,1,Dz,Hy,Wx)
    out = F.interpolate(x, size=(Z, Y, X), mode="trilinear", align_corners=False)
    out = out.squeeze(0).squeeze(0)            # (Z,Y,X)
    # 归一化到[0,1]（以防极端值）
    mn, mx = torch.min(out), torch.max(out)
    if float(mx - mn) > 1e-12:
        out = (out - mn) * 255.0 / (mx - mn)
    else:
        out = torch.zeros_like(out)
    return out

def save_heatmap_nifti(heat_ZYX: np.ndarray, ref_img_path: str, out_path: str):
    """复制 ref 的空间信息保存 NIfTI。heat_ZYX 应为 float32，(Z,Y,X)。"""
    ref = sitk.ReadImage(ref_img_path)
    img = sitk.GetImageFromArray(heat_ZYX.astype(np.uint8))
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, out_path)

def load_tokens(pt_path: str, device="cpu") -> torch.Tensor:
    """加载 image tokens，返回 (1,L,D) 的 Tensor。支持 dict 包裹。"""
    obj = torch.load(pt_path, map_location=device)
    if isinstance(obj, dict):
        # 常见key名尝试
        for k in ["tokens", "image_tokens", "features", "x"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                break
        else:
            # dict里只有一个tensor也接受
            only_tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            if len(only_tensors) == 1:
                t = only_tensors[0]
            else:
                raise ValueError(f"{pt_path} 内未找到可识别的 tokens。keys={list(obj.keys())[:10]}")
    elif isinstance(obj, torch.Tensor):
        t = obj
    else:
        raise ValueError(f"不支持的 tokens 文件类型：{type(obj)}")
    assert t.dim() == 3 and t.size(0) == 1, f"expect (1,L,D), got {list(t.shape)}"
    return t

# def load_text_embeds(embed_path: str, device="cpu") -> Dict[str, torch.Tensor]:
#     """
#     加载保存好的类别文本向量（同一对齐空间）。
#     支持：
#       - .pt/.pth: dict{name->tensor(E)} 或 { 'embeds': 2D(C,E), 'classes': [name...] }
#       - .npy: 若为 2D(C,E)，需配合 --class_list；若为 object dict 则 np.load(..., allow_pickle=True).item()
#     返回 dict: {class_name: (E,) tensor}
#     """
#     ext = os.path.splitext(embed_path)[1].lower()
#     name2vec = {}
#     if ext in [".pt", ".pth"]:
#         obj = torch.load(embed_path, map_location=device)
#         if isinstance(obj, dict):
#             if all(isinstance(v, torch.Tensor) and v.dim() == 1 for v in obj.values()):
#                 # 直接 {name: vec}
#                 for k, v in obj.items():
#                     name2vec[str(k)] = v.to(device).float()
#             elif "embeds" in obj and "classes" in obj:
#                 embeds = obj["embeds"]
#                 classes = obj["classes"]
#                 assert embeds.dim() == 2 and len(classes) == embeds.size(0)
#                 for name, row in zip(classes, embeds):
#                     name2vec[str(name)] = row.to(device).float()
#             else:
#                 # 兜底：只有一个 2D tensor（无类别名）
#                 only_tensors = [v for v in obj.values() if isinstance(v, torch.Tensor) and v.dim() == 2]
#                 if len(only_tensors) == 1:
#                     arr = only_tensors[0].to(device).float()
#                     # 需要使用者传 --class_list 来对应
#                     raise ValueError("检测到 2D 文本嵌入但缺少类别名，请使用 --class_list 指定。")
#                 raise ValueError("无法识别的文本向量文件结构。")
#         else:
#             raise ValueError("期望 .pt 文件保存为 dict。")
#     elif ext == ".npy":
#         obj = np.load(embed_path, allow_pickle=True)
#         if obj.dtype == object:
#             m = obj.item()
#             for k, v in m.items():
#                 name2vec[str(k)] = torch.as_tensor(v, device=device, dtype=torch.float32)
#         else:
#             # 2D(C,E)
#             raise ValueError("检测到 2D numpy，但缺少类别名，请使用 --class_list 指定。")
#     else:
#         raise ValueError(f"不支持的文本向量文件：{embed_path}")
#     # 单位化
#     for k in list(name2vec.keys()):
#         v = name2vec[k]
#         name2vec[k] = F.normalize(v, dim=-1)
#     return name2vec

def load_text_embeds(
    embed_path: str,
    device: str = "cpu",
    class_name: Optional[str] = None,
    class_list: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    加载保存好的类别文本向量（同一对齐空间），并统一返回 {class_name: (E,) tensor}。
    支持：
      - .pt/.pth:
          * dict{name->1D tensor}（直接返回）
          * dict{'embeds': 2D(C,E), 'classes': [name...]}
          * 单个 Tensor: 1D(E,) 需 class_name（若缺省，用文件名 stem）
          * 单个 Tensor: 2D(C,E) 需 class_list
      - .npy:
          * object-dict（键->向量/行）
          * 单个 1D(E,) 需 class_name（若缺省，用文件名 stem）
          * 2D(C,E) 需 class_list
    """
    ext = os.path.splitext(embed_path)[1].lower()
    stem = os.path.splitext(os.path.basename(embed_path))[0]
    name2vec: Dict[str, torch.Tensor] = {}

    def _normalize_all(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in d.items():
            t = torch.as_tensor(v, device=device, dtype=torch.float32)
            out[k] = F.normalize(t, dim=-1)
        return out

    if ext in [".pt", ".pth"]:
        obj = torch.load(embed_path, map_location=device)

        # --- 直接是 Tensor 的情况 ---
        if isinstance(obj, torch.Tensor):
            if obj.dim() == 1:
                # 单个向量
                cname = class_name if class_name else stem
                name2vec[cname] = obj.to(device).float()
                return _normalize_all(name2vec)
            elif obj.dim() == 2:
                # 多类矩阵
                if not class_list or len(class_list) != obj.size(0):
                    raise ValueError(
                        f"检测到 2D 文本嵌入 (shape={tuple(obj.shape)})，"
                        f"需要提供长度为 {obj.size(0)} 的 class_list。"
                    )
                for cname, row in zip(class_list, obj):
                    name2vec[str(cname)] = row.to(device).float()
                return _normalize_all(name2vec)
            else:
                raise ValueError(f"不支持的 Tensor 维度: {obj.dim()}")

        # --- 字典的情况 ---
        if isinstance(obj, dict):
            # 1) dict{name -> 1D tensor}
            if all(isinstance(v, torch.Tensor) and v.dim() == 1 for v in obj.values()):
                for k, v in obj.items():
                    name2vec[str(k)] = v.to(device).float()
                return _normalize_all(name2vec)

            # 2) {'embeds': 2D(C,E), 'classes': [...]}
            if "embeds" in obj and "classes" in obj:
                embeds = obj["embeds"]
                classes = obj["classes"]
                if not (isinstance(embeds, torch.Tensor) and embeds.dim() == 2):
                    raise ValueError("embeds 需为 2D Tensor")
                if len(classes) != embeds.size(0):
                    raise ValueError("classes 数量与 embeds 行数不匹配")
                for cname, row in zip(classes, embeds):
                    name2vec[str(cname)] = torch.as_tensor(row, device=device, dtype=torch.float32)
                return _normalize_all(name2vec)

            # 3) 兜底：字典里只有一个 2D tensor，无类别名
            only_tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            if len(only_tensors) == 1 and only_tensors[0].dim() == 2:
                arr = only_tensors[0]
                if not class_list or len(class_list) != arr.size(0):
                    raise ValueError(
                        f"检测到字典内唯一 2D Tensor (shape={tuple(arr.shape)})，"
                        f"但缺少匹配的 class_list。"
                    )
                for cname, row in zip(class_list, arr):
                    name2vec[str(cname)] = torch.as_tensor(row, device=device, dtype=torch.float32)
                return _normalize_all(name2vec)

            raise ValueError("无法识别的 .pt/.pth 文本向量结构。")

        raise ValueError("期望 .pt/.pth 为 Tensor 或 dict。")

    elif ext == ".npy":
        obj = np.load(embed_path, allow_pickle=True)

        # object-dict
        if obj.dtype == object:
            m = obj.item()
            if not isinstance(m, dict):
                raise ValueError("object npy 需为 dict")
            for k, v in m.items():
                t = torch.as_tensor(v, device=device, dtype=torch.float32)
                if t.dim() == 1:
                    name2vec[str(k)] = t
                elif t.dim() == 2:
                    # 若值本身是 2D，多行命名方式：k_i
                    for i, row in enumerate(t):
                        name2vec[f"{k}_{i}"] = row
                else:
                    raise ValueError(f"字典中值 Tensor 维度不支持: {t.dim()}")
            return _normalize_all(name2vec)

        # 普通数组：1D 或 2D
        arr = torch.as_tensor(obj, device=device, dtype=torch.float32)
        if arr.dim() == 1:
            cname = class_name if class_name else stem
            name2vec[cname] = arr
            return _normalize_all(name2vec)
        elif arr.dim() == 2:
            if not class_list or len(class_list) != arr.size(0):
                raise ValueError(
                    f"检测到 2D numpy (shape={tuple(arr.shape)})，但缺少匹配的 class_list。"
                )
            for cname, row in zip(class_list, arr):
                name2vec[str(cname)] = row
            return _normalize_all(name2vec)
        else:
            raise ValueError(f"不支持的 numpy 数组维度: {arr.dim()}")

    else:
        raise ValueError(f"不支持的文本向量文件扩展名：{ext}")
    

# ------------------ 主逻辑：文本引导的 per-class heatmap ------------------

@torch.no_grad()
def build_class_heatmaps_from_projected_tokens(
    tokens_pt_path: str,
    text_embed_path: str,
    img_size_xyz: Tuple[int,int,int],
    ref_nifti_path: str,
    out_dir: str,
    patch_size_xyz=(16,16,8),
    device="cpu",
    class_list: Union[str, None] = None  # 当 text_embed_path 只有矩阵而无名字时，传入逗号分隔的类名列表
):
    """
    假设 image tokens 与文本向量已在同一对齐空间（都可直接做 cosine similarity）。
    为每个类别生成一张 NIfTI 热力图：cos( patch_token, class_text_embed )。
    """
    device = torch.device(device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) 加载 image tokens
    tokens = load_tokens(tokens_pt_path, device=device)  # (1,L,D)

    # 2) 计算 patch 网格，并还原到 (Dz,Hy,Wx,D)
    grid = grid_from_img(img_size_xyz, patch_size_xyz)
    tokens_grid = tokens_to_grid(tokens, grid)           # (Dz,Hy,Wx,D)

    # 3) 归一化 patch 向量（逐patch L2）
    Dz, Hy, Wx, D = tokens_grid.shape
    patches = tokens_grid.reshape(-1, D)                 # (N_patches, D)
    patches = F.normalize(patches, dim=-1)               # 单位化

    # 4) 加载类别文本向量（已单位化）
    try:
        name2vec = load_text_embeds(text_embed_path, device=device)
    except ValueError as e:
        # 如果文本嵌入是 2D 矩阵且没有名字，这里处理 class_list
        if "缺少类别名" in str(e) and class_list is not None:
            # 重新以 numpy 方式加载
            ext = os.path.splitext(text_embed_path)[1].lower()
            if ext == ".npy":
                arr = np.load(text_embed_path)
                assert arr.ndim == 2, "文本向量应为 2D 矩阵 (C,E)"
                names = [s.strip() for s in class_list.split(",")]
                assert len(names) == arr.shape[0], "class_list 数量需与嵌入行数一致"
                name2vec = {n: F.normalize(torch.as_tensor(arr[i], device=device, dtype=torch.float32), dim=-1)
                            for i, n in enumerate(names)}
            else:
                raise
        else:
            raise

    # 5) 对每个类别计算相似度并保存
    for cls_name, tvec in name2vec.items():
        # (N_patches,) = (N_patches,D) @ (D,)
        sim = torch.matmul(patches, tvec)               # 余弦相似度（两边已单位化）
        sim = sim.view(Dz, Hy, Wx).contiguous()

        # 归一化到 0-1（可视化更直观）
        mn, mx = torch.min(sim), torch.max(sim)
        heat = (sim - mn) / (mx - mn + 1e-12)
        

        # 上采样到体素级 (Z,Y,X)
        heat_full = upsample_to_image(heat, img_size_xyz)

        

        heat_full = heat_full.round().to(torch.uint8)             # 转为整数型 0~255

        # import ipdb; ipdb.set_trace()

        # 保存 NIfTI
        out_path = os.path.join(out_dir, f"{Path(tokens_pt_path).stem}__{cls_name}.nii.gz")
        save_heatmap_nifti(heat_full.cpu().numpy(), ref_nifti_path, out_path)
        print(f"[Saved] {out_path}")

    print("✅ All class-conditioned heatmaps saved.")


# ------------------ CLI ------------------

def parse_img_size(s: str):
    # "X,Y,Z"
    parts = [int(x.strip()) for x in s.split(",")]
    assert len(parts) == 3
    return (parts[0], parts[1], parts[2])

def parse_patch_size(s: str):
    parts = [int(x.strip()) for x in s.split(",")]
    assert len(parts) == 3
    return (parts[0], parts[1], parts[2])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CLIP per-class heatmaps from projected image tokens and saved text embeddings.")
    ap.add_argument("--tokens_pt", required=True, help="Path to image tokens .pt (shape 1xLxD or dict with a tensor).")
    ap.add_argument("--text_embeds", required=True, help="Path to saved text embeddings (.pt/.pth/.npy).")
    ap.add_argument("--img_size", required=True, type=parse_img_size, help="Image size as X,Y,Z (divisible by patch size).")
    ap.add_argument("--ref_image", required=True, help="Reference NIfTI path (for spacing/origin/direction).")
    ap.add_argument("--out_dir", required=True, help="Output directory for per-class heatmaps.")
    ap.add_argument("--patch_size", default="16,16,8", type=parse_patch_size, help="Patch size as px,py,pz (default 16,16,8).")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--class_list", default=None, help="Comma-separated class names if text_embeds is a 2D matrix without names.")
    args = ap.parse_args()

    build_class_heatmaps_from_projected_tokens(
        tokens_pt_path=args.tokens_pt,
        text_embed_path=args.text_embeds,
        img_size_xyz=args.img_size,
        ref_nifti_path=args.ref_image,
        out_dir=args.out_dir,
        patch_size_xyz=args.patch_size,
        device=args.device,
        class_list=args.class_list
    )
