#!/usr/bin/env python3
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

def load_volume(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)  # 数值用来显示即可
    affine = img.affine
    return data, affine

def maybe_reorient(data, affine, target_axcodes=None):
    """
    仅在指定 target_axcodes 时重排（不插值）。例如 ('L','P','S') 或 ('R','A','I')
    如果 target_axcodes is None，就保持原样。
    """
    if target_axcodes is None:
        return data

    cur_ornt = io_orientation(affine)          # 当前取向
    tgt_ornt = axcodes2ornt(target_axcodes)    # 目标取向
    xfm = ornt_transform(cur_ornt, tgt_ornt)   # 当前->目标 的重排/翻转
    data_r = apply_orientation(data, xfm)
    return data_r

def apply_model_view_ops(vol, perm="zyx", flip_axes=()):
    """
    用来“模拟你模型会做的轴顺序/翻转”，只做 numpy 级别变换：
    - perm: 期望的体素顺序字符串，来自 {'zyx', 'zxy', 'yzx', 'yxz', 'xzy', 'xyz'}
      假定输入 vol 是 (Z, Y, X)（Nibabel默认读取出来就可以这么理解）
    - flip_axes: 一个集合/列表，比如 ('x','y') 表示对相应轴做 np.flip
    """
    # 先按 perm 置换轴
    axis_map = {'z':0, 'y':1, 'x':2}
    if sorted(perm) != ['x','y','z']:
        raise ValueError("perm 必须是三个不同字符的排列，比如 zyx / xyz / yzx ...")
    order = tuple(axis_map[c] for c in perm)
    vol = np.transpose(vol, order)

    # 再按需要翻转
    for c in flip_axes:
        if c not in axis_map:
            raise ValueError("flip_axes 里的元素必须在 {'x','y','z'}")
        vol = np.flip(vol, axis=axis_map[c])
    return vol

def windowing(img, vmin=-1000.0, vmax=1000.0):
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)
    return (img * 255).astype(np.uint8)

def middle_slices(volume):
    """
    假设 volume 当前是 (Z, Y, X) 的意义（经过上面的 perm/flip 之后，我们就把它当成模型看到的“Z/Y/X”）
    返回三视图的中间切片：轴状/冠状/矢状
    """
    z, y, x = volume.shape
    axial   = volume[z // 2, :, :]
    coronal = volume[:, y // 2, :]
    sagitt  = volume[:, :, x // 2]
    return axial, coronal, sagitt

def show_side_by_side(vol1, vol2, vmin, vmax, name1="Image 1", name2="Image 2"):
    a1, c1, s1 = middle_slices(vol1)
    a2, c2, s2 = middle_slices(vol2)

    a1, c1, s1 = windowing(a1, vmin, vmax), windowing(c1, vmin, vmax), windowing(s1, vmin, vmax)
    a2, c2, s2 = windowing(a2, vmin, vmax), windowing(c2, vmin, vmax), windowing(s2, vmin, vmax)

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    plt.suptitle("Middle Slices — what your MODEL would see (after your ops)")

    axes[0,0].imshow(a1, cmap="gray"); axes[0,0].set_title(f"{name1} - Axial");   axes[0,0].axis("off")
    axes[1,0].imshow(c1, cmap="gray"); axes[1,0].set_title(f"{name1} - Coronal"); axes[1,0].axis("off")
    axes[2,0].imshow(s1, cmap="gray"); axes[2,0].set_title(f"{name1} - Sagittal");axes[2,0].axis("off")

    axes[0,1].imshow(a2, cmap="gray"); axes[0,1].set_title(f"{name2} - Axial");   axes[0,1].axis("off")
    axes[1,1].imshow(c2, cmap="gray"); axes[1,1].set_title(f"{name2} - Coronal"); axes[1,1].axis("off")
    axes[2,1].imshow(s2, cmap="gray"); axes[2,1].set_title(f"{name2} - Sagittal");axes[2,1].axis("off")

    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Just show what the MODEL will see: compare two NIfTI volumes by visual middle slices."
    )
    parser.add_argument("--img1", default="/data4/haoranlai/Dataset/CT-RATE/valid_fixed_256_128_high/valid_999/valid_999_a/valid_999_a_2.nii.gz",)
    parser.add_argument("--img2", default="/data4/haoranlai/Dataset/INSPECT/filtered_nii_all/PEc3c049.nii.gz")
    # 是否统一到一个取向，仅用于视觉一致；默认不重排，等于“照你原始的数据来”
    parser.add_argument("--orient", default="NONE",
                        help="目标取向（如 LPS/RAS/RAI/...）。默认 NONE 表示不重排，仅做你指定的 perm/flip。")
    # 用来模拟“模型吃数据”的轴顺序；我们把初始数据理解为 (Z,Y,X)
    parser.add_argument("--perm", default="zyx",
                        help="三字符排列，表示把当前 (Z,Y,X) 变换到什么顺序给模型看。例如 zyx(默认)/xyz/yzx 等")
    parser.add_argument("--flip", default="",
                        help="要翻转的轴，用字符集合表示，如 'x', 'yz', 'xyz'。为空表示不翻转")
    parser.add_argument("--vmin", type=float, default=-1000.0)
    parser.add_argument("--vmax", type=float, default=1000.0)
    parser.add_argument("--save", default="/data2/haoranlai/Project/gloria/plot_img.png", help="保存对比图路径（留空则直接显示）")
    args = parser.parse_args()

    # 1) 读取
    vol1, aff1 = load_volume(args.img1)
    vol2, aff2 = load_volume(args.img2)

    # 2) 可选：统一到一个取向（只是为了你肉眼更好对比；如果你想完全“照模型原样”，就用 --orient NONE）
    target_axcodes = None if args.orient.upper() == "NONE" else tuple(args.orient.upper())
    vol1 = maybe_reorient(vol1, aff1, target_axcodes)
    vol2 = maybe_reorient(vol2, aff2, target_axcodes)

    # 3) 按你模型的数据路径做同样的 numpy 变换（轴顺序 + 翻转）
    flips = tuple(args.flip.lower())  # e.g. "yz" -> ('y','z')
    vol1 = apply_model_view_ops(vol1, perm=args.perm.lower(), flip_axes=flips)
    vol2 = apply_model_view_ops(vol2, perm=args.perm.lower(), flip_axes=flips)

    # 4) 三视图展示（这就是模型“看到”的方向）
    fig = show_side_by_side(vol1, vol2, args.vmin, args.vmax, name1="Image 1", name2="Image 2")

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print("Saved to:", args.save)
    else:
        plt.show()

if __name__ == "__main__":
    main()
