#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import os
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def _maybe_eval(v):
    """Safely parse a list-like string to Python object; if already parsed, return as-is."""
    if isinstance(v, str):
        return ast.literal_eval(v)
    return v


def process_row(row, problematic_files, img_root, save_root, subset):
    """
    Build paths from args and metadata, read & fix image header, resample/crop/pad/clamp, then save.
    """
    VolumeName = row["VolumeName"]
    dir1 = VolumeName.rsplit("_", 1)[0]
    dir2 = VolumeName.rsplit("_", 2)[0]

    in_filepath = os.path.join(img_root, subset, dir2, dir1, VolumeName)
    out_dir = os.path.join(save_root, subset, dir2, dir1)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_filepath = os.path.join(out_dir, os.path.basename(in_filepath))

    # Skip if already processed
    if os.path.exists(out_filepath):
        return

    if not os.path.exists(in_filepath):
        problematic_files.append(in_filepath)
        print(f"[Missing] {in_filepath}")
        return

    try:
        # Read Image
        image = sitk.ReadImage(in_filepath)

        # Spacing
        xy = _maybe_eval(row["XYSpacing"])
        x, y = map(float, xy)
        z = float(row["ZSpacing"])
        image.SetSpacing((x, y, z))

        # Origin
        image.SetOrigin(_maybe_eval(row["ImagePositionPatient"]))

        # Direction
        orientation = _maybe_eval(row["ImageOrientationPatient"])
        row_cosine, col_cosine = orientation[:3], orientation[3:6]
        z_cosine = np.cross(row_cosine, col_cosine).tolist()
        image.SetDirection(row_cosine + col_cosine + z_cosine)

        # Rescale to HU
        intercept = float(row["RescaleIntercept"])
        slope = float(row["RescaleSlope"])
        adjusted_hu = image * slope + intercept

        # Process (resample/crop/pad/clamp)
        out_img = process_nii_sitk(adjusted_hu)

        # Save
        sitk.WriteImage(out_img, out_filepath)

    except Exception as e:
        problematic_files.append(in_filepath)
        print(f"[Error] {in_filepath} -> {e}")


def process_nii_sitk(sitk_img):
    # Target spacing & shape (SimpleITK array is Z,Y,X)
    target_spacing = (1.5, 1.5, 3.0)
    target_shape = (128, 256, 256)  # (Z, Y, X)

    original_size = sitk_img.GetSize()       # (X, Y, Z)
    original_spacing = sitk_img.GetSpacing() # (X, Y, Z)

    # Compute new size (X, Y, Z)
    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    # Resample
    resample = sitk.ResampleImageFilter()
    resample.SetSize(new_size)
    resample.SetOutputSpacing(target_spacing)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)
    resized_img = resample.Execute(sitk_img)

    # Array shape is (Z, Y, X)
    resized_data = sitk.GetArrayFromImage(resized_img)
    current_shape = resized_data.shape  # (Z, Y, X)

    # Center crop if larger
    crop_total = [cs - ts if cs > ts else 0 for cs, ts in zip(current_shape, target_shape)]
    crop_lower = [math.ceil(ct / 2) for ct in crop_total]
    crop_upper = [ct - cl for ct, cl in zip(crop_total, crop_lower)]

    # Convert to XYZ for SimpleITK cropping
    crop_lower_xyz = [crop_lower[2], crop_lower[1], crop_lower[0]]
    crop_upper_xyz = [crop_upper[2], crop_upper[1], crop_upper[0]]

    if any(cs > ts for cs, ts in zip(current_shape, target_shape)):
        cropped_img = sitk.Crop(resized_img, crop_lower_xyz, crop_upper_xyz)
    else:
        cropped_img = resized_img

    cropped_data = sitk.GetArrayFromImage(cropped_img)
    current_shape = cropped_data.shape  # (Z, Y, X)

    # Pad with air (-1024) if smaller
    padding_to_add = [(ts - cs) if ts > cs else 0 for ts, cs in zip(target_shape, current_shape)]
    padding_lower = [pad // 2 for pad in padding_to_add]
    padding_upper = [pad - pl for pad, pl in zip(padding_to_add, padding_lower)]

    padding_lower_xyz = [padding_lower[2], padding_lower[1], padding_lower[0]]
    padding_upper_xyz = [padding_upper[2], padding_upper[1], padding_upper[0]]

    if any(pad > 0 for pad in padding_to_add):
        padded_img = sitk.ConstantPad(cropped_img, padding_lower_xyz, padding_upper_xyz, constant=-1024)
    else:
        padded_img = cropped_img

    # Clamp HU
    final_img = sitk.Clamp(padded_img, lowerBound=-1000, upperBound=1000)
    return final_img


def worker(args_tuple):
    return process_row(*args_tuple)


def write_csv(problematic_files, out_csv="label_data.csv"):
    volume_names = [os.path.basename(path) for path in problematic_files]

    labels = [
        "Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
        "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
        "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
        "Pleural effusion", "Mosaic attenuation pattern", "Peribronchial thickening",
        "Consolidation", "Bronchiectasis", "Interlobular septal thickening"
    ]

    data = {"VolumeName": volume_names}
    data.update({label: [0] * len(volume_names) for label in labels})
    pd.DataFrame(data).to_csv(out_csv, index=False)


def main(args):
    # Load metadata and select slice by part
    metadata = pd.read_csv(args.metadata_csv)

    total_rows = len(metadata)
    part_size = total_rows // args.total_parts
    remainder = total_rows % args.total_parts

    start = (args.part_num - 1) * part_size + min(args.part_num - 1, remainder)
    end = start + part_size + (1 if args.part_num <= remainder else 0)

    meta_slice = metadata.iloc[start:end]
    rows = [row for _, row in meta_slice.iterrows()]

    # Shared list for failures
    manager = mp.Manager()
    problematic_files = manager.list()

    # Pool
    pool = mp.Pool(args.num_workers)

    tasks = [
        (row, problematic_files, args.img_root, args.save_root, args.subset)
        for row in rows
    ]

    for _ in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
        pass

    pool.close()
    pool.join()

    write_csv(list(problematic_files), out_csv=args.missing_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valid-set preprocessing with explicit input/output roots.")
    # ★ 输入与输出根路径
    parser.add_argument("--img-root", type=str, required=True,
                        help="Root directory of raw images. e.g., /data/CT-RATE")
    parser.add_argument("--save-root", type=str, required=True,
                        help="Root directory to save processed images. e.g., /data/CT-RATE_valid_fixed_256_128_high")

    # 元数据与子集（默认就是 valid，保持与你原脚本一致）
    parser.add_argument("--metadata-csv", type=str, required=True,
                        help="Path to metadata CSV. e.g., /data/CT-RATE/Info/metadata/dataset_metadata_validation_metadata.csv")
    parser.add_argument("--subset", type=str, default="valid", choices=["train", "valid", "val", "test"],
                        help="Subfolder under img-root/save-root to read/save volumes (default: valid).")

    # 分片（与原脚本一致）
    parser.add_argument("--part_num", type=int, default=1, help="1-indexed part number to process.")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts to split the metadata.")

    # 并行与失败清单
    parser.add_argument("--num-workers", type=int, default=32, help="Number of worker processes.")
    parser.add_argument("--missing-csv", type=str, default="label_data.csv",
                        help="CSV file to write missing/problematic cases with zero labels.")

    args = parser.parse_args()
    main(args)
