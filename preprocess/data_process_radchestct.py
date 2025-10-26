#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm


# =========================
# Core image processing
# =========================
def process_nii_sitk(
    sitk_img,
    target_spacing=(1.5, 1.5, 3.0),
    target_shape=(128, 256, 256),
    clamp_low=-1000,
    clamp_high=1000,
    pad_value=-1024,
):
    """Resample -> center crop/pad -> HU clamp. SimpleITK array order is (Z, Y, X)."""
    # force float32 for safety
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

    original_size = sitk_img.GetSize()       # (X, Y, Z)
    original_spacing = sitk_img.GetSpacing() # (X, Y, Z)

    # Compute new size (X, Y, Z)
    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    # Resample
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetOutputDirection(sitk_img.GetDirection())
    resample_filter.SetOutputOrigin(sitk_img.GetOrigin())
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resized_img = resample_filter.Execute(sitk_img)

    # Array is (Z, Y, X)
    resized_data = sitk.GetArrayFromImage(resized_img)
    current_shape = resized_data.shape  # (Z, Y, X)

    # Center crop if larger
    crop_total = [cs - ts if cs > ts else 0 for cs, ts in zip(current_shape, target_shape)]
    crop_lower = [math.ceil(ct / 2) for ct in crop_total]
    crop_upper = [ct - cl for ct, cl in zip(crop_total, crop_lower)]

    # Convert to XYZ for ITK
    crop_lower_xyz = [crop_lower[2], crop_lower[1], crop_lower[0]]
    crop_upper_xyz = [crop_upper[2], crop_upper[1], crop_upper[0]]

    if any(cs > ts for cs, ts in zip(current_shape, target_shape)):
        cropped_img = sitk.Crop(resized_img, crop_lower_xyz, crop_upper_xyz)
    else:
        cropped_img = resized_img

    # Pad with air if smaller
    cropped_data = sitk.GetArrayFromImage(cropped_img)
    current_shape = cropped_data.shape  # (Z, Y, X)

    padding_to_add = [(ts - cs) if ts > cs else 0 for ts, cs in zip(target_shape, current_shape)]
    padding_lower = [pad // 2 for pad in padding_to_add]
    padding_upper = [pad - pl for pad, pl in zip(padding_to_add, padding_lower)]

    padding_lower_xyz = [padding_lower[2], padding_lower[1], padding_lower[0]]
    padding_upper_xyz = [padding_upper[2], padding_upper[1], padding_upper[0]]

    if any(pad > 0 for pad in padding_to_add):
        padded_img = sitk.ConstantPad(
            cropped_img, padding_lower_xyz, padding_upper_xyz, constant=pad_value
        )
    else:
        padded_img = cropped_img

    # Clamp HU
    final_img = sitk.Clamp(padded_img, lowerBound=clamp_low, upperBound=clamp_high)
    return final_img


def load_npz_volume(npz_path, array_key="ct", flip_z=True):
    data = np.load(npz_path)
    arr = data[array_key]
    if flip_z:
        arr = np.flip(arr, axis=0)
    return arr


# =========================
# Sub-commands
# =========================
def cmd_write_dataset(args):
    """Merge multiple CSVs (RAD-ChestCT *_Abnormality_and_Location_Labels.csv) into a disease-level multi-hot table."""
    files = args.label_csvs  # list[str]
    out_path = args.out_csv

    df_list = []
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            df_list.append(df)
    if not df_list:
        raise ValueError("No CSV loaded. Please check --label-csvs.")

    combined_df = pd.concat(df_list, ignore_index=True)

    # Assume first col is ID
    id_column = combined_df.iloc[:, 0]
    df_data = combined_df.iloc[:, 1:]
    columns = df_data.columns

    abnormality_dict = {}
    for col in columns:
        if "*" in col:
            abn_type = col.split("*")[0]
            col_data = df_data[col].astype(bool)
            if abn_type not in abnormality_dict:
                abnormality_dict[abn_type] = col_data
            else:
                abnormality_dict[abn_type] = abnormality_dict[abn_type] | col_data

    merged_abnormality_df = pd.DataFrame(abnormality_dict)
    final_df = pd.concat([id_column, merged_abnormality_df], axis=1)

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"[write-dataset] Saved merged disease table: {out_path}")


def cmd_write_image_csv(args):
    """Create a CSV of existing image paths from merged table by NoteAcc_DEID."""
    merge_path = args.merge_csv
    image_root = args.image_root
    out_csv = args.out_csv
    ext = args.ext

    df = pd.read_csv(merge_path)
    if "NoteAcc_DEID" not in df.columns:
        raise KeyError("Column 'NoteAcc_DEID' not found in merge CSV.")

    ids = df["NoteAcc_DEID"].tolist()
    paths = [os.path.join(image_root, f"{p}{ext}") for p in ids]
    paths = [p for p in paths if os.path.exists(p)]

    Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(paths, columns=["Path"]).to_csv(out_csv, index=False)
    print(f"[write-image-csv] Saved {len(paths)} existing paths to: {out_csv}")


def _row_to_nifti(
    row,
    image_folder,
    out_root,
    npz_array_key,
    default_spacing_xyz,
    orientation_column,
    meta_columns,
    target_spacing,
    target_shape,
    clamp_low,
    clamp_high,
    pad_value,
    overwrite,
):
    file_name = row["VolumeAcc_DEID"]  # e.g., xxx.npz
    in_path = os.path.join(image_folder, file_name)
    out_path = os.path.join(out_root, file_name.replace("npz", "nii.gz"))
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    if (not overwrite) and os.path.exists(out_path):
        return f"[skip] exists: {out_path}"

    if not os.path.exists(in_path):
        return f"[missing] {in_path}"

    try:
        vol = load_npz_volume(in_path, array_key=npz_array_key, flip_z=True)
        sitk_image = sitk.GetImageFromArray(vol)

        # spacing (X, Y, Z)
        spacing = list(default_spacing_xyz)
        sitk_image.SetSpacing(spacing)

        # direction from CSV column string like "1,0,0,0,1,0,0,0,1" or "1,0,0,0,1,0"
        orientation_str = row[orientation_column]
        orientation_list = [float(num) for num in str(orientation_str).split(",")]
        if len(orientation_list) == 6:
            orientation_list.extend([0.0, 0.0, 1.0])  # complete to 3x3
        sitk_image.SetDirection(orientation_list)

        # optional metadata
        for k in meta_columns:
            if k in row.index:
                sitk_image.SetMetaData(k, str(row[k]))

        # Process (resample/crop/pad/clamp)
        sitk_image = process_nii_sitk(
            sitk_image,
            target_spacing=target_spacing,
            target_shape=target_shape,
            clamp_low=clamp_low,
            clamp_high=clamp_high,
            pad_value=pad_value,
        )

        sitk.WriteImage(sitk_image, out_path)
        return f"[ok] {out_path}"
    except Exception as e:
        return f"[error] {in_path} -> {e}"


def cmd_convert_from_csv(args):
    df = pd.read_csv(args.csv_file)
    rows = [df.iloc[i] for i in range(len(df))]

    meta_cols = args.meta_columns or []
    if isinstance(meta_cols, str):
        meta_cols = [c for c in meta_cols.split(",") if c]

    task_args = dict(
        image_folder=args.image_folder,
        out_root=args.out_root,
        npz_array_key=args.npz_key,
        default_spacing_xyz=tuple(map(float, args.default_spacing.split(","))),
        orientation_column=args.orientation_col,
        meta_columns=meta_cols,
        target_spacing=tuple(map(float, args.target_spacing.split(","))),
        target_shape=tuple(map(int, args.target_shape.split(","))),
        clamp_low=args.clamp_low,
        clamp_high=args.clamp_high,
        pad_value=args.pad_value,
        overwrite=args.overwrite,
    )

    with Pool(args.num_workers) as pool:
        it = pool.imap_unordered(
            lambda r: _row_to_nifti(r, **task_args),
            rows,
        )
        for _ in tqdm(it, total=len(rows)):
            pass

    print("[convert-from-csv] Done.")


def cmd_convert_rest(args):
    # identical to convert-from-csv but with different folders typically
    args.image_folder = args.rest_image_folder
    cmd_convert_from_csv(args)


def cmd_compare_nifti(args):
    img1_path = args.img1
    img2_path = args.img2

    img1 = nib.load(img1_path)
    img2 = nib.load(img2_path)

    hdr1 = img1.header
    hdr2 = img2.header

    keys1 = set(hdr1.keys())
    keys2 = set(hdr2.keys())
    common_keys = keys1.intersection(keys2)

    differences = {}
    for key in common_keys:
        val1 = hdr1[key]
        val2 = hdr2[key]
        try:
            equal = (val1 == val2).all()
        except Exception:
            equal = val1 == val2
        if not equal:
            differences[key] = (val1, val2)

    if differences:
        print("Differences found in the following header fields:")
        for key, (val1, val2) in differences.items():
            print(f"\nField: {key}\nImage 1: {val1}\nImage 2: {val2}")
    else:
        print("No differences found in the header fields.")


def cmd_list_missing(args):
    merge_path = args.merge_csv
    image_root = args.image_root
    ext = args.ext

    df = pd.read_csv(merge_path)
    if "NoteAcc_DEID" not in df.columns:
        raise KeyError("Column 'NoteAcc_DEID' not found in merge CSV.")

    ids = df["NoteAcc_DEID"].tolist()
    missing = [os.path.join(image_root, f"{p}{ext}") for p in ids if not os.path.exists(os.path.join(image_root, f"{p}{ext}"))]
    print(f"[list-missing] Missing count: {len(missing)}")
    if args.out_txt:
        Path(os.path.dirname(args.out_txt) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out_txt, "w") as f:
            for p in missing:
                f.write(p + "\n")
        print(f"[list-missing] Saved list to: {args.out_txt}")


def cmd_test_image(args):
    npz_path = args.npz_path
    key = args.npz_key
    arr = load_npz_volume(npz_path, array_key=key, flip_z=args.flip_z)
    print(f"npz: {npz_path}")
    print(f"key: {key}")
    print("max:", arr.max())
    print("min:", arr.min())
    print("mean:", arr.mean())
    print("shape:", arr.shape)


# =========================
# CLI
# =========================
def build_parser():
    p = argparse.ArgumentParser(description="RAD-ChestCT data utilities (all params via args).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # write-dataset
    pd1 = sub.add_parser("write-dataset", help="Merge *_Abnormality_and_Location_Labels.csv to disease-level table.")
    pd1.add_argument("--label-csvs", nargs="+", required=True,
                     help="List of CSVs, e.g. imgtrain_*.csv imgvalid_*.csv imgtest_*.csv")
    pd1.add_argument("--out-csv", required=True,
                     help="Output merged CSV path (e.g., merged_abnormality_labels_with_id.csv)")
    pd1.set_defaults(func=cmd_write_dataset)

    # write-image-csv
    pd2 = sub.add_parser("write-image-csv", help="Map NoteAcc_DEID to NIfTI paths and save a CSV (existing only).")
    pd2.add_argument("--merge-csv", required=True, help="Merged CSV (with NoteAcc_DEID column).")
    pd2.add_argument("--image-root", required=True, help="Directory where <NoteAcc_DEID>.nii.gz lives.")
    pd2.add_argument("--ext", default=".nii.gz", help="File extension to check (default: .nii.gz).")
    pd2.add_argument("--out-csv", required=True, help="Output CSV path to save paths.")
    pd2.set_defaults(func=cmd_write_image_csv)

    # convert-from-csv
    pd3 = sub.add_parser("convert-from-csv", help="Convert .npz to NIfTI based on a metadata CSV.")
    pd3.add_argument("--csv-file", required=True, help="Metadata CSV (e.g., CT_Scan_Metadata_Complete_35747.csv)")
    pd3.add_argument("--image-folder", required=True, help="Directory containing .npz volumes.")
    pd3.add_argument("--out-root", required=True, help="Output root for .nii.gz")
    pd3.add_argument("--npz-key", default="ct", help="Array key inside npz (default: ct)")
    pd3.add_argument("--default-spacing", default="0.8,0.8,0.8",
                     help="Default spacing as 'x,y,z' (used if CSV spacings not applied)")
    pd3.add_argument("--orientation-col", default="orig_orientation",
                     help="CSV column storing direction matrix string (comma-separated)")
    pd3.add_argument("--meta-columns", default="orig_square,orig_numslices,SliceThickness,SpacingBetweenSlices,"
                                               "Manufacturer,ManufacturerModelName,StudyDescription,Modality,"
                                               "SoftwareVersions,ConvolutionKernel,ProtocolName,orig_gantry_tilt",
                     help="Comma-separated CSV columns to copy into NIfTI header as metadata")
    pd3.add_argument("--target-spacing", default="1.5,1.5,3.0", help="Resample target spacing 'x,y,z'")
    pd3.add_argument("--target-shape", default="128,256,256", help="Target (Z,Y,X) after crop/pad")
    pd3.add_argument("--clamp-low", type=int, default=-1000, help="HU lower bound")
    pd3.add_argument("--clamp-high", type=int, default=1000, help="HU upper bound")
    pd3.add_argument("--pad-value", type=int, default=-1024, help="Padding constant for air")
    pd3.add_argument("--num-workers", type=int, default=18, help="Workers for multiprocessing")
    pd3.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    pd3.set_defaults(func=cmd_convert_from_csv)

    # convert-rest (reuse convert-from-csv args but override image-folder)
    pd4 = sub.add_parser("convert-rest", help="Convert remaining/error samples from another folder.")
    for a in pd3._actions:
        if a.dest not in ("help",):
            pd4._add_action(a)
    pd4.add_argument("--rest-image-folder", required=True, help="Alternate .npz folder for remaining/error samples.")
    pd4.set_defaults(func=cmd_convert_rest)

    # compare-nifti
    pd5 = sub.add_parser("compare-nifti", help="Compare NIfTI headers between two files.")
    pd5.add_argument("--img1", required=True)
    pd5.add_argument("--img2", required=True)
    pd5.set_defaults(func=cmd_compare_nifti)

    # list-missing
    pd6 = sub.add_parser("list-missing", help="List missing NIfTI files for a merged CSV of NoteAcc_DEID.")
    pd6.add_argument("--merge-csv", required=True)
    pd6.add_argument("--image-root", required=True)
    pd6.add_argument("--ext", default=".nii.gz")
    pd6.add_argument("--out-txt", default="", help="Optional path to save missing list.")
    pd6.set_defaults(func=cmd_list_missing)

    # test-image
    pd7 = sub.add_parser("test-image", help="Inspect one .npz file (max/min/mean/shape).")
    pd7.add_argument("--npz-path", required=True)
    pd7.add_argument("--npz-key", default="ct")
    pd7.add_argument("--flip-z", action="store_true")
    pd7.set_defaults(func=cmd_test_image)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
