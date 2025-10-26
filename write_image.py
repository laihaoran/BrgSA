import os
import pandas as pd

# 配置路径
csv_path = "/data4/haoranlai/Dataset/CT-RATE/Info/radiology_text_reports/train_reports.csv"
data_dir = "/data4/haoranlai/Dataset/CT-RATE/train_fixed_256_128_high/"  # 你的数据根目录
save_path = "./Dataset/CT-RATE-train_image.csv"

# 读取CSV
df = pd.read_csv(csv_path)

# 构造绝对路径函数
def build_path(volume_name):
    # 去掉后缀 .nii.gz
    base_name = volume_name.replace(".nii.gz", "")
    # 按照规则 train_1/train_1_a/train_1_a_1.nii.gz
    parts = base_name.split("_")  # ["train", "1", "a", "1"]
    folder1 = f"{parts[0]}_{parts[1]}"       # train_1
    folder2 = f"{parts[0]}_{parts[1]}_{parts[2]}"  # train_1_a
    return os.path.join(data_dir, folder1, folder2, volume_name)

# 生成新的一列
df_out = pd.DataFrame()
df_out["Path"] = df["VolumeName"].apply(build_path)

# 保存为新的CSV
df_out.to_csv(save_path, index=False)

print(f"保存完成: {save_path}")
