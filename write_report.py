import pandas as pd
import json

# 输入 CSV 路径
csv_path = "/data4/haoranlai/Dataset/CT-RATE/Info/radiology_text_reports/train_reports.csv"
# 输出 JSON 路径
json_path = "./Dataset/CT-RATE-train-reports.json"

# 读取 CSV
df = pd.read_csv(csv_path)

reports = {}
for idx, row in df.iterrows():
    parts = []
    
    findings = str(row["Findings_EN"]).strip()
    impressions = str(row["Impressions_EN"]).strip()
    
    # 如果不是缺失或 Not given. 就加入
    if findings and findings.lower() != "nan" and findings != "Not given.":
        parts.append(findings)
    if impressions and impressions.lower() != "nan" and impressions != "Not given.":
        parts.append(impressions)
    
    # 拼接两个字段（用两个空格隔开）
    if parts:
        reports[str(idx)] = "  ".join(parts)

# 保存为 JSON 文件
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(reports, f, ensure_ascii=False, indent=4)

print(f"Saved {len(reports)} reports to {json_path}")
