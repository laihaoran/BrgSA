import pandas as pd
import numpy as np
import os
import argparse

# Argument parser to accept mainpath as input
def parse_args():
    parser = argparse.ArgumentParser(description='Process performance data')
    parser.add_argument('--mainpath', type=str, required=True, help='Path to the directory containing the performance files')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Use the mainpath provided via args
mainpath = args.mainpath
# mainpath = './performance/BrgSA_bs64_radchestct'
# 文件路径
auroc_file = os.path.join(mainpath, 'aurocs_bootstrap.xlsx')
auprc_file = os.path.join(mainpath, 'auprc_bootstrap.xlsx')
f1_file    = os.path.join(mainpath, 'f1_bootstrap.xlsx')
acc_file   = os.path.join(mainpath, 'acc_bootstrap.xlsx')
precision_file = os.path.join(mainpath, 'precision_bootstrap.xlsx')
recall_file    = os.path.join(mainpath, 'recall_bootstrap.xlsx')

# 单列（只做 Overall）
recall_1_file    = os.path.join(mainpath, 'recall_at1_bootstrap.xlsx')
precision_3_file = os.path.join(mainpath, 'precision_at3_bootstrap.xlsx')

#  读取Excel文件（逐病种多列）
df_auroc = pd.read_excel(auroc_file)
df_auprc = pd.read_excel(auprc_file)
df_f1    = pd.read_excel(f1_file)
df_acc   = pd.read_excel(acc_file)
df_precision = pd.read_excel(precision_file)
df_recall    = pd.read_excel(recall_file)

# 逐病种指标集合
metrics = {
    'AUROC': df_auroc,
    'AUPRC': df_auprc,
    'F1': df_f1,
    'Accuracy': df_acc,
    'Precision': df_precision,
    'Recall': df_recall,
}

results_list = []

# 逐病种明细
for metric_name, df in metrics.items():
    num_df = df.select_dtypes(include=[np.number])  # 只取数值列
    mean_values  = num_df.mean()
    std_values   = num_df.std()
    lower_values = np.percentile(num_df, 2.5, axis=0)
    upper_values = np.percentile(num_df, 97.5, axis=0)

    for pathology, mean, std, lower, upper in zip(num_df.columns, mean_values, std_values, lower_values, upper_values):
        results_list.append({
            'Metric': metric_name,
            'Pathology': pathology,
            'Mean': mean,
            'Std': std,
            '95% CI Lower': lower,
            '95% CI Upper': upper
        })

# 先把逐病种汇总成 DataFrame
results = pd.DataFrame(results_list)

# === 全部 Overall 放在最下方 ===
overall_results_list = []

# 1) 多列指标的 Overall（顺序与上面 metrics 一致）
for metric_name, df in metrics.items():
    num_df = df.select_dtypes(include=[np.number])
    overall_mean  = num_df.mean().mean()
    overall_std   = num_df.std().mean()
    overall_lower = np.percentile(num_df, 2.5, axis=0).mean()
    overall_upper = np.percentile(num_df, 97.5, axis=0).mean()
    overall_results_list.append({
        'Metric': metric_name,
        'Pathology': 'Overall',
        'Mean': overall_mean,
        'Std': overall_std,
        '95% CI Lower': overall_lower,
        '95% CI Upper': overall_upper
    })

# 2) 单列的 Recall@1 与 Precision@3 只统计 Overall，并追加在最后
def summarize_single_column_bootstrap(df_like, metric_name):
    vals = pd.DataFrame(df_like).select_dtypes(include=[np.number]).values.ravel()
    vals = vals[~np.isnan(vals)]
    mean  = float(np.mean(vals))
    std   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    lower = float(np.percentile(vals, 2.5))
    upper = float(np.percentile(vals, 97.5))
    return {
        'Metric': metric_name,
        'Pathology': 'Overall',
        'Mean': mean,
        'Std': std,
        '95% CI Lower': lower,
        '95% CI Upper': upper
    }

df_recall_1    = pd.read_excel(recall_1_file)
df_precision_3 = pd.read_excel(precision_3_file)
overall_results_list.append(summarize_single_column_bootstrap(df_recall_1, 'Recall@1'))
overall_results_list.append(summarize_single_column_bootstrap(df_precision_3, 'Precision@3'))

# 拼到最后
overall_results = pd.DataFrame(overall_results_list)
results = pd.concat([results, overall_results], ignore_index=True)

# 保存
output_file = os.path.join(mainpath, 'final_metrics_summary.xlsx')
results.to_excel(output_file, index=False, float_format="%.6f")
print(f"结果已保存到 {output_file}")
