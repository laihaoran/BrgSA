import multiprocessing as mp
import numpy as np
import pandas as pd
import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, average_precision_score, roc_auc_score
from eval import evaluate_internal
import math
import argparse

def find_threshold(probabilities, true_labels):
    """
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities.
        true_labels (numpy.ndarray): True labels.

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        confusion = confusion_matrix(true_labels, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        current_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if current_roc <= best_roc:
            best_roc = current_roc
            best_threshold = threshold

    return best_threshold

def find_ours_threshold(probabilities, true_labels):
    """
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        probabilities (numpy.ndarray): Predicted scores or probabilities (not limited to 0~1 range).
        true_labels (numpy.ndarray): True labels.

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_roc = 10000

    # Use unique sorted output values as thresholds
    thresholds = np.sort(np.unique(probabilities))

    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        confusion = confusion_matrix(true_labels, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]

        TP_r = TP / (TP + FN) if (TP + FN) > 0 else 0
        FP_r = FP / (FP + TN) if (FP + TN) > 0 else 0
        current_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))

        if current_roc <= best_roc:
            best_roc = current_roc
            best_threshold = threshold

    return best_threshold

def find_best_threshold_with_f1(probabilities, true_labels):
    """
    Finds the optimal threshold for binary classification based on weighted F1-score.

    Args:
        probabilities (numpy.ndarray): Predicted scores or probabilities.
        true_labels (numpy.ndarray): True labels.

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_f1 = 0

    # Use unique sorted output values as thresholds
    thresholds = np.sort(np.unique(probabilities))

    for threshold in thresholds:
        # Generate predictions based on the current threshold
        predictions = (probabilities > threshold).astype(int)

        # Calculate weighted F1-score
        current_f1 = f1_score(true_labels, predictions, average="weighted")

        # Update the best threshold if current F1 is better
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold

def _maybe_zscore(scores: np.ndarray,
                  apply_zscore: bool = False,
                  mu: np.ndarray = None,
                  sd: np.ndarray = None,
                  eps: float = 1e-8):
    """
    对 scores 做列内（按类别）z-score： (x - mu) / (sd + eps)
    若未提供 mu/sd，则在本次数据上估计并返回它们，便于复用。
    """
    X = scores.astype(np.float64, copy=False)
    if not apply_zscore:
        # 返回占位 mu/sd，方便外部复用接口一致
        return X, mu, sd
    if mu is None or sd is None:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
    Z = (X - mu) / (sd + eps)
    return Z, mu, sd



# ================= Top-k 指标函数（支持可选 z-score） =================
def strict_recall_at_k(scores: np.ndarray,
                       labels: np.ndarray,
                       k: int = 1,
                       skip_zero_pos: bool = True,
                       apply_zscore: bool = False,
                       z_mu: np.ndarray  = None,
                       z_sd: np.ndarray  = None) -> float:
    """
    严格的报告级 Recall@k：每样本 (#(Top-k ∩ Y))/|Y| 再取平均
    可选：列内 z-score 以做跨类校准
    """
    assert scores.shape == labels.shape, "scores 与 labels 形状必须一致"
    N, C = scores.shape
    k = int(min(max(1, k), C))

    scores_z, _, _ = _maybe_zscore(scores, apply_zscore, z_mu, z_sd)

    topk_idx = np.argpartition(scores_z, -k, axis=1)[:, -k:]          # (N, k)
    hits_k   = (labels[np.arange(N)[:, None], topk_idx] > 0.5).sum(axis=1)
    pos      = labels.sum(axis=1)

    if skip_zero_pos:
        mask = pos > 0
        if not np.any(mask):
            return float('nan')
        return (hits_k[mask] / pos[mask]).mean()
    else:
        return (hits_k / np.clip(pos, 1, None)).mean()


def precision_at_k(scores: np.ndarray,
                   labels: np.ndarray,
                   k: int = 3,
                   apply_zscore: bool = False,
                   z_mu: np.ndarray  = None,
                   z_sd: np.ndarray  = None) -> float:
    """
    报告级 Precision@k（PPV@k）：每样本 (#(Top-k ∩ Y))/k 再取平均
    可选：列内 z-score 以做跨类校准
    """
    assert scores.shape == labels.shape, "scores 与 labels 形状必须一致"
    N, C = scores.shape
    k = int(min(max(1, k), C))

    scores_z, _, _ = _maybe_zscore(scores, apply_zscore, z_mu, z_sd)

    topk_idx = np.argpartition(scores_z, -k, axis=1)[:, -k:]
    hits_k   = (labels[np.arange(N)[:, None], topk_idx] > 0.5).sum(axis=1)
    return (hits_k / k).mean()

# 定义单次bootstrap任务的函数
def bootstrap_iteration(args):
    """
    Perform a single bootstrap iteration and calculate metrics.

    Args:
        args (tuple): Arguments for the function, including index, labels, predicted, thresholds, pathologies, and data_dir.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    index, labels, predicted, thresholds, pathologies, data_dir = args
    np.random.seed(index)  # Set random seed for reproducibility
    indices = np.random.choice(range(len(labels)), size=len(labels), replace=True)
    sampled_labels = labels[indices]
    sampled_predicted = predicted[indices]

    # Evaluate AUROC
    dfs_auroc = evaluate_internal(sampled_predicted, sampled_labels, pathologies, data_dir)

    # ========== NEW: 严格 Recall@1 & Precision@3（基于同一套 scores）==========
    # 这里直接用 sampled_predicted 作为可比较的排序分数（越大越好）
    recall_at1_strict = strict_recall_at_k(sampled_predicted, sampled_labels, k=1, skip_zero_pos=True, apply_zscore=True)
    precision_at3 = precision_at_k(sampled_predicted, sampled_labels, k=3, apply_zscore=True)



    # Calculate metrics for each label
    f1s, accs, precisions, recalls, auprcs = [], [], [], [], []

    #  # ========= NEW: Recall@1 & mAP =========
    # # Recall@1  (single scalar for this bootstrap sample)
    # top1 = sampled_predicted.argmax(axis=1)
    # recall_at1 = np.mean([
    #     sampled_labels[i, top1[i]] for i in range(len(sampled_labels))
    # ])


    
    for i in range(len(pathologies)):
        prob = sampled_predicted[:, i]
        label = sampled_labels[:, i]
        threshold = thresholds[i]
        pred = (prob > threshold).astype(int)

        f1s.append(f1_score(label, pred, average="weighted"))
        accs.append(accuracy_score(label, pred))
        precisions.append(precision_score(label, pred))
        recalls.append(recall_score(label, pred))
        auprcs.append(average_precision_score(label, prob))

    return {
        "dfs_auroc": dfs_auroc,
        "dfs_f1": pd.DataFrame([f1s], columns=pathologies),
        "dfs_acc": pd.DataFrame([accs], columns=pathologies),
        "dfs_precision": pd.DataFrame([precisions], columns=pathologies),
        "dfs_recall": pd.DataFrame([recalls], columns=pathologies),
        "dfs_auprc": pd.DataFrame([auprcs], columns=pathologies),
        # NEW: 两个整体指标
        "recall_at1": pd.DataFrame([[recall_at1_strict]], columns=['Recall@1']),
        "precision_at3": pd.DataFrame([[precision_at3]], columns=['Precision@3']),
    }

# Argument parser to accept mainpath as input
def parse_args():
    parser = argparse.ArgumentParser(description='Process performance data')
    parser.add_argument('--mainpath', type=str, required=True, help='Path to the directory containing the performance files')
    return parser.parse_args()


def save_labels_to_csv(volumes, labels, pathologies, out_csv):
    """
    将样本ID (volumes) 和标签矩阵 (labels) 及疾病名称 (pathologies) 
    组装成一个CSV并保存。

    Args:
        volumes (list[str]): 样本 ID 列表
        labels (np.ndarray): 标签矩阵, shape = (N, C)
        pathologies (list[str]): 疾病名称列表, 长度 = C
        out_csv (str or Path): 输出文件路径
    """
    import os
    df = pd.read_csv(volumes)
    volumes = df['Path'].tolist()
    volumes = [os.path.basename(v) for v in volumes]
    labels = labels.astype(int)
    df_out = pd.DataFrame(labels, columns=pathologies)
    df_out.insert(0, "Volume", volumes)

    out_path = Path(out_csv)
    df_out.to_csv(out_path, index=False)
    print(f"保存成功: {out_path}")


# 多进程处理bootstrap
if __name__ == "__main__":

    # Parse arguments
    args = parse_args()

    # Use the mainpath provided via args
    data_dir = args.mainpath
    # Define constants
    # data_dir = "./performance/BrgSA_bs64_radchestct"
    labels = np.load(Path(data_dir) / 'labels_weights.npz')['data']
    predicted = np.load(Path(data_dir) / 'predicted_weights.npz')['data']

    # Thresholds list
    thresholds = []
    f1s = []
    accs = []
    precisions = []
    auprcs = []
    recalls = []
    numberlabel = labels.shape[1]
    # Find threshold for each label
    for i in range(numberlabel):
        logit = predicted[:, i]
        l = labels[:, i]
        prob = logit
        threshold = find_ours_threshold(prob, l)
        thresholds.append(threshold)

        pred = (prob > threshold).astype(int)
        label = l


    # # # ct-rate
    # pathologies = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
    #                'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy',
    #                'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
    #                'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
    #                'Bronchiectasis', 'Interlobular septal thickening']
    
    # # # radchestct
    # pathologies = ['Medical material', 'Calcification', 'Cardiomegaly', 'Pericardial effusion',
    #                 'Hiatal hernia', 'Lymphadenopathy',
    #                'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
    #                'Pleural effusion', 'Peribronchial thickening', 'Consolidation',
    #                'Bronchiectasis', 'Interlobular septal thickening']
    
    # # ct-rate-lt
    pathologies = [
        "Pneumonia",
        "Pneumothorax",
        "Empyema",
        "Tuberculosis",
        "Pulmonary fibrosis",
        "Lung cyst",
        "Bronchial thickening",
        "Lung mass",
        "Osteophyte",
        "Rib fracture",
        "Atherosclerosis",
        "Pericardial calcification",
        "Spondylosis",
        "Pleural calcification",
        "Pneumomediastinum"
    ]

    

    # # # radchetsct-lt
    # pathologies = [
    # "Pleural thickening",
    # "Coronary artery bypass graft",
    # "Infection",
    # "Tree-in-bud pattern",
    # "Lung resection",
    # "Debris",
    # "Air trapping",
    # "Cavitation",
    # "Scattered calcifications",
    # "Hemothorax",
    # "Heart valve replacement",
    # "Dilation or ectasia",
    # "Sternotomy",
    # "Lesion",
    # "Deformity",
    # "Fibrosis",
    # "Bronchiolitis",
    # "Ground-glass opacity",
    # "Tuberculosis",
    # "Mucous plugging",
    # "Mass",
    # "Pulmonary edema",
    # "Lucency",
    # "Arthritis",
    # "Pneumonia",
    # "Inflammation",
    # "Bronchitis",
    # "Fracture",
    # "Secretion",
    # "Congestion",
    # "Soft tissue abnormality",
    # "Breast surgery",
    # "Aspiration",
    # "Post-surgical changes",
    # "Pneumothorax",
    # "Honeycombing",
    # "Airspace disease",
    # "Heart failure",
    # "Plaque",
    # "Reticulation",
    # "Aneurysm",
    # "Coronary artery disease",
    # "Distention",
    # "Infiltrate",
    # "Transplant",
    # "Abnormal density",
    # "Interstitial lung disease",
    # "Scattered nodules",
    # "Bronchiolectasis",
    # "Atherosclerosis",
    # "Cyst",
    # "Band-like or linear opacity",
    # "Pericardial thickening",
    # "Granuloma",
    # "Pneumonitis",
    # "Cancer"
    # ]

    # INSPECT
    # pathologies = ['Pulmonary embolism']

    

    # Prepare storage for results
    concatenated_df_auroc = pd.DataFrame()
    concatenated_df_f1 = pd.DataFrame()
    concatenated_df_acc = pd.DataFrame()
    concatenated_df_precision = pd.DataFrame()
    concatenated_df_recall = pd.DataFrame()
    concatenated_df_auprc = pd.DataFrame()
    # NEW: 两个整体指标
    concatenated_df_recall_at1 = pd.DataFrame(columns=['Recall@1'])
    concatenated_df_precision_at3 = pd.DataFrame(columns=['Precision@3'])


    # Number of bootstrap iterations
    num_iterations = 5000

    # Prepare arguments for each iteration
    args = [
        (i, labels, predicted, thresholds, pathologies, data_dir)
        for i in range(num_iterations)
    ]

    # Use multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(bootstrap_iteration, args),
                total=num_iterations
            )
        )

    # Aggregate results
    for result in results:
        concatenated_df_auroc = pd.concat([concatenated_df_auroc, result["dfs_auroc"]])
        concatenated_df_f1 = pd.concat([concatenated_df_f1, result["dfs_f1"]])
        concatenated_df_acc = pd.concat([concatenated_df_acc, result["dfs_acc"]])
        concatenated_df_precision = pd.concat([concatenated_df_precision, result["dfs_precision"]])
        concatenated_df_recall = pd.concat([concatenated_df_recall, result["dfs_recall"]])
        concatenated_df_auprc = pd.concat([concatenated_df_auprc, result["dfs_auprc"]])
        # NEW
        concatenated_df_recall_at1 = pd.concat([concatenated_df_recall_at1, result["recall_at1"]])
        concatenated_df_precision_at3 = pd.concat([concatenated_df_precision_at3, result["precision_at3"]])

    # # Save results to Excel
    concatenated_df_auroc.to_excel(Path(data_dir) / 'aurocs_bootstrap.xlsx', index=False)
    concatenated_df_f1.to_excel(Path(data_dir) / 'f1_bootstrap.xlsx', index=False)
    concatenated_df_acc.to_excel(Path(data_dir) / 'acc_bootstrap.xlsx', index=False)
    concatenated_df_precision.to_excel(Path(data_dir) / 'precision_bootstrap.xlsx', index=False)
    concatenated_df_recall.to_excel(Path(data_dir) / 'recall_bootstrap.xlsx', index=False)
    concatenated_df_auprc.to_excel(Path(data_dir) / 'auprc_bootstrap.xlsx', index=False)
    # ---------- NEW ----------
     # NEW:
    concatenated_df_recall_at1.to_excel(Path(data_dir) / 'recall_at1_bootstrap.xlsx', index=False)
    concatenated_df_precision_at3.to_excel(Path(data_dir) / 'precision_at3_bootstrap.xlsx', index=False)


    # print mean of recall@1
    print("Recall@1 (strict) mean:", concatenated_df_recall_at1['Recall@1'].mean())
    print("Precision@3 mean:", concatenated_df_precision_at3['Precision@3'].mean())

