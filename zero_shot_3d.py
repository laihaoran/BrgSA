import torch
import gloria
import pandas as pd 
import json
import numpy as np
# import tensorboard
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score,  balanced_accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import pickle as pkl
import os
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
from collections import Counter
import ipdb
import time
import math
from nltk.tokenize import RegexpTokenizer
import re
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# 
# def find_ours_threshold(probabilities, true_labels):
#     """
#     Finds the optimal threshold for binary classification based on ROC curve.

#     Args:
#         probabilities (numpy.ndarray): Predicted scores or probabilities (not limited to 0~1 range).
#         true_labels (numpy.ndarray): True labels.

#     Returns:
#         float: Optimal threshold.
#     """
#     best_threshold = 0
#     best_roc = 10000

#     # Use unique sorted output values as thresholds
#     thresholds = np.sort(np.unique(probabilities))

#     for threshold in thresholds:
#         predictions = (probabilities > threshold).astype(int)
#         confusion = confusion_matrix(true_labels, predictions)
#         TP = confusion[1, 1]
#         TN = confusion[0, 0]
#         FP = confusion[0, 1]
#         FN = confusion[1, 0]

#         TP_r = TP / (TP + FN) if (TP + FN) > 0 else 0
#         FP_r = FP / (FP + TN) if (FP + TN) > 0 else 0
#         current_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))

#         if current_roc <= best_roc:
#             best_roc = current_roc
#             best_threshold = threshold

#     return best_threshold


def find_ours_threshold(probabilities, true_labels, alpha=0.5, method="roc"):
    """
    Finds the optimal threshold for binary classification.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities.
        true_labels (numpy.ndarray): True labels.
        alpha (float): Weighting factor for TPR and FPR (only used in ROC method).
        method (str): Optimization method ("roc" or "pr").

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_metric = float("inf") if method == "roc" else float("-inf")

    # Use unique sorted probabilities or finer granularity
    thresholds = np.sort(np.unique(probabilities))
    # thresholds = np.linspace(0, 1, num=100)

    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        confusion = confusion_matrix(true_labels, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]

        if method == "roc":
            # Calculate TPR and FPR
            TP_r = TP / (TP + FN) if (TP + FN) > 0 else 0
            FP_r = FP / (FP + TN) if (FP + TN) > 0 else 0
            current_metric = alpha * ((1 - TP_r) ** 2) + (1 - alpha) * (FP_r ** 2)
            if current_metric < best_metric:
                best_metric = current_metric
                best_threshold = threshold

        elif method == "pr":
            # Precision-Recall optimization
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_metric:
                best_metric = f1
                best_threshold = threshold

    return best_threshold


def balanced_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm.diagonal() / cm.sum(axis=1)  # 计算每个类别的TPR
    bacc = sum(tpr) / len(tpr)
    return bacc # 计算平均值

def eval_precision(results, gt_labels):
    y_pred = np.asarray(results) > 0
    tp = gt_labels * y_pred
    fp = (1 - gt_labels) * y_pred
    under = np.sum(tp + fp, axis=0)
    under = np.where(under == 0, 1, under)
    precision = np.sum(tp, axis=0)  / under
    mAP = np.mean(precision)
    return mAP, precision

def eval_recall(results, gt_labels):
    y_pred = np.asarray(results) > 0
    tp = gt_labels * y_pred
    fn = gt_labels * (1 - y_pred)
    recall = np.sum(tp, axis=0)  / np.sum(tp + fn, axis=0)
    mAR = np.mean(recall)
    return mAR, recall


# tpr =  tp / (tp + fn)
# tnr = tn / (fp + tn )
def eval_bacc(y_pred, gt_labels):
    tp = gt_labels * y_pred
    tn = (1 - gt_labels) * (1 - y_pred)
    fp = (1 - gt_labels) * y_pred
    fn = gt_labels * (1 - y_pred)
    assert np.sum(tp + tn + fp + fn) == np.shape(gt_labels)[0]*np.shape(gt_labels)[1]
    tpr =  np.sum(tp, axis=0)  / np.sum(tp + fn, axis=0) 
    tnr = np.sum(tn, axis=0)  / np.sum(fp + tn, axis=0)
    per_class_acc = (tpr + tnr) / 2
    acc = np.mean(per_class_acc)
    return acc, per_class_acc

def eval_SE(y_pred, gt_labels):
    tp = gt_labels * y_pred
    tn = (1 - gt_labels) * (1 - y_pred)
    fp = (1 - gt_labels) * y_pred
    fn = gt_labels * (1 - y_pred)
    assert np.sum(tp + tn + fp + fn) == np.shape(gt_labels)[0]*np.shape(gt_labels)[1]
    tpr =  np.sum(tp, axis=0)  / np.sum(tp + fn, axis=0) 
    tnr = np.sum(tn, axis=0)  / np.sum(fp + tn, axis=0)
    sen = np.mean(tpr)
    spe = np.mean(tnr)
    return sen, spe


def eval_auc(results, gt_labels):
    macro_auc = roc_auc_score(gt_labels, results , average="macro")
    micro_auc = roc_auc_score(gt_labels, results,  average="micro")
    weighted_auc = roc_auc_score(gt_labels, results,  average="weighted")
    per_auc = roc_auc_score(gt_labels, results,  average=None)
    return macro_auc, micro_auc, weighted_auc, per_auc

def eval_F1(y_pred, gt_labels):
    micro_f1 = f1_score(gt_labels, y_pred, average='micro')
    macro_f1 = f1_score(gt_labels, y_pred, average='macro')
    weighted_f1 = f1_score(gt_labels, y_pred, average='weighted')
    # print(f"Total macro F1-score: {macro_f1}")
    return micro_f1, macro_f1, weighted_f1

def obtaion_LT_distribution(y_true):
        # 计算每个类别的频率
    class_freq = np.bincount(y_true)

    # 确定 head, medium, tail 的阈值
    num_classes = len(class_freq)
    head_threshold = np.percentile(class_freq, 65)
    medium_threshold = np.percentile(class_freq, 20)
    # 划分类别
    head_classes = np.where(class_freq > head_threshold)[0]
    medium_classes = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail_classes = np.where(class_freq <= medium_threshold)[0]
    return head_classes, medium_classes, tail_classes

def obtaion_LT_multi_label_distribution(y_true):
        # 计算每个类别的频率
    class_freq = np.sum(y_true, axis=0)

    # 确定 head, medium, tail 的阈值
    num_classes = len(class_freq)
    head_threshold = np.percentile(class_freq, 65)
    medium_threshold = np.percentile(class_freq, 20)
    # 划分类别
    head_classes = np.where(class_freq > head_threshold)[0]
    medium_classes = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail_classes = np.where(class_freq <= medium_threshold)[0]
    return head_classes, medium_classes, tail_classes


def LT_eval_F1score(y_true, y_pred):
    # 假设 y_true 是测试集的标签，y_pred 是模型的预测结果
    # y_true = ...
    # y_pred = ...

    # 计算每个类别的频率
    class_freq = np.bincount(y_true)

    # 确定 head, medium, tail 的阈值
    num_classes = len(class_freq)
    head_threshold = np.percentile(class_freq, 65)
    medium_threshold = np.percentile(class_freq, 20)

    # 划分类别
    head_classes = np.where(class_freq > head_threshold)[0]
    medium_classes = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail_classes = np.where(class_freq <= medium_threshold)[0]

    # 计算每个类别的 F1-score
    head_f1 = f1_score(y_true[np.isin(y_true, head_classes)], y_pred[np.isin(y_true, head_classes)], average='macro')
    medium_f1 = f1_score(y_true[np.isin(y_true, medium_classes)], y_pred[np.isin(y_true, medium_classes)], average='macro')
    tail_f1 = f1_score(y_true[np.isin(y_true, tail_classes)], y_pred[np.isin(y_true, tail_classes)], average='macro')

    # head_acc = balanced_accuracy(y_true[np.isin(y_true, head_classes)], y_pred[np.isin(y_true, head_classes)])
    # medium_acc = balanced_accuracy(y_true[np.isin(y_true, medium_classes)], y_pred[np.isin(y_true, medium_classes)])
    # tail_acc = balanced_accuracy(y_true[np.isin(y_true, tail_classes)], y_pred[np.isin(y_true, tail_classes)])


    print(f"Head F1-score: {head_f1}")
    print(f"Medium F1-score: {medium_f1}")
    print(f"Tail F1-score: {tail_f1}")

    # print(f"Head Balance Accuracy: {head_acc}")
    # print(f"Medium Balance Accuracy: {medium_acc}")
    # print(f"Tail Balance Accuracy: {tail_acc}")

def LT_eval_accscore(y_true, y_pred):
    # 假设 y_true 是测试集的标签，y_pred 是模型的预测结果
    # y_true = ...
    # y_pred = ...

    # 计算每个类别的频率
    class_freq = np.bincount(y_true)

    # 确定 head, medium, tail 的阈值
    num_classes = len(class_freq)
    head_threshold = np.percentile(class_freq, 65)
    medium_threshold = np.percentile(class_freq, 20)

    # 划分类别
    head_classes = np.where(class_freq > head_threshold)[0]
    medium_classes = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail_classes = np.where(class_freq <= medium_threshold)[0]

    # 计算每个类别的 F1-score

    head_acc = accuracy_score(y_true[np.isin(y_true, head_classes)], y_pred[np.isin(y_true, head_classes)])
    medium_acc = accuracy_score(y_true[np.isin(y_true, medium_classes)], y_pred[np.isin(y_true, medium_classes)])
    tail_acc = accuracy_score(y_true[np.isin(y_true, tail_classes)], y_pred[np.isin(y_true, tail_classes)])

    print(f"Head Accuracy: {head_acc}")
    print(f"Medium Accuracy: {medium_acc}")
    print(f"Tail Accuracy: {tail_acc}")


def split_list(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i+chunk_size]
        result.append(chunk)
    return result


from torch.utils.data import DataLoader, Dataset
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, image_paths, gloria_model, device):
        self.image_paths = image_paths
        self.gloria_model = gloria_model
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        processed_img = self.gloria_model.process_img(img_path, self.device)
        return processed_img, img_path

def create_dataloader(df, batch_size, gloria_model, device):
    image_dataset = ImageDataset(df['Path'].tolist(), gloria_model, device)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    return dataloader

def sanitize_name(name: str) -> str:
    # 文件名安全处理
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for b in bad:
        name = name.replace(b, '_')
    return name.strip().replace(' ', '_')

def process_images(df, gloria_model, cls_prompts, device, batch_size=64):
    dataloader = create_dataloader(df, batch_size, gloria_model, device)
    processed_txt = gloria_model.process_class_prompts(cls_prompts, device)
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    class_spans = []  # [(cls_name, start, end), ...]
    cap_lens = []
    cursor = 0
    for cls_name, txts in processed_txt.items():
        # ipdb.set_trace()
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])
        cap_lens += txts["cap_lens"]
        class_spans.append((cls_name, cursor, cursor + 1))
    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)

    txts = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids,  "cap_lens": cap_lens}
    # 
    # # 创建数据加载器分批处理
    # dataset = TensorDataset(caption_ids, attention_mask, token_type_ids)
    # data_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_text_emb_l = []
    all_text_emb_g = []

    with torch.no_grad():
        text_emb_l, text_emb_g, _ = gloria_model.text_encoder_forward(
                txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
            )

    
       # #  # 初始化矩阵

    num_images = len(df)  # 假设 DataFrame 每行对应一张图像

    start_idx = 0

    similar = None
    # similar = []
    image_feat = torch.zeros((num_images, 768), dtype=torch.float32).cuda()  # 预分配内存
    for processed_imgs, path in dataloader:
        # zero-shot classification on images
        
        similarities, img_batch = gloria.zero_shot_fast_classification(gloria_model, processed_imgs.to(device), text_emb_g, cls_prompts, path[0])

        if similar is None:
            similar = similarities
        else:
            similar = pd.concat([similar, similarities], axis=0)

    text_emb_g = torch.functional.F.normalize(text_emb_g, p=2, dim=-1)


    return similar, text_emb_g


def process_none_txt_images(df, gloria_model, cls_prompts, device, batch_size=64):
    dataloader = create_dataloader(df, batch_size, gloria_model, device)
    processed_txt = gloria_model.process_class_prompts(cls_prompts, device)
    none_txt = {'0':'no finding in CT.'}
    porcessed_none_txt = gloria_model.process_class_prompts(none_txt, device)
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    cap_lens = []
    for cls_name, txts in processed_txt.items():
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])
        cap_lens += txts["cap_lens"]
    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    txts = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids, "cap_lens": cap_lens}

    with torch.no_grad():
        text_emb_l, text_emb_g, _ = gloria_model.text_encoder_forward(
                txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
            )
        text_emb_l, text_emb_g_none, _ = gloria_model.text_encoder_forward( porcessed_none_txt['0']['caption_ids'], porcessed_none_txt['0']['attention_mask'], porcessed_none_txt['0']['token_type_ids'])

    similar = None
    for processed_imgs in dataloader:
        
        # zero-shot classification on images
        similarities = gloria.zero_shot_fast_classification(gloria_model, processed_imgs.to(device), text_emb_g, text_emb_g_none, cls_prompts)
        
        if similar is None:
            similar = similarities
        else:
            similar = pd.concat([similar, similarities], axis=0)
            # similar = np.concatenate((similar, similarities), axis=0)
        
    text_emb_g = torch.functional.F.normalize(text_emb_g, p=2, dim=-1)
    text_emb_g = text_emb_g.cpu().numpy()
    return similar, text_emb_g



def obtain_sim(image_path, text_path):
    df = pd.read_csv(image_path)
    with open(text_path, 'r') as f:
        cls_prompts = json.load(f)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gloria_model = gloria.load_gloria(name="gloria_resnet50", device=device)

    # state_dict = gloria_model.img_encoder.state_dict()

 
    similar, text_emb_g = process_images(df, gloria_model, cls_prompts, device, batch_size=1)
    # similar, text_emb_g = process_none_txt_images(df, gloria_model, cls_prompts, device, batch_size=8)

    return similar, text_emb_g


def obtain_chexpert_sim():
    df = pd.read_csv(gloria.constants.CHEXPERT_5x200)
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gloria_model = gloria.load_gloria(name="gloria_resnet50", device=device)

    # generate class prompt
    # cls_promts = {
    #    'Atelectasis': ['minimal residual atelectasis ', 'mild atelectasis' ...]
    #    'Cardiomegaly': ['cardiomegaly unchanged', 'cardiac silhouette enlarged' ...] 
    # ...
    # } 
    # cls_prompts = gloria.generate_chexpert_class_prompts()

    with open('/haoranlai/Project/gloria/Dataset/chexpert_5x200.json', 'r') as f:
        cls_prompts = json.load(f)

    # process input images and class prompts 
    ## batchsize
    bs = 100
    image_list = split_list(df['Path'].tolist(), bs)
    processed_txt = gloria_model.process_class_prompts(cls_prompts, device)
    for i, img in enumerate(image_list):
        processed_imgs = gloria_model.process_img(img, device)
        # zero-shot classification on 1000 images
        similarities = gloria.zero_shot_classification(
            gloria_model, processed_imgs, processed_txt)
        if i == 0:
            similar = similarities
        else:
            similar = pd.concat([similar, similarities], axis=0)
    return similar



    # print(similarities)

def softmax(x):
    # 对每个样本的输出进行指数运算
    exp_scores = np.exp(x)
    
    # 按行进行求和，得到每个样本的总和
    sum_scores = np.sum(exp_scores, axis=1, keepdims=True)
    
    # 计算 softmax 概率
    softmax_probs = exp_scores / sum_scores
    
    return softmax_probs


def triple_merge_LTNih_result(predict_csv):
    nih_id_test_label_path = '/data/haoranlai/Project/REFERS/NIHlabel/nih-cxr-lt_single-label_test.csv'
    # nih_id_test_label_path = '/data/haoranlai/Project/REFERS/NIHlabel/nih-cxr-lt_single-label_balanced-test.csv'
    LT_path = '/data/haoranlai/Project/REFERS/NIHlabel/nih-cxr-lt_single-label_test.csv'
    use_balance_data = False 

    predict = pd.read_csv(predict_csv).values
    predict = softmax(predict)


    df_test = pd.read_csv(nih_id_test_label_path)
    key = df_test.keys()[1:-1]
        
    label = df_test[key].values
    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1
    
    haxi_pre = np.where(pre == 1)[1]
    haxi_label = np.where(label == 1)[1]

    df_lt = pd.read_csv(LT_path)
    lt_label = df_lt[key].values
    LT_y_ture = np.where(lt_label == 1)[1]
    head, medium, tail = obtaion_LT_distribution(LT_y_ture)

    # LT_eval_F1score(haxi_label, haxi_pre)   
    micro_f1, macro_f1, weighted_f1 = eval_F1(pre, label)

    if use_balance_data:
        # LT_eval_accscore(haxi_label, haxi_pre)
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, head)], haxi_pre[np.isin(haxi_label, head)])
        print(f"Head balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, medium)], haxi_pre[np.isin(haxi_label, medium)])
        print(f"Medium balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, tail)], haxi_pre[np.isin(haxi_label, tail)])
        print(f"Tail balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label, haxi_pre)
        print(f"Total balance accuracy: {new_bacc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
        print(f"Total AUC: {macro_auc}")
        print(per_auc)
    else:
        LT_eval_F1score(haxi_label, haxi_pre)  
        new_bacc = balanced_accuracy(haxi_label, haxi_pre)
        print(f"Total balance accuracy: {new_bacc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
        print(f"Total AUC: {macro_auc}")
        print(per_auc)


def triple_merge_LTNih_Balance_result(predict_csv):
    nih_id_test_label_path = '/data/haoranlai/Project/REFERS/NIHlabel/nih-cxr-lt_single-label_balanced-test.csv'
    LT_path = '/data/haoranlai/Project/REFERS/NIHlabel/nih-cxr-lt_single-label_test.csv'
    use_balance_data = True 

    predict = pd.read_csv(predict_csv).values
    predict = softmax(predict)

    df_test = pd.read_csv(nih_id_test_label_path)
    key = df_test.keys()[1:-1]
        
    label = df_test[key].values
    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1
    
    haxi_pre = np.where(pre == 1)[1]
    haxi_label = np.where(label == 1)[1]

    df_lt = pd.read_csv(LT_path)
    lt_label = df_lt[key].values
    LT_y_ture = np.where(lt_label == 1)[1]
    head, medium, tail = obtaion_LT_distribution(LT_y_ture)

    # LT_eval_F1score(haxi_label, haxi_pre)   
    micro_f1, macro_f1, weighted_f1 = eval_F1(pre, label)

    if use_balance_data:
        # LT_eval_accscore(haxi_label, haxi_pre)
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, head)], haxi_pre[np.isin(haxi_label, head)])
        print(f"Head balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, medium)], haxi_pre[np.isin(haxi_label, medium)])
        print(f"Medium balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, tail)], haxi_pre[np.isin(haxi_label, tail)])
        print(f"Tail balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label, haxi_pre)
        print(f"Total balance accuracy: {new_bacc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
        print(f"Total AUC: {macro_auc}")
    else:
        LT_eval_F1score(haxi_label, haxi_pre)  
        new_bacc = balanced_accuracy(haxi_label, haxi_pre)
        print(f"Total balance accuracy: {new_bacc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
        print(f"Total AUC: {macro_auc}")
        print(per_auc)

def triple_Chexpert_result(predict_csv):
    use_balance_data = False 

    predict = pd.read_csv(predict_csv).values
    predict = softmax(predict)


    df_test = pd.read_csv(gloria.constants.CHEXPERT_5x200)
    key = gloria.constants.CHEXPERT_COMPETITION_TASKS
        
    label = df_test[key].values
    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1
    
    haxi_pre = np.where(pre == 1)[1]
    haxi_label = np.where(label == 1)[1]

    # df_lt = pd.read_csv(LT_path)
    # lt_label = df_lt[key].values
    # LT_y_ture = np.where(lt_label == 1)[1]
    # head, medium, tail = obtaion_LT_distribution(LT_y_ture)

    # LT_eval_F1score(haxi_label, haxi_pre)   
    micro_f1, macro_f1, weighted_f1 = eval_F1(pre, label)
    # print(f"Total macro F1: {macro_f1}")
    new_bacc = accuracy_score(haxi_label, haxi_pre)
    print(f"Total balance accuracy: {new_bacc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
    print(f"Total AUC: {macro_auc}")
    # print(per_auc)


def tripple_branchmark_rusult_merge(predict_csv):
    # img_feature_path = '/data/haoranlai/Project/REFERS/Pre-train/Pretrain_retrieval/img_feature.npy'
    # text_feature_path = '/data/haoranlai/Project/REFERS/Pre-train/Pretrain_retrieval/text_feature.npy'
    # nih_id_test_label_path = '/data/haoranlai/Project/REFERS/NIHlabel/nih-cxr-lt_single-label_test.csv'
    nih_id_test_label_path = '/data/haoranlai/Project/REFERS/NIHlabel/test_list.txt'
    LT_path = '/data/haoranlai/Project/REFERS/NIHlabel/nih-cxr-lt_single-label_test.csv'
    use_balance_data = False 

    predict = pd.read_csv(predict_csv).values

    prediction_Score = np.zeros((predict.shape[0], predict.shape[1] // 2))
    prediction_hard = np.zeros((predict.shape[0], predict.shape[1] // 2))
    count = 0
    # ipdb.set_trace()
    for cls in range(0, predict.shape[1], 2):
        temp = predict[:, cls:cls+2]
        # ipdb.set_trace()
        prediction_Score[:, count] = temp[:, 0]
        prediction_hard[:, count] = 1 - np.argmax(temp)
        count += 1

    CLASS_NAMES = ['path', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    df_test = pd.read_csv(nih_id_test_label_path, sep=' ', names=CLASS_NAMES)
    key = df_test.keys()[1:] 
    label = df_test[key].values

    # pre = np.zeros((predict.shape[0] , predict.shape[1]))
    # for i in range(predict.shape[0]):
    #     logit = predict[i]
    #     ind = np.argmax(logit)
    #     pre[i, ind] = 1
    
    # haxi_pre = np.where(pre == 1)[1]
    # haxi_label = np.where(label == 1)[1]

    # df_lt = pd.read_csv(LT_path)
    # lt_label = df_lt[key].values
    # LT_y_ture = np.where(lt_label == 1)[1]
    # head, medium, tail = obtaion_LT_distribution(LT_y_ture)

    # LT_eval_F1score(haxi_label, haxi_pre)   
    # micro_f1, macro_f1, weighted_f1 = eval_F1(pre, label)
    head, medium, tail = obtaion_LT_multi_label_distribution(label)

    if use_balance_data:
        # LT_eval_accscore(haxi_label, haxi_pre)
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, head)], haxi_pre[np.isin(haxi_label, head)])
        print(f"Head balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, medium)], haxi_pre[np.isin(haxi_label, medium)])
        print(f"Medium balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, tail)], haxi_pre[np.isin(haxi_label, tail)])
        print(f"Tail balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label, haxi_pre)
        print(f"Total balance accuracy: {new_bacc}")
    else:
        # LT_eval_F1score(haxi_label, haxi_pre)  
        # new_bacc = balanced_accuracy(haxi_label, haxi_pre)
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(prediction_Score[:, head], label[:, head])
        print(f"Head AUC: {macro_auc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(prediction_Score[:, medium], label[:, medium])
        print(f"Medium AUC: {macro_auc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(prediction_Score[:, tail], label[:, tail])
        print(f"Tail AUC: {macro_auc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(prediction_Score, label)
        print(f"Total AUC: {macro_auc}")
        # print(per_auc)
        # for i, k in enumerate(key):
        #      print(f"{k}: {per_auc[i]}")
        # ipdb.set_trace()


def tripple_openi_rusult_merge(predict_csv):
    # img_feature_path = '/data/haoranlai/Project/REFERS/Pre-train/Pretrain_retrieval/img_feature.npy'
    # text_feature_path = '/data/haoranlai/Project/REFERS/Pre-train/Pretrain_retrieval/text_feature.npy'
    # nih_id_test_label_path = '/data/haoranlai/Project/REFERS/NIHlabel/nih-cxr-lt_single-label_test.csv'
    xmlpath="/haoranlai/Dataset/OpenI/NLMCXR_reports"
    csv_path="/haoranlai/Dataset/OpenI/custom.csv"

    # np.random.seed(seed)  # Reset the seed so all runs are the same.
    # self.imgpath = imgpath
    # self.transform = transform
    # self.data_aug = data_aug

    pathologies = [
        # NIH
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        ## "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        # ---------
        "Fracture",
        "Opacity",
        "Lesion",
        # ---------
        "Calcified Granuloma",
        "Granuloma",
        # ---------
        "No_Finding",
    ]
    # self.pathologies = sorted(self.pathologies)
    # self.pathologies.append("No_Finding")

    mapping = dict()
    mapping["Pleural_Thickening"] = ["pleural thickening"]
    mapping["Infiltration"] = ["Infiltrate"]
    mapping["Atelectasis"] = ["Atelectases"]

    # Load data
    csv = pd.read_csv(csv_path)
    csv = csv.replace(np.nan, "-1")

    gt = []
    for pathology in pathologies:
        mask = csv["labels_automatic"].str.contains(pathology.lower())
        if pathology in mapping:
            for syn in mapping[pathology]:
                # print("mapping", syn)
                mask |= csv["labels_automatic"].str.contains(syn.lower())
        gt.append(mask.values)

    gt = np.asarray(gt).T
    gt = gt.astype(np.float32)
    # Rename pathologies
    pathologies = np.char.replace(pathologies, "Opacity", "Lung Opacity")
    pathologies = np.char.replace(pathologies, "Lesion", "Lung Lesion")

    ## Rename by myself
    pathologies = np.char.replace(pathologies, "Pleural_Thickening", "pleural thickening")
    pathologies = np.char.replace(pathologies, "Infiltration", "Infiltrate")
    pathologies = np.char.replace(pathologies, "Atelectasis", "Atelectases")
    gt[np.where(np.sum(gt, axis=1) == 0), -1] = 1
    
    label = gt[:, :-1]


    use_balance_data = False 

    predict = pd.read_csv(predict_csv).values

    # prediction_Score = np.zeros((predict.shape[0], predict.shape[1] // 2))
    # prediction_hard = np.zeros((predict.shape[0], predict.shape[1] // 2))
    # count = 0
 
    # for cls in range(0, predict.shape[1], 2):
    #     temp = predict[:, cls:cls+2]
    #     # ipdb.set_trace()
    #     prediction_Score[:, count] = temp[:, 0]
    #     prediction_hard[:, count] = 1 - np.argmax(temp)
    #     count += 1


    # for cls in range(0, predict.shape[1], 2):
    #     temp = predict[:, cls:cls+2]
    #     # ipdb.set_trace()
    #     # temp = softmax(temp)
    #     prediction_Score[:, count] = temp[:, 0]
    #     prediction_hard[:, count] = 1 - np.argmax(temp)
    #     count += 1


    # prediction_Score = prediction_Score[:, :-1]
    # prediction_hard = prediction_hard[:, :-1]

    head, medium, tail = obtaion_LT_multi_label_distribution(label)

    if use_balance_data:
        # LT_eval_accscore(haxi_label, haxi_pre)
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, head)], haxi_pre[np.isin(haxi_label, head)])
        print(f"Head balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, medium)], haxi_pre[np.isin(haxi_label, medium)])
        print(f"Medium balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label[np.isin(haxi_label, tail)], haxi_pre[np.isin(haxi_label, tail)])
        print(f"Tail balance accuracy: {new_bacc}")
        new_bacc = accuracy_score(haxi_label, haxi_pre)
        print(f"Total balance accuracy: {new_bacc}")
    else:
        # LT_eval_F1score(haxi_label, haxi_pre)  
        # new_bacc = balanced_accuracy(haxi_label, haxi_pre)
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, head], label[:, head])
        print(f"Head AUC: {macro_auc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, medium], label[:, medium])
        print(f"Medium AUC: {macro_auc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, tail], label[:, tail])
        print(f"Tail AUC: {macro_auc}")
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
        print(f"Total AUC: {macro_auc}")
        # for i, k in enumerate(key):
        #      print(f"{k}: {per_auc[i]}")
        # ipdb.set_trace()
    # ipdb.set_trace()


def tripple_padchest_rusult_merge(predict_csv):
    test_query = ['atelectasis', 'cardiomegaly', 'consolidation', 'pulmonary edema', 'pneumonia']
    from sklearn.preprocessing import MultiLabelBinarizer
    with open('/haoranlai/Dataset/PadChest/manual_image.json', "r") as file:
         data = json.load(file) 
    label = []
    key = data.keys()
    for k in key:
        label += data[k]
    unique_label = list(set(label))
    # Sort the unique strings with stable sorting
    sorted_strings = sorted(unique_label, key=lambda x: (x, label.index(x)))
    
    index = sorted_strings.index('normal')

    labels = [ data[k] for k in key ]

    # 创建MultiLabelBinarizer对象
    mlb = MultiLabelBinarizer(classes=sorted_strings)

    # 使用fit_transform()方法进行One-Hot编码
    encoded_labels = mlb.fit_transform(labels)

    predict = pd.read_csv(predict_csv).values

    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1

    encoded_labels =  np.delete(encoded_labels, index, axis=1)
    # 删除normal
    sorted_strings.remove('normal')

    # ipdb.set_trace()s

    ## 查找test_query的index
    test_query_index = []
    for i in test_query:
        test_query_index.append(sorted_strings.index(i))
    

    use_balance_data = False

    head, medium, tail = obtaion_LT_multi_label_distribution(encoded_labels)
    
    # LT_eval_F1score(haxi_label, haxi_pre)  
    # new_bacc = balanced_accuracy(haxi_label, haxi_pre)
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, head], encoded_labels[:, head])
    print(f"Head AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, medium], encoded_labels[:, medium])
    print(f"Medium AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, tail], encoded_labels[:, tail])
    print(f"Tail AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, encoded_labels)
    print(f"Total AUC: {macro_auc}")

    # 打印test_query的AUC
    for i in test_query_index:
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, i], encoded_labels[:, i])
        print(f"{sorted_strings[i]} AUC: {macro_auc}")
            
    


def triple_Chexpert14_result(predict_csv):
    path ="/haoranlai/Dataset/ChestX-ray14/test_list.txt"

    csv_head = ['path', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Lung Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

    df_test = pd.read_csv(path, sep=' ', names=csv_head)
    # images_path = df['path'].tolist()
    
    # df_test = pd.read_csv(gloria.constants.CHEXPERT_5x200)
    # key = gloria.constants.CHEXPERT_COMPETITION_TASKS
    key = csv_head[1:]

    predict = pd.read_csv(predict_csv).values
        
    label = df_test[key].values
    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1
    
    haxi_pre = np.where(pre == 1)[1]
    haxi_label = np.where(label == 1)[1]

    # df_lt = pd.read_csv(LT_path)
    # lt_label = df_lt[key].values
    # LT_y_ture = np.where(lt_label == 1)[1]
    # head, medium, tail = obtaion_LT_distribution(LT_y_ture)

    # LT_eval_F1score(haxi_label, haxi_pre)   
    micro_f1, macro_f1, weighted_f1 = eval_F1(pre, label)
    # print(f"Total macro F1: {macro_f1}")
    # new_bacc = accuracy_score(haxi_label, haxi_pre)
    # print(f"Total balance accuracy: {new_bacc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
    print(f"Total AUC: {macro_auc}")
    # print(per_auc)

def calculate_youden_thresholds(predict, label):
    n_classes = label.shape[1]
    best_thresholds = np.zeros(n_classes)
    auc_scores = np.zeros(n_classes)

    for i in range(n_classes):
        # 计算每个类别的 ROC 曲线
        fpr, tpr, thresholds = roc_curve(label[:, i], predict[:, i])
        auc_scores[i] = auc(fpr, tpr)
        
        # 计算 Youden's Index
        youden_index = tpr - fpr
        best_threshold_idx = np.argmax(youden_index)
        
        # 对应的最优阈值
        best_thresholds[i] = thresholds[best_threshold_idx]

    return best_thresholds, auc_scores


def calculate_optimal_f1_thresholds(predict, label):
    n_classes = label.shape[1]
    best_thresholds = np.zeros(n_classes)
    best_f1_scores = np.zeros(n_classes)

    for i in range(n_classes):
        thresholds = np.linspace(-1, 1, 201)  # 生成从0到1的101个候选阈值
        best_f1 = 0
        best_t = 0  # 初始化阈值为0.5

        for t in thresholds:
            # 对每个阈值，计算当前阈值下的预测结果
            temp_pre = predict[:, i] > t
            # 计算F1 score
            f1 = f1_score(label[:, i], temp_pre)

            # 找到最优F1 score及对应的阈值
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[i] = best_t
        best_f1_scores[i] = best_f1

    return best_thresholds, best_f1_scores

def calculate_classwise_accuracy(predict, label):
    n_classes = label.shape[1]
    accuracies = np.zeros(n_classes)

    for i in range(n_classes):
        # 逐类别计算每个类别的准确率
        accuracies[i] = accuracy_score(label[:, i], predict[:, i])

    # 返回每个类别的准确率和平均准确率
    return np.mean(accuracies)


# def save_npz(predict, label, path):
#     np.savez(os.path.join(path, 'predicted_weights.npz'), data=predict)
#     np.savez(os.path.join(path, 'labels_weights.npz'), data=label)


def save_npz(predict, label, dataset='CT-RATE'):
    os.makedirs(f'./bootstrap_evaluation/{dataset}/', exist_ok=True)
    np.savez( f'./bootstrap_evaluation/{dataset}/predicted_weights.npz', data=predict)
    np.savez(f'./bootstrap_evaluation/{dataset}/labels_weights.npz', data=label)
    

def triple_CT_RATE_result(predict_csv):
    path = "./Dataset/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
    df = pd.read_csv(path)
    predict = pd.read_csv(predict_csv).values
    key = df.keys().tolist()[1:]

    label = df[key].values

    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)

    for i, k in enumerate(key):
        print(f"{k}: {per_auc[i]}")

    print(f"Macro AUC: {macro_auc}")
    return predict, label


import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, log_loss

def _safe_auc(y_true, y_prob):
    # 若该类在标签中没有正/负样本，AUC 不定义；返回 NaN
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan

def _safe_ap(y_true, y_prob):
    try:
        return average_precision_score(y_true, y_prob)
    except Exception:
        return np.nan

def _best_threshold_for_f1(y_true, y_prob, grid=None):
    # 在给定网格上寻找 F1 最高的阈值
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def load_predictions_aligned(predict_csv, categories):
    pdf = pd.read_csv(predict_csv)
    if set(categories).issubset(set(pdf.columns)):
        y_prob = pdf[categories].values.astype(float)
    else:
        y_prob = pdf.values.astype(float)
        assert y_prob.shape[1] == len(categories), \
            f"预测列数 {y_prob.shape[1]} != 类别数 {len(categories)}；需要提供列名或显式映射！"
    return y_prob

def analyze_worst_samples_and_categories(
    predict_csv,
    label_csv,
    topk=20
):
    df = pd.read_csv(label_csv)
    sample_col = df.columns[0]
    categories = df.columns[1:].tolist()
    y_true = df[categories].values.astype(int)
    sample_names = df[sample_col].astype(str).tolist()

    y_prob = load_predictions_aligned(predict_csv, categories)
    assert y_prob.shape == y_true.shape, f"predict {y_prob.shape} vs label {y_true.shape}"

    C = y_true.shape[1]
    per_auc, per_ap, per_best_t, per_best_f1 = [], [], [], []
    for j in range(C):
        a = _safe_auc(y_true[:,j], y_prob[:,j]); per_auc.append(a)
        p = _safe_ap(y_true[:,j], y_prob[:,j]);  per_ap.append(p)
        t,f = _best_threshold_for_f1(y_true[:,j], y_prob[:,j])
        per_best_t.append(t); per_best_f1.append(f)

    y_pred = np.zeros_like(y_prob, dtype=int)
    for j in range(C): y_pred[:,j] = (y_prob[:,j] >= per_best_t[j]).astype(int)

    # 样本级损失/错误
    eps=1e-7; prob=np.clip(y_prob,eps,1-eps)
    bce = -(y_true*np.log(prob)+(1-y_true)*np.log(1-prob))
    mean_bce = bce.mean(axis=1)
    mistakes  = (y_true!=y_pred).sum(axis=1)
    fn_count  = ((y_true==1)&(y_pred==0)).sum(axis=1)
    fp_count  = ((y_true==0)&(y_pred==1)).sum(axis=1)

    df_sample = pd.DataFrame({
        "Sample": sample_names, "MeanBCE": mean_bce, "Mistakes": mistakes, "FN": fn_count, "FP": fp_count
    }).sort_values(["MeanBCE","Mistakes","FN","FP"], ascending=[False,False,False,False]).reset_index(drop=True)

    df_class = pd.DataFrame({
        "Category": categories, "AUC": per_auc, "AP": per_ap, "BestThreshold": per_best_t, "F1_at_BestT": per_best_f1
    }).sort_values(["AUC","AP","F1_at_BestT"], ascending=[True,True,True]).reset_index(drop=True)

    return df_class, df_sample, y_pred, y_true, y_prob, categories, sample_names

# ===== 你原函数的入口可以这样用 =====
def triple_CT_RATE_grounding_result(predict_csv):
    df_class, df_sample, y_pred, y_true, y_prob, categories = analyze_worst_samples_and_categories(
        predict_csv=predict_csv,
        label_csv="/data4/haoranlai/Dataset/CT-RATE/ReXGroundingCT/sample_labels_18cls.csv",
        topk=20,
        out_class_csv="/data4/haoranlai/Dataset/CT-RATE/ReXGroundingCT/report_categories.csv",
        out_sample_csv="/data4/haoranlai/Dataset/CT-RATE/ReXGroundingCT/report_samples.csv"
    )
    # 想继续做别的（比如返回 per-class 阈值），可以从 df_class['BestThreshold'] 里取
    return df_class, df_sample



def get_fail_cases(y_true, y_pred, categories, sample_names, topk=10):
    mistakes = np.sum(y_true!=y_pred, axis=1)
    idx_sorted = np.argsort(-mistakes)[:topk]
    cases=[]
    for idx in idx_sorted:
        true=y_true[idx]; pred=y_pred[idx]
        fn=[categories[j] for j in range(len(categories)) if true[j]==1 and pred[j]==0]
        fp=[categories[j] for j in range(len(categories)) if true[j]==0 and pred[j]==1]
        cases.append({"Sample": sample_names[idx], "Mistakes": int(mistakes[idx]),
                      "FN_classes": fn, "FP_classes": fp})
    return cases




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

def metrics(new_labels,predict):
        # Thresholds list
    thresholds = []
    f1s = []
    accs = []
    precisions = []
    auprcs = []
    recalls = []
    numberlabel = new_labels.shape[1]

    for i in range(numberlabel):
        logit = predict[:, i]
        l = new_labels[:, i]
        prob = logit
        threshold = find_threshold(prob, l)
        thresholds.append(threshold)

        pred = (prob > threshold).astype(int)
        label = l

        f1 = f1_score(label, pred, average="weighted")
        acc = accuracy_score(label, pred)
        precision = precision_score(label, pred)
        recall = recall_score(label, pred)
        auprc = average_precision_score(label, prob)

        f1s.append(f1)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        auprcs.append(auprc)

    macro_auc = roc_auc_score(new_labels, predict , average="macro")

    print("Macro AUC: ", macro_auc)
    print("Total f1s: ", np.mean(f1s))
    print("Total accs: ", np.mean(accs))
    print("Total precisions: ", np.mean(precisions))
    print("Total recalls: ", np.mean(recalls))
    print("Total auprcs: ", np.mean(auprcs))
    return auprcs


def metrics_balance(new_labels,predict):
        # Thresholds list
    thresholds = []
    f1s = []
    accs = []
    precisions = []
    auprcs = []
    recalls = []
    numberlabel = new_labels.shape[1]

    for i in range(numberlabel):
        logit = predict[:, i]
        l = new_labels[:, i]
        prob = logit
        threshold = find_ours_threshold(prob, l, method="pr")
        thresholds.append(threshold)

        pred = (prob > threshold).astype(int)
        label = l

        f1 = f1_score(label, pred, average="weighted")
        acc = accuracy_score(label, pred)
        precision = precision_score(label, pred)
        recall = recall_score(label, pred)
        auprc = average_precision_score(label, prob)

        f1s.append(f1)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        auprcs.append(auprc)

    # macro_auc = roc_auc_score(new_labels, predict , average="macro")
    macro_auc = roc_auc_score(new_labels, predict , average=None)

    print("Macro AUC: ", macro_auc)
    print("Total f1s: ", np.mean(f1s))
    print("Total accs: ", np.mean(accs))
    print("Total precisions: ", np.mean(precisions))
    print("Total recalls: ", np.mean(recalls))
    print("Total auprcs: ", np.mean(auprcs))
    return macro_auc, f1s, accs, precisions, recalls, auprcs


import pandas as pd
import numpy as np

def triple_CT_RATE_GPT4_result(predict_csv):
    # 固定的15类顺序（与 predict 的顺序一致）
    disease_order = [
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

    # 只需要标签文件（gt）和预测文件（predict）
    modify_path = "./Dataset/CT-RATE-LT-label.csv"

    # 读取
    df_new = pd.read_csv(modify_path)
    pred_df = pd.read_csv(predict_csv)

    # 如果第一列是ID之类的，去掉；保留病种列
    def strip_id(df):
        first_col = df.columns[0].lower()
        if first_col in ["id", "study_id", "seriesuid", "name", "index"]:
            return df.drop(columns=[df.columns[0]])
        return df

    df_new = strip_id(df_new)
    pred_df = strip_id(pred_df)

    # 标签侧：严格按 disease_order 重排
    missing_in_gt = [d for d in disease_order if d not in df_new.columns]
    if missing_in_gt:
        raise ValueError(f"GT缺少列: {missing_in_gt}")

    gt_df = df_new[disease_order].copy()

    # 预测侧：我们假设顺序已与 disease_order 一致；
    # 若预测csv包含表头且列名正好是病种名，也可强制对齐：
    if set(disease_order).issubset(set(pred_df.columns)):
        pred_df = pred_df[disease_order].copy()
    else:
        # 否则认为 pred_df 只有分数、不带这些列名，就直接按顺序取数值
        # 并断言列数=15
        if pred_df.shape[1] != len(disease_order):
            raise ValueError(f"预测列数({pred_df.shape[1]})与15类不一致。请确认 predict_csv 列顺序与 disease_order 匹配。")
        # 保留原顺序
        pred_df = pred_df

    # 转为矩阵；标签转成0/1
    y_true = gt_df.fillna(0).astype(int).values
    y_pred = pred_df.values

    # 形状检查（行数必须一致）
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"样本数不一致：gt={y_true.shape[0]}, pred={y_pred.shape[0]}")

    # 计算AUC
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(y_pred, y_true)

    # 打印/返回结果
    print("Evaluation order:", disease_order)
    print(f"Macro AUC: {macro_auc}")
    print(f"Micro AUC: {micro_auc}")
    print(f"Weighted AUC: {weighted_auc}")
    print("Per-class AUC:")
    for k, v in zip(disease_order, per_auc):
        print(f"  {k}: {v:.4f}")

    return y_pred, y_true




def plot_disease_frequency(predict_csv):
    # Paths
    path_old = "/data/haoranlai/Dataset/CT-RATE/Info/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
    modify_path = "/data/haoranlai/Project/gloria/ctrate_gpt4_label/CT-RATE_disease_data_valid_modify.csv"

    # Load data
    df_old = pd.read_csv(path_old)
    df_new = pd.read_csv(modify_path)
    predict = pd.read_csv(predict_csv).values
    predict = np.delete(predict, 24, axis=1)

    old_keys = df_old.keys().tolist()[1:]
    new_keys = df_new.keys().tolist()[1:]

    # Merge pneumonia types
    if "Covid 19 pneumonia" in df_new and "Viral pneumonia" in df_new:
        df_new["Pneumonia"] = (
            df_new["Pneumonia"].fillna(0).astype(int) |
            df_new["Covid 19 pneumonia"].fillna(0).astype(int) |
            df_new["Viral pneumonia"].fillna(0).astype(int)
        ).astype(int)
        covid_index = new_keys.index("Covid 19 pneumonia")
        viral_index = new_keys.index("Viral pneumonia")
        predict = np.delete(predict, [covid_index, viral_index], axis=1)
        df_new = df_new.drop(columns=["Covid 19 pneumonia", "Viral pneumonia"])

    if "Renal parapelvic cyst" in df_new:
        df_new["Renal cyst"] = (
            df_new["Renal cyst"].fillna(0).astype(int) |
            df_new["Renal parapelvic cyst"].fillna(0).astype(int)
        ).astype(int)
        renal_index = new_keys.index("Renal parapelvic cyst")
        predict = np.delete(predict, renal_index, axis=1)
        df_new = df_new.drop(columns=["Renal parapelvic cyst"])

    new_keys = df_new.keys().tolist()[1:]

    disease_list = [
        "Pneumonia", "Pneumothorax", "Empyema", "Tuberculosis", "Pulmonary fibrosis",
        "Lung cyst", "Bronchial thickening", "Lung mass", "Osteophyte", "Rib fracture",
        "Atherosclerosis", "Pericardial calcification", "Spondylosis", "Pleural calcification",
        "Pneumomediastinum"
    ]

    common_keys = set(old_keys).intersection(set(new_keys))
    remaining_keys = [key for key in new_keys if key not in common_keys]
    remaining_indices = [new_keys.index(key) for key in remaining_keys]
    remaining_labels = df_new[remaining_keys].values
    remaining_predictions = predict[:, remaining_indices]

    valid_keys = [key for key in new_keys if key in disease_list]
    valid_indices = [remaining_keys.index(key) for key in valid_keys]

    filtered_labels = remaining_labels[:, valid_indices]
    filtered_predictions = remaining_predictions[:, valid_indices]

    per_auc = roc_auc_score(filtered_labels, filtered_predictions, average=None)

    frequencies = {key: df_new[key].sum() for key in valid_keys}
    sorted_frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))

    diseases = list(sorted_frequencies.keys())
    counts = list(sorted_frequencies.values())

    # 设置字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['font.size'] = 12

    # 创建图表（竖直长，适合论文单栏）
    fig, ax = plt.subplots(figsize=(5, 6))  # 宽度小、高度适中

    # 压缩频次，仅将 count > 20 压为 26
    compressed_counts = []
    for count in counts:
        if count > 20:
            compressed_counts.append(26)
        else:
            compressed_counts.append(count)

    # 绘制横向柱状图
    y_pos = np.arange(len(diseases))
    bars = ax.barh(y_pos, compressed_counts, color='skyblue')

    # Y轴：疾病名
    ax.set_yticks(y_pos)
    ax.set_yticklabels(diseases)
    ax.invert_yaxis()  # 频率高的在上方
    ax.set_xlabel('Frequency')

    # 自定义 X 轴刻度（包含压缩提示）
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels(['0', '5', '10', '15', '20', '...', '120'])

    # 标注每个条形的真实频率
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_width() + 0.8,  # 右侧偏移
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            va='center',
            fontsize=10,
        )

    # 去除多余边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    plt.tight_layout()
    plt.savefig('disease_frequency_histogram_vertical.pdf', format="pdf", dpi=300)
    # plt.savefig('disease_frequency_histogram_vertical.png', dpi=300)

def plot_disease_frequency_with_auc(predict_csv):
    # Paths
    path_old = "/data/haoranlai/Dataset/CT-RATE/Info/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
    modify_path = "/data/haoranlai/Project/gloria/ctrate_gpt4_label/CT-RATE_disease_data_valid_modify.csv"

    # Load data
    df_old = pd.read_csv(path_old)
    df_new = pd.read_csv(modify_path)
    predict = pd.read_csv(predict_csv).values

    # Delete 24th column in predict
    predict = np.delete(predict, 24, axis=1)

    # Extract keys and labels
    old_keys = df_old.keys().tolist()[1:]
    new_keys = df_new.keys().tolist()[1:]

    # Merge "Covid 19 pneumonia" and "Viral pneumonia" into "Pneumonia"
    if "Covid 19 pneumonia" in df_new and "Viral pneumonia" in df_new:
        df_new["Pneumonia"] = (
            df_new["Pneumonia"].fillna(0).astype(int) |
            df_new["Covid 19 pneumonia"].fillna(0).astype(int) |
            df_new["Viral pneumonia"].fillna(0).astype(int)
        ).astype(int)
        covid_index = new_keys.index("Covid 19 pneumonia")
        viral_index = new_keys.index("Viral pneumonia")
        predict = np.delete(predict, [covid_index, viral_index], axis=1)
        df_new = df_new.drop(columns=["Covid 19 pneumonia", "Viral pneumonia"])

    if "Renal parapelvic cyst" in df_new:
        df_new["Renal cyst"] = (
            df_new["Renal cyst"].fillna(0).astype(int) |
            df_new["Renal parapelvic cyst"].fillna(0).astype(int)
        ).astype(int)
        renal_index = new_keys.index("Renal parapelvic cyst")
        predict = np.delete(predict, renal_index, axis=1)
        df_new = df_new.drop(columns=["Renal parapelvic cyst"])

    # Update keys after merging
    new_keys = df_new.keys().tolist()[1:]

    # Define disease list
    disease_list = [
        "Pneumonia",
        "Pneumothorax",
        "Empyema",
        "Tuberculosis",
        "Acute respiratory distress syndrome",
        "Pulmonary hypertension",
        "Pulmonary fibrosis",
        "Lung cyst",
        "Bronchial thickening",
        "Lung mass",
        "Bronchiolitis",
        "Bronchopneumonia",
        "Tracheal stenosis",
        "Osteophyte",
        "Rib fracture",
        "Atherosclerosis",
        "Heart failure",
        "Infarction",
        "Pericardial calcification",
        "Spondylosis",
        "Pleural thickening",
        "Pleural calcification",
        "Pneumomediastinum"
    ]

    # Find common keys and remove them
    common_keys = set(old_keys).intersection(set(new_keys))
    remaining_keys = [key for key in new_keys if key not in common_keys]

    # Update labels and predictions to exclude common keys
    remaining_indices = [new_keys.index(key) for key in remaining_keys]
    remaining_labels = df_new[remaining_keys].values
    remaining_predictions = predict[:, remaining_indices]

    valid_keys = [key for key in new_keys if key in disease_list]

    valid_indices = [remaining_keys.index(key) for key in valid_keys]

    # Update labels and predictions to exclude low-precision diseases
    filtered_labels = remaining_labels[:, valid_indices]
    filtered_predictions = remaining_predictions[:, valid_indices]

    # Calculate AUC values
    per_auc = roc_auc_score(filtered_labels, filtered_predictions, average=None)

    # Count frequency for each disease
    frequencies = {key: df_new[key].sum() for key in valid_keys}

    # Sort frequencies and AUCs by disease order
    sorted_frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))
    sorted_auc = [per_auc[valid_keys.index(disease)] for disease in sorted_frequencies.keys()]

    # Separate keys and values
    diseases = list(sorted_frequencies.keys())
    counts = list(sorted_frequencies.values())

    # Plot bar chart (Frequency) with line chart (AUC)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['font.size'] = 14

    fig, ax1 = plt.subplots(figsize=(12, 8))

        # Compress bar heights for the range 20~114
    compressed_counts = []
    for count in counts:
        if count > 20:
            compressed_counts.append(26)
        else:
            compressed_counts.append(count)

    # Plot the bars
    bars = ax1.bar(diseases, compressed_counts, color='skyblue', label='Frequency')


    # Bar plot for frequencies
    # bars = ax1.bar(diseases, counts, color='skyblue', label='Frequency')
    # ax1.set_xlabel("Diseases", fontsize=14)
    ax1.set_ylabel("Frequency", fontsize=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(range(len(diseases)))
    ax1.set_xticklabels(diseases, rotation=45, ha='right')

    # Custom y-axis for frequency
    ax1.set_ylim(0, 30)
    ax1.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax1.set_yticklabels(['0', '5', '10', '15', '20', '...', '120'])

    # Line plot for AUCs
    ax2 = ax1.twinx()
    ax2.plot(diseases, sorted_auc, color='orange', marker='o', label='AUC')
    ax2.set_ylabel("AUC", fontsize=14, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

        # Add counts above each bar
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{count}",
            ha='center',
            va='bottom',
        )


    # Add legends
    # fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=12)

    # Adjust layout
    # plt.title("Disease Frequency and AUC", fontsize=16)
    plt.tight_layout()
    plt.savefig('disease_frequency_auc.png', dpi=300)



def ct_clip_result():
  # Paths to the data
    predict_path = '/data/haoranlai/Project/CT-CLIP/performance/inference_zeroshot-Gpt4Modify-ct-rate-sort/predicted_weights.npz'
    labels_path = "/data/haoranlai/Project/CT-CLIP/performance/inference_zeroshot-Gpt4Modify-ct-rate-sort/labels_weights.npz"

    # Load predicted and labels data
    predicted_data = np.load(predict_path)
    labels_data = np.load(labels_path)

    # Extracting the arrays from the loaded files
    labels = labels_data['data']
    predicted = predicted_data['data']

    # Compute the frequency of each disease based on label matrix
    disease_frequencies = labels.sum(axis=0)  # Sum across rows to get frequency per column (disease)

    # Group diseases based on frequency: head, medium, tail
    head_indices = np.where(disease_frequencies >= 100)[0]  # Diseases with >= 100 annotations
    medium_indices = np.where((disease_frequencies >= 10) & (disease_frequencies < 100))[0]  # 10 <= annotations < 100
    tail_indices = np.where(disease_frequencies < 10)[0]  # Diseases with < 10 annotations

    # Subset predictions and labels for head, medium, and tail
    head_predictions = predicted[:, head_indices]
    medium_predictions = predicted[:, medium_indices]
    tail_predictions = predicted[:, tail_indices]

    head_labels = labels[:, head_indices]
    medium_labels = labels[:, medium_indices]
    tail_labels = labels[:, tail_indices]

    metrics(labels, predicted)
    metrics(head_labels, head_predictions)
    metrics(medium_labels, medium_predictions)
    metrics(tail_labels, tail_predictions)

def triple_Rad_ChestCT_new_result(predict_csv):
    # Mapping dictionary: RAD-ChestCT labels to CT-RATE labels
    label_mapping = {
        'Path': 'NoteAcc_DEID',
        'Medical material': ['catheter_or_port', 'tracheal_tube', 'chest_tube', 'breast_implant', 'pacemaker_or_defib', 'stent', 'clip', 'staple', 'gi_tube', 'hardware', 'suture'],  # 医疗器械
        'Arterial wall calcification': ['calcification'],  # 动脉壁钙化
        'Cardiomegaly': ['cardiomegaly'],  # 心脏肥大
        'Pericardial effusion': ['pericardial_effusion'],  # 心包积液
        'Coronary artery wall calcification': ['calcification'],  # 冠状动脉壁钙化
        'Hiatal hernia': ['hernia'],  # 食管裂孔疝
        'Lymphadenopathy': ['lymphadenopathy'],  # 淋巴结肿大
        'Emphysema': ['emphysema'],  # 肺气肿
        'Atelectasis': ['atelectasis'],  # 肺不张
        'Lung nodule': ['nodule', 'nodulegr1cm'],  # 肺结节
        'Lung opacity': ['opacity'],  # 肺部模糊
        'Pulmonary fibrotic sequela': ['scarring'],  # 肺纤维化后遗症
        'Pleural effusion': ['pleural_effusion'],  # 胸腔积液
        'Mosaic attenuation pattern': None,  # 没有对应标签
        'Peribronchial thickening': ['bronchial_wall_thickening'],  # 支气管壁增厚
        'Consolidation': ['consolidation'],  # 实变
        'Bronchiectasis': ['bronchiectasis'],  # 支气管扩张
        'Interlobular septal thickening': ['septal_thickening']  # 小叶间隔增厚
    }

    # Load the CSV files
    rad_chestct_df = pd.read_csv("./Dataset/merged_abnormality_labels_with_id.csv")  # Replace with the actual file path
    # Exclude Mosaic attenuation and apply the mapping
    def map_labels(row):
        mapped_row = {}
        for ct_label, rad_labels in label_mapping.items():
            if rad_labels is None:
                continue
            if isinstance(rad_labels, list):
                mapped_row[ct_label] = max([row.get(label, 0) for label in rad_labels])  # Use max value if multiple labels
            else:
                mapped_row[ct_label] = row.get(rad_labels, 0)
        return mapped_row

    # Apply the mapping function to each row in the RAD-ChestCT dataframe
    mapped_labels = rad_chestct_df.apply(map_labels, axis=1)

    # Create a new dataframe with the mapped labels
    mapped_df = pd.DataFrame(mapped_labels.tolist())

    # Map existing files
    exits_files = pd.read_csv('./Dataset/Rad_ChestCT_all_image.csv')
    exits_files = [path_name.split('/')[-1].split('.')[0] for path_name in exits_files['Path'].tolist()]
    mapped_df = mapped_df[mapped_df['Path'].isin(exits_files)]

    label = mapped_df.iloc[:, 1:].values
    # Remove Coronary artery wall calcification
    label = np.delete(label, 4, axis=1)

    # Load the CT-RATE labels
    predict = pd.read_csv(predict_csv).values
    predict = np.delete(predict, 13, axis=1)

    # Combine columns for 'nodule' and 'nodulegr1cm'
    max_col = np.maximum(predict[:, 1], predict[:, 4])
    predict[:, 1] = max_col
    predict = np.delete(predict, 4, axis=1)

    # Calculate evaluation metrics
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)


    print(f"Macro AUC: {macro_auc}")

    return predict, label




def triple_INSPECT_result(predict_csv, test_csv="/data4/haoranlai/Dataset/INSPECT/test_list.csv"):
    # 1) 读取测试列表，并取最后一列作为标签
    df = pd.read_csv(test_csv)
    label_col_name = df.columns[-1]              # 最后一列
    y = df[label_col_name].to_numpy()

    # 保证是二维 (N,1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # 2) 读取预测
    pred_df = pd.read_csv(predict_csv)

    # 常见几种情况都兼容：
    # - 只有一列分数
    # - 多列（比如前面是ID/文件名，最后一列是分数）
    if pred_df.shape[1] == 1:
        pred = pred_df.iloc[:, 0].to_numpy()
    else:
        # 默认取最后一列作为该单一类别的预测分数
        pred = pred_df.iloc[:, -1].to_numpy()

    # 保证是二维 (N,1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    # 3) 基本一致性校验
    if pred.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: predictions={pred.shape[0]} vs labels={y.shape[0]}")

    # 4) 计算 AUC（假设 eval_auc 支持二分类单列输入）
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(pred, y)

    # save_npz(pred, y)
    print(f"Macro AUC: {macro_auc}")

    return pred, y


def triple_Rad_ChestCT_result(predict_csv):
    path ="./Dataset/merged_abnormality_labels_with_id.csv"

    # Load the CSV files
    rad_chestct_df = pd.read_csv("./Dataset/merged_abnormality_labels_with_id.csv")  # Replace with the actual file path
 
    # Mapping dictionary: RAD-ChestCT labels to CT-RATE labels (in your original order)
    label_mapping = {
        'Path' : 'NoteAcc_DEID',
        'Medical material': 'catheter_or_port',  # 医疗器械
        'Arterial wall calcification': 'calcification',  # 动脉壁钙化
        'Cardiomegaly': 'cardiomegaly',  # 心脏肥大
        'Pericardial effusion': 'pericardial_effusion',  # 心包积液
        'Coronary artery wall calcification': 'calcification',  # 冠状动脉壁钙化
        'Hiatal hernia': 'hernia',  # 食管裂孔疝 1
        'Lymphadenopathy': 'lymphadenopathy',  # 淋巴结肿大
        'Emphysema': 'emphysema',  # 肺气肿
        'Atelectasis': 'atelectasis',  # 肺不张
        'Lung nodule': 'nodule',  # 肺结节         2
        'Lung opacity': 'opacity',  # 肺部模糊     3
        'Pulmonary fibrotic sequela': 'scarring',  # 肺纤维化后遗症（瘢痕）  4
        'Pleural effusion': 'pleural_effusion',  # 胸腔积液
        'Mosaic attenuation pattern': None,  # 没有对应标签             5
        'Peribronchial thickening': 'bronchial_wall_thickening',  # 支气管壁增厚
        'Consolidation': 'consolidation',  # 实变
        'Bronchiectasis': 'bronchiectasis',  # 支气管扩张
        'Interlobular septal thickening': 'septal_thickening'  # 小叶间隔增厚
    }
    # Exclude Mosaic attenuation and apply the mapping
    def map_labels(row):
        mapped_row = {}
        for ct_label, rad_label  in label_mapping.items():
            if rad_label in row:
                mapped_row[ct_label] = row[rad_label] if rad_label in row else 0
        # Exclude Mosaic attenuation and add calcification handling
        return mapped_row

    # Apply the mapping function to each row in the RAD-ChestCT dataframe
    mapped_labels = rad_chestct_df.apply(map_labels, axis=1)

  
    # Create a new dataframe with the mapped labels
    mapped_df = pd.DataFrame(mapped_labels.tolist())

    # map exits files
    exits_files = pd.read_csv('./Dataset/Rad_ChestCT_all_image.csv')
    exits_files = [ path_name.split('/')[-1].split('.')[0] for path_name in exits_files['Path'].tolist()]
    mapped_df = mapped_df[mapped_df['Path'].isin(exits_files)]

    label = mapped_df.iloc[:, 1:].values
    # remove Coronary artery wall calcification
    label = np.delete(label, 4, axis=1)

    # Load the CT-RATE labels
    predict = pd.read_csv(predict_csv).values
    # neg_predict = pd.read_csv(neg_predict_csv).values
    predict = np.delete(predict, 13, axis=1)
    # neg_predict = np.delete(neg_predict, 13, axis=1)

    # 取第 2 列和第 4 列的元素，求最大值
    max_col = np.maximum(predict[:, 1], predict[:, 4])
    # min_col = np.minimum(predict[:, 1], predict[:, 4])

    # 将 max_col 放入第 2 列（即第 3 索引处）
    predict[:, 1] = max_col
    # 将 min_col 放入第 4 列（即第 5 索引处）
    # neg_predict[:, 1] = min_col

    # 删除原来的第 4 列（索引 3）
    predict = np.delete(predict, 4, axis=1)
    # neg_predict = np.delete(neg_predict, 4, axis=1)


    # # 找到每个类别的最优阈值和 AUC
    # best_thresholds, auc_scores = calculate_youden_thresholds(predict, label)

    # # 找到每个类别的最佳F1阈值
    # best_thresholds, best_f1_scores = calculate_optimal_f1_thresholds(predict, label)
    
    # # 应用每个类别的最佳阈值进行预测
    # pre = np.zeros_like(predict)
    # for i in range(len(best_thresholds)):
    #     pre[:, i] = predict[:, i] > best_thresholds[i]

    # pre = predict > 0

    # pre = predict > neg_predict

    # # Calculate F1 scores
    # micro_f1, macro_f1, weighted_f1 = eval_F1(pre, label)

    # # Calculate Accuracy
    # accuracy = calculate_classwise_accuracy(label, pre)

    # # Calculate Precision (macro and micro)
    # macro_precision = precision_score(label, pre, average='macro')
    
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)

    # save_npz(predict, label, './')

    # print(f"Best thresholds: {best_thresholds}")
    print(f"Macro AUC: {macro_auc}")
    # print(f"Total F1 score: {macro_f1}")
    # print(f"Accuracy: {accuracy}")
    # print(f"Macro Precision: {macro_precision}")


def triple_Rad_ChestCT_rest_result(predict_csv):

    # Load the CSV files
    rad_chestct_df = pd.read_csv("./Dataset/merged_abnormality_labels_with_id.csv")  # Replace with the actual file path

    # Mapping dictionary: RAD-ChestCT labels to CT-RATE labels (in your original order)
    label_mapping ={'Path' : 'NoteAcc_DEID', 'Intravenous Catheter or Port': 'catheter_or_port', 'Pleural thickening': 'pleural_thickening', 'Coronary artery bypass graft': 'cabg', 'Infection': 'infection', 'Tree-in-bud pattern': 'tree_in_bud', 'Lung resection': 'lung_resection', 'Debris': 'debris', 'Air trapping': 'air_trapping', 'Cavitation': 'cavitation', 'Scattered calcifications': 'scattered_calc', 'Hemothorax': 'hemothorax', 'Heart valve replacement': 'heart_valve_replacement', 'Dilation or ectasia': 'dilation_or_ectasia', 'Sternotomy': 'sternotomy', 'Lesion': 'lesion', 'Deformity': 'deformity', 'Fibrosis': 'fibrosis', 'Bronchiolitis': 'bronchiolitis', 'Ground-glass opacity': 'groundglass', 'Tuberculosis': 'tuberculosis', 'Mucous plugging': 'mucous_plugging', 'Implanted hardware': 'hardware', 'Mass': 'mass', 'Pulmonary edema': 'pulmonary_edema', 'Lucency': 'lucency', 'Arthritis': 'arthritis', 'Pneumonia': 'pneumonia', 'Inflammation': 'inflammation', 'Bronchitis': 'bronchitis', 'Fracture': 'fracture', 'Secretion': 'secretion', 'Congestion': 'congestion', 'Soft tissue abnormality': 'soft_tissue', 'Breast implant': 'breast_implant', 'Breast surgery': 'breast_surgery', 'Chest tube': 'chest_tube', 'Aspiration': 'aspiration', 'Post-surgical changes': 'postsurgical', 'Pneumothorax': 'pneumothorax', 'Tracheal tube': 'tracheal_tube', 'Honeycombing': 'honeycombing', 'Airspace disease': 'airspace_disease', 'Heart failure': 'heart_failure', 'Plaque': 'plaque', 'Gastrointestinal tube': 'gi_tube', 'Reticulation': 'reticulation', 'Aneurysm': 'aneurysm', 'Surgical staples': 'staple', 'Coronary artery disease': 'coronary_artery_disease', 'Pacemaker or defibrillator': 'pacemaker_or_defib', 'Distention': 'distention', 'Infiltrate': 'infiltrate', 'Transplant': 'transplant', 'Surgical clip': 'clip', 'Abnormal density': 'density', 'Interstitial lung disease': 'interstitial_lung_disease', 'Scattered nodules': 'scattered_nod', 'Suture': 'suture', 'Bronchiolectasis': 'bronchiolectasis', 'Atherosclerosis': 'atherosclerosis', 'Stent': 'stent', 'Cyst': 'cyst', 'Band-like or linear opacity': 'bandlike_or_linear', 'Pericardial thickening': 'pericardial_thickening', 'Granuloma': 'granuloma', 'Pneumonitis': 'pneumonitis', 'Cancer': 'cancer'}
    # Exclude Mosaic attenuation and apply the mapping
    def map_labels(row):
        mapped_row = {}
        for ct_label, rad_label  in label_mapping.items():
            if rad_label in row:
                mapped_row[ct_label] = row[rad_label] if rad_label in row else 0
        # Exclude Mosaic attenuation and add calcification handling
        return mapped_row

    # Apply the mapping function to each row in the RAD-ChestCT dataframe
    mapped_labels = rad_chestct_df.apply(map_labels, axis=1)

  
    # Create a new dataframe with the mapped labels
    mapped_df = pd.DataFrame(mapped_labels.tolist())

    # map exits files
    exits_files = pd.read_csv('./Dataset/Rad_ChestCT_all_image.csv')
    exits_files = [ path_name.split('/')[-1].split('.')[0] for path_name in exits_files['Path'].tolist()]
    mapped_df = mapped_df[mapped_df['Path'].isin(exits_files)]

    label = mapped_df.iloc[:, 1:].values
    # remove Coronary artery wall calcification

    # Load the CT-RATE labels
    predict = pd.read_csv(predict_csv).values
    # neg_predict = pd.read_csv(neg_predict_csv).values
    ipdb.set_trace()
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)

    save_npz(predict, label, './')

    # print(f"Best thresholds: {best_thresholds}")
    print(f"Macro AUC: {macro_auc}")
    # print(f"Total F1 score: {macro_f1}")
    # print(f"Accuracy: {accuracy}")
    # print(f"Macro Precision: {macro_precision}")


def triple_Rad_ChestCT_rest_56_result(predict_csv):
    # Load the CSV files
    rad_chestct_df = pd.read_csv("./Dataset/merged_abnormality_labels_with_id.csv")  # Replace with the actual file path

       # Mapping dictionary: RAD-ChestCT labels to CT-RATE labels (in your original order)
    label_mapping ={'Path' : 'NoteAcc_DEID', 'Intravenous Catheter or Port': 'catheter_or_port', 'Pleural thickening': 'pleural_thickening', 'Coronary artery bypass graft': 'cabg', 'Infection': 'infection', 'Tree-in-bud pattern': 'tree_in_bud', 'Lung resection': 'lung_resection', 'Debris': 'debris', 'Air trapping': 'air_trapping', 'Cavitation': 'cavitation', 'Scattered calcifications': 'scattered_calc', 'Hemothorax': 'hemothorax', 'Heart valve replacement': 'heart_valve_replacement', 'Dilation or ectasia': 'dilation_or_ectasia', 'Sternotomy': 'sternotomy', 'Lesion': 'lesion', 'Deformity': 'deformity', 'Fibrosis': 'fibrosis', 'Bronchiolitis': 'bronchiolitis', 'Ground-glass opacity': 'groundglass', 'Tuberculosis': 'tuberculosis', 'Mucous plugging': 'mucous_plugging', 'Implanted hardware': 'hardware', 'Mass': 'mass', 'Pulmonary edema': 'pulmonary_edema', 'Lucency': 'lucency', 'Arthritis': 'arthritis', 'Pneumonia': 'pneumonia', 'Inflammation': 'inflammation', 'Bronchitis': 'bronchitis', 'Fracture': 'fracture', 'Secretion': 'secretion', 'Congestion': 'congestion', 'Soft tissue abnormality': 'soft_tissue', 'Breast implant': 'breast_implant', 'Breast surgery': 'breast_surgery', 'Chest tube': 'chest_tube', 'Aspiration': 'aspiration', 'Post-surgical changes': 'postsurgical', 'Pneumothorax': 'pneumothorax', 'Tracheal tube': 'tracheal_tube', 'Honeycombing': 'honeycombing', 'Airspace disease': 'airspace_disease', 'Heart failure': 'heart_failure', 'Plaque': 'plaque', 'Gastrointestinal tube': 'gi_tube', 'Reticulation': 'reticulation', 'Aneurysm': 'aneurysm', 'Surgical staples': 'staple', 'Coronary artery disease': 'coronary_artery_disease', 'Pacemaker or defibrillator': 'pacemaker_or_defib', 'Distention': 'distention', 'Infiltrate': 'infiltrate', 'Transplant': 'transplant', 'Surgical clip': 'clip', 'Abnormal density': 'density', 'Interstitial lung disease': 'interstitial_lung_disease', 'Scattered nodules': 'scattered_nod', 'Suture': 'suture', 'Bronchiolectasis': 'bronchiolectasis', 'Atherosclerosis': 'atherosclerosis', 'Stent': 'stent', 'Cyst': 'cyst', 'Band-like or linear opacity': 'bandlike_or_linear', 'Pericardial thickening': 'pericardial_thickening', 'Granuloma': 'granuloma', 'Pneumonitis': 'pneumonitis', 'Cancer': 'cancer'}

    # Define the categories to be removed
    categories_to_remove = ['Intravenous Catheter or Port', 'Tracheal tube', 'Chest tube', 'Breast implant', 'Pacemaker or defibrillator', 'Stent', 'Surgical clip', 'Surgical staples', 'Gastrointestinal tube', 'Implanted hardware', 'Suture']

    list_label_map = list(label_mapping.keys())

    removed_indices = [i-1 for i, cat in enumerate(list_label_map) if cat in categories_to_remove]
    # Remove the specified categories from the mapping
    label_mapping = {key: value for key, value in label_mapping.items() if key not in categories_to_remove}
    
    # Mapping function
    def map_labels(row):
        mapped_row = {}
        for ct_label, rad_label in label_mapping.items():
            if rad_label in row:
                mapped_row[ct_label] = row[rad_label] if rad_label in row else 0
        return mapped_row

    # Apply the mapping function to each row in the RAD-ChestCT dataframe
    mapped_labels = rad_chestct_df.apply(map_labels, axis=1)

    # Create a new dataframe with the mapped labels
    mapped_df = pd.DataFrame(mapped_labels.tolist())

    # Map exits files
    exits_files = pd.read_csv('./Dataset/Rad_ChestCT_all_image.csv')
    exits_files = [path_name.split('/')[-1].split('.')[0] for path_name in exits_files['Path'].tolist()]
    mapped_df = mapped_df[mapped_df['Path'].isin(exits_files)]

    # Extract labels (excluding the removed categories)
    label = mapped_df.iloc[:, 1:].values
    # Load the CT-RATE predictions without headers
    predict = pd.read_csv(predict_csv).values
    # load npz file
    # predict = np.load('/data/haoranlai/Project/CT-CLIP/inference_zeroshot_v2_rest_label_radchestct/predicted_weights.npz')['data']
    # predict = np.load('/data/haoranlai/Project/Merlin/sim_result/RAD-ChestCT-56_similarity.npy')
    # Delete the corresponding columns from label and predict

    predict = np.delete(predict, removed_indices, axis=1)


    metrics(label, predict)

    # Evaluate AUC
    # macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)

    # save_npz(predict, label, './')

    # print(f"Macro AUC: {macro_auc}")


def triple_Rad_ChestCT_long_tail_result(predict_csv):
    # Load the RAD-ChestCT data
    rad_chestct_df = pd.read_csv("./Dataset/merged_abnormality_labels_with_id.csv")

    # Mapping dictionary: RAD-ChestCT labels to CT-RATE labels
    label_mapping = {
        'Path': 'NoteAcc_DEID', 'Intravenous Catheter or Port': 'catheter_or_port', 'Pleural thickening': 'pleural_thickening',
        'Coronary artery bypass graft': 'cabg', 'Infection': 'infection', 'Tree-in-bud pattern': 'tree_in_bud',
        'Lung resection': 'lung_resection', 'Debris': 'debris', 'Air trapping': 'air_trapping', 'Cavitation': 'cavitation',
        'Scattered calcifications': 'scattered_calc', 'Hemothorax': 'hemothorax', 'Heart valve replacement': 'heart_valve_replacement',
        'Dilation or ectasia': 'dilation_or_ectasia', 'Sternotomy': 'sternotomy', 'Lesion': 'lesion', 'Deformity': 'deformity',
        'Fibrosis': 'fibrosis', 'Bronchiolitis': 'bronchiolitis', 'Ground-glass opacity': 'groundglass', 'Tuberculosis': 'tuberculosis',
        'Mucous plugging': 'mucous_plugging', 'Implanted hardware': 'hardware', 'Mass': 'mass', 'Pulmonary edema': 'pulmonary_edema',
        'Lucency': 'lucency', 'Arthritis': 'arthritis', 'Pneumonia': 'pneumonia', 'Inflammation': 'inflammation',
        'Bronchitis': 'bronchitis', 'Fracture': 'fracture', 'Secretion': 'secretion', 'Congestion': 'congestion',
        'Soft tissue abnormality': 'soft_tissue', 'Breast implant': 'breast_implant', 'Breast surgery': 'breast_surgery',
        'Chest tube': 'chest_tube', 'Aspiration': 'aspiration', 'Post-surgical changes': 'postsurgical',
        'Pneumothorax': 'pneumothorax', 'Tracheal tube': 'tracheal_tube', 'Honeycombing': 'honeycombing',
        'Airspace disease': 'airspace_disease', 'Heart failure': 'heart_failure', 'Plaque': 'plaque',
        'Gastrointestinal tube': 'gi_tube', 'Reticulation': 'reticulation', 'Aneurysm': 'aneurysm', 'Surgical staples': 'staple',
        'Coronary artery disease': 'coronary_artery_disease', 'Pacemaker or defibrillator': 'pacemaker_or_defib',
        'Distention': 'distention', 'Infiltrate': 'infiltrate', 'Transplant': 'transplant', 'Surgical clip': 'clip',
        'Abnormal density': 'density', 'Interstitial lung disease': 'interstitial_lung_disease',
        'Scattered nodules': 'scattered_nod', 'Suture': 'suture', 'Bronchiolectasis': 'bronchiolectasis',
        'Atherosclerosis': 'atherosclerosis', 'Stent': 'stent', 'Cyst': 'cyst', 'Band-like or linear opacity': 'bandlike_or_linear',
        'Pericardial thickening': 'pericardial_thickening', 'Granuloma': 'granuloma', 'Pneumonitis': 'pneumonitis', 'Cancer': 'cancer'
    }

    # Map labels to new column names
    def map_labels(row):
        return {ct_label: row[rad_label] if rad_label in row else 0 for ct_label, rad_label in label_mapping.items()}

    # Apply mapping to create a new dataframe
    mapped_labels = rad_chestct_df.apply(map_labels, axis=1)
    mapped_df = pd.DataFrame(mapped_labels.tolist())

    # Filter by existing files
    exits_files = pd.read_csv('./Dataset/Rad_ChestCT_all_image.csv')
    exits_files = [path_name.split('/')[-1].split('.')[0] for path_name in exits_files['Path'].tolist()]
    mapped_df = mapped_df[mapped_df['Path'].isin(exits_files)]

    # Calculate frequency for each disease
    frequency = mapped_df.iloc[:, 1:].sum(axis=0)
    frequency = frequency.sort_values(ascending=False)

    # Categorize into head, medium, and tail
    head_threshold = frequency.quantile(0.75)
    tail_threshold = frequency.quantile(0.25)

    categories = frequency.apply(
        lambda x: 'head' if x >= head_threshold else ('tail' if x <= tail_threshold else 'medium')
    )

    # Create a results dataframe
    results_df = pd.DataFrame({'Disease': frequency.index, 'Frequency': frequency.values, 'Category': categories.values})
    results_df.reset_index(drop=True, inplace=True)

    # Load predictions and ground truth labels
    predict = pd.read_csv(predict_csv).values
    label = mapped_df.iloc[:, 1:].values

    # Function to calculate AUCs
    def calculate_category_aucs(category, frequency, categories):
        # Get indices of the specific category
        indices = [i for i, disease in enumerate(frequency.index) if categories[disease] == category]

        if len(indices) == 0:
            return None, None, None

        # Macro AUC
        macro_auc = np.mean([
            roc_auc_score(label[:, i], predict[:, i])
            for i in indices if np.sum(label[:, i]) > 0
        ])

        # Micro AUC
        micro_auc = roc_auc_score(label[:, indices].ravel(), predict[:, indices].ravel())

        # Weighted AUC
        label_sums = label[:, indices].sum(axis=0)
        weights = label_sums / label_sums.sum()
        weighted_auc = np.sum([
            weights[i] * roc_auc_score(label[:, indices[i]], predict[:, indices[i]])
            for i in range(len(indices)) if label_sums[i] > 0
        ])

        return macro_auc, micro_auc, weighted_auc
    
    # Calculate AUCs for head, medium, and tail categories
    head_aucs = calculate_category_aucs('head', frequency, categories)
    medium_aucs = calculate_category_aucs('medium', frequency, categories)
    tail_aucs = calculate_category_aucs('tail', frequency, categories)

    # Save results
    results_df.to_csv("disease_frequency_and_categories.csv", index=False)

    # Print results
    print("Disease frequency and categories saved to 'disease_frequency_and_categories.csv'.")
    print(f"Head Macro AUC: {head_aucs[0]:.4f}, Micro AUC: {head_aucs[1]:.4f}, Weighted AUC: {head_aucs[2]:.4f}")
    print(f"Medium Macro AUC: {medium_aucs[0]:.4f}, Micro AUC: {medium_aucs[1]:.4f}, Weighted AUC: {medium_aucs[2]:.4f}")
    print(f"Tail Macro AUC: {tail_aucs[0]:.4f}, Micro AUC: {tail_aucs[1]:.4f}, Weighted AUC: {tail_aucs[2]:.4f}")

    return results_df, head_aucs, medium_aucs, tail_aucs



def triple_Xiehe_result(predict_csv):
    path = "/data/haoranlai/Dataset/Xiehe/CQ_cleaned_metadata.csv"
    image_dir = "/data/haoranlai/Dataset/Xiehe/xiehe_2_preprocess_256/"
    
    df = pd.read_csv(path)
    # 构建 image filename 到 label 的映射
    df['filename'] = df['影像号'].astype(str).str.upper() + '.nii.gz'
    df['label_vector'] = df['label_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=int))

    # 获取图像实际存在的列表
    available_files = set(os.listdir(image_dir))

    # 保留存在图像的行
    df = df[df['filename'].isin(available_files)].reset_index(drop=True)

    # 构造完整路径与标签向量
    image_list = [os.path.join(image_dir, fname) for fname in df['filename']]
    label_vectors = df['label_vector'].tolist()


    predict = pd.read_csv(predict_csv).values
    # neg_predict = pd.read_csv(neg_predict_csv).values
  
    key = df.keys().tolist()[1:]

    label = np.asarray(label_vectors)


    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)


    # print(f"Best thresholds: {best_thresholds}")
    print(f"Macro AUC: {macro_auc}")

    return predict, label



def triple_Xiehe_ertong_result(predict_csv):
    image_dir = "/data/haoranlai/Dataset/Xiehe/ertong_preprocess_256/"
    path = "/data/haoranlai/Dataset/Xiehe/ertong_cleaned_metadata.csv"
    
    df = pd.read_csv(path)
    # 构建 image filename 到 label 的映射
    df['filename'] = df['影像文件名'].astype(str)
    df['label_vector'] = df['label_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=int))

    # 获取图像实际存在的列表
    available_files = set(os.listdir(image_dir))

    # 保留存在图像的行
    df = df[df['filename'].isin(available_files)].reset_index(drop=True)

    # 构造完整路径与标签向量
    image_list = [os.path.join(image_dir, fname) for fname in df['filename']]
    label_vectors = df['label_vector'].tolist()



    predict = pd.read_csv(predict_csv).values
    # neg_predict = pd.read_csv(neg_predict_csv).values
  
    key = df.keys().tolist()[1:]

    label = np.asarray(label_vectors)

    
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)


    # print(f"Best thresholds: {best_thresholds}")
    print(f"Macro AUC: {macro_auc}")

    return predict, label



def process_reports(captions):
    # use space instead of newline
    captions = captions.replace("\n", " ")

    # split sentences
    splitter = re.compile("[0-9]+\.")
    captions = splitter.split(captions)
    captions = [point.split(".") for point in captions]
    captions = [sent for point in captions for sent in point]

    cnt = 0
    study_sent = []
    # create tokens from captions
    for cap in captions:

        if len(cap) == 0:
            continue

        cap = cap.replace("\ufffd\ufffd", " ")
        # picks out sequences of alphanumeric characters as tokens
        # and drops everything else
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(cap.lower())

        # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
        if len(tokens) <= 1:
            # if len(tokens) < 3:
            continue

        # filter tokens for current sentence
        included_tokens = []
        for t in tokens:
            t = t.encode("ascii", "ignore").decode("ascii")
            if len(t) > 0:
                included_tokens.append(t)
        study_sent.append(" ".join(included_tokens))

    return study_sent

def write_norm_reports():
    path = "/data/haoranlai/Dataset/CT-RATE/Info/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv"
    reports_path = "/data/haoranlai/Dataset/CT-RATE/Info/radiology_text_reports/dataset_radiology_text_reports_validation_reports.csv"

    # Load predicted and labels data
    df = pd.read_csv(path)
    reports_df = pd.read_csv(reports_path)
    # Extracting the arrays from the loaded files
    key = df.keys().tolist()[1:]
    labels = df[key].values
    reports = reports_df['Findings_EN'].tolist()

    # Compute the frequency of each disease based on label matrix
    disease_frequencies = labels.sum(axis=1)  # Sum across rows to get frequency per column (disease)
    index = np.where(disease_frequencies == 0)[0]

    # get total normal reports
    normal_reports = [reports[i] for i in index]

    split_normal_reports = []
    for report in normal_reports:
        split_normal_reports.extend(process_reports(report))
    
    unique_reports = list(set(split_normal_reports))

    # create dict for unique reports
    unique_reports = {i: report for i, report in enumerate(unique_reports)}

    # write to json
    with open("/data/haoranlai/Project/gloria/Dataset/valid_norm_reports.json", "w") as f:
        json.dump(unique_reports, f)

    

if __name__ == '__main__':
    # write_norm_reports()
    # ipdb.set_trace()


    images = [ 
               './Dataset/CT-RATE_valid_image.csv',
               './Dataset/Rad_ChestCT_all_image.csv',
                './Dataset/CT-RATE_valid_image.csv',
                './Dataset/Rad_ChestCT_all_image.csv',
                './Dataset/Rad_ChestCT_all_image.csv',
                './Dataset/Rad_ChestCT_all_image.csv',
                './Dataset/CT-RATE_valid_image.csv',
                './Dataset/CT-RATE_valid_image.csv',
                './Dataset/xiehe_1.csv',
                './Dataset/xiehe_1_ertong.csv',
                '/data2/haoranlai/Project/gloria/Dataset/CT-RATE-train_image.csv',
                './Dataset/Rad_ChestCT_all_image.csv',
                './Dataset/Rad_ChestCT_all_image.csv',
                '/data2/haoranlai/Project/gloria/Dataset/ReXGroundingCT_valid_image.csv',
                './Dataset/INSPECT_image.csv'

               ]
    
    # # positive
    texts = [ 
               './Dataset/CT-RATE_valid_text.json',
                './Dataset/CT-RATE_valid_text.json',
                './Dataset/CT-RATE-LT-text.json',
                './Dataset/CT-RATE_valid_universal_text.json',
                './Dataset/valid_norm_reports.json',
                './Dataset/RADChest-CT-text-rest.json',
                './Dataset/CT-RATE-valid-reports.json',
                './Dataset/CT-RATE_valid_universal_text_100.json',
                './Dataset/virus_valid_text.json',
                './Dataset/virus_valid_ertong_text.json',
                './Dataset/CT-RATE_valid_text.json',
                './Dataset/CT-RATE_valid_text_no.json',
                './Dataset/RADChest-CT-text-rest_no.json',
                './Dataset/CT-RATE_valid_text.json',
                './Dataset/INSPECT_valid.json'
               ]
    



    result_file_name = 'BrgSA_best_model'

    os.makedirs('./Performance/'+ result_file_name, exist_ok=True)

    # positive
    save_csvs = [     
                     './Performance/'+ result_file_name +'/CT-RATE_epoch38.csv',
                     './Performance/'+ result_file_name +'/Rad_ChestCT_resolution0.8_epoch38.csv',
                     './Performance/'+ result_file_name +'/CT-RATE_LT_epoch38.csv',
                     './Performance/'+ result_file_name +'/Rad_ChestCT_universal_epoch38.csv',
                     './Performance/'+ result_file_name +'/Rad_ChestCT_normal_epoch38.csv',
                    './Performance/'+ result_file_name +'/Rad_ChestCT_resolution0.8_rest_label_epoch38.csv',
                    '',
                    '',
                    #  '/data/haoranlai/Project/gloria/InMap/inmap_multilabel_probs.csv'
                    './Performance/'+ result_file_name +'/xiehe_1_label_epoch38.csv',
                    './Performance/'+ result_file_name +'/xiehe_1_ertong_label_epoch38.csv',
                    '',
                     './Performance/'+ result_file_name +'/Rad_ChestCT_resolution0.8_no_epoch38.csv',
                     './Performance/'+ result_file_name +'/Rad_ChestCT_resolution0.8_rest_label_no_epoch38.csv',
                     './Performance/'+ result_file_name +'/ReXGroundingCT_epoch38.csv',
                     './Performance/'+ result_file_name +'/INSPECT_epoch38.csv',
                     ]
    
    
    for i, (img, txt, savecsv) in  enumerate(zip(images, texts, save_csvs)): 
        if i == 0 or i == 1 or i == 2:
                start = time.time()
                similarities, text_emb_g = obtain_sim(img, txt)
                similarities.to_csv(savecsv, index=False)
                print(time.time() - start)

    print('triple_CT_RATE_result')
    predict, label = triple_CT_RATE_result(save_csvs[0])

    save_npz(predict, label, dataset='CT-RATE')


    print('triple_RAD_ChestCT_result')
    predict, label = triple_Rad_ChestCT_new_result(save_csvs[1])

    save_npz(predict, label, dataset='RAD_ChestCT')

    print('triple_CT_RATE_GPT4_result')
    predict, label = triple_CT_RATE_GPT4_result(save_csvs[2])

    save_npz(predict, label, dataset='CT-RATE-LT')



 
 


    