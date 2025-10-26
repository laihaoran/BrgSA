import os
import math
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# tqdm 进度条（无则降级为空实现）
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False
    class _DummyTqdm:
        def __init__(self, it, **kwargs): self.it = it
        def __iter__(self): return iter(self.it)
        def set_postfix(self, **kwargs): pass
    def tqdm(it, **kwargs): return _DummyTqdm(it)

# MONAI
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose, LoadImage, ToTensor, AddChannel,
    RandSpatialCrop, CenterSpatialCrop,
    RandRotate90, RandFlip,
    RandScaleIntensity, RandShiftIntensity,
    ScaleIntensityRange, NormalizeIntensity,
)

# 示例 3D ResNet（如有自定义模型请替换）
from monai.networks.nets import ResNet as ResNet3D

# 可选指标（未安装则自动跳过）
_HAS_TORCHMETRICS = True
try:
    from torchmetrics.classification import (
        MulticlassAUROC, MultilabelAUROC,
        MulticlassAveragePrecision, MultilabelAveragePrecision
    )
except Exception:
    _HAS_TORCHMETRICS = False


# =========================
# 配置
# =========================
@dataclass
class TrainCfg:
    # 数据
    train_csv: str
    val_csv: str
    train_data_root: str
    valid_data_root: str
    num_workers: int = 16
    cache_rate: float = 0.0

    # 任务与类别
    task: str = "multilabel"   # "multilabel" 或 "multiclass"
    num_classes: int = 2
    multilabel_threshold: float = 0.5

    # 训练
    epochs: int = 50
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    grad_clip_norm: float = 1.0
    accum_steps: int = 1
    amp: bool = True

    # 调度
    warmup_epochs: int = 0
    cosine_final_lr: float = 1e-6

    # 其他
    seed: int = 42
    out_dir: str = "./outputs"
    save_best_metric: str = "val_f1"  # 可选：val_acc / val_f1 / val_auroc / val_map

    # 供 build_transformation_3D_M3D 使用（你的增强参数外壳）
    transforms: object = None


# =========================
# 实用函数
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ====== 从 CSV 读样本并自动拼子路径 ======
def load_items_from_csv(csv_path: str, root_dir: str):
    """
    CSV（首列 VolumeName，余下为 0/1 多标签）:
      VolumeName,Label1,Label2,...
      valid_1_a_1.nii.gz,0,1,...

    磁盘路径：
      <root_dir>/valid_1/valid_1_a/valid_1_a_1.nii.gz
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)

    # 清理列名里的多余空格与换行断裂
    clean_cols = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    df.columns = clean_cols

    image_col_csv = "VolumeName"
    assert image_col_csv in df.columns, f"{image_col_csv} not found in CSV columns: {df.columns.tolist()}"

    # 标签列：除 VolumeName 外全部
    label_cols = [c for c in df.columns if c != image_col_csv]

    # 数值化 + 缺失填 0
    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

    items = []
    missing = []
    for _, row in df.iterrows():
        fname = str(row[image_col_csv])  # e.g. valid_1_a_1.nii.gz
        # 去扩展名（兼容 .nii.gz）
        stem = fname[:-7] if fname.endswith(".nii.gz") else os.path.splitext(fname)[0]
        parts = stem.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected VolumeName format: {fname}. Expect like 'valid_1_a_1.nii.gz'")

        first = "_".join(parts[:2])   # valid_1
        second = "_".join(parts[:3])  # valid_1_a
        full_path = os.path.join(root_dir, first, second, fname)

        if not os.path.isfile(full_path):
            missing.append(full_path)

        labels = row[label_cols].values.astype("float32")
        items.append({"image": full_path, "label": labels})

    if missing:
        raise FileNotFoundError(f"{len(missing)} files not found. First few: {missing[:5]}")

    return items, label_cols


# ====== 适配器：仅对 data['image'] 应用影像级 transforms ======
class ImageOnlyTransform:
    def __init__(self, img_tfms):
        self.img_tfms = img_tfms

    def __call__(self, data):
        img = self.img_tfms(data["image"])
        return {"image": img, "label": data["label"]}


# ====== 影像级（非 dict）transforms 构建 ======
def build_transformation_3D_M3D(cfg, split):
    t = []
    # 这里不放 LoadImage；在数据集处统一处理
    t.append(ToTensor(dtype=torch.float))
    t.append(AddChannel())

    if split == "train":
        if cfg.transforms.rand_spatial_crop is not None:
            t.append(RandSpatialCrop(roi_size=cfg.transforms.rand_spatial_crop.roi_size, random_size=False))

        if cfg.transforms.random_rotate90 is not None:
            t.append(RandRotate90(
                prob=cfg.transforms.random_rotate90.prob,
                spatial_axes=cfg.transforms.random_rotate90.spatial_axes
            ))

        if cfg.transforms.random_flip is not None:
            if getattr(cfg.transforms.random_flip, "axis_0_prob", None) is not None:
                t.append(RandFlip(prob=cfg.transforms.random_flip.axis_0_prob, spatial_axis=0))
            if getattr(cfg.transforms.random_flip, "axis_1_prob", None) is not None:
                t.append(RandFlip(prob=cfg.transforms.random_flip.axis_1_prob, spatial_axis=1))
            if getattr(cfg.transforms.random_flip, "axis_2_prob", None) is not None:
                t.append(RandFlip(prob=cfg.transforms.random_flip.axis_2_prob, spatial_axis=2))

        if cfg.transforms.random_scale_intensity is not None:
            t.append(RandScaleIntensity(
                factors=cfg.transforms.random_scale_intensity.factors,
                prob=cfg.transforms.random_scale_intensity.prob
            ))

        if cfg.transforms.random_shift_intensity is not None:
            t.append(RandShiftIntensity(
                offsets=cfg.transforms.random_shift_intensity.offsets,
                prob=cfg.transforms.random_shift_intensity.prob
            ))
    else:
        if cfg.transforms.rand_spatial_crop is not None:
            t.append(CenterSpatialCrop(roi_size=cfg.transforms.rand_spatial_crop.roi_size))

    if cfg.transforms.scale_intensity_range is not None:
        t.append(ScaleIntensityRange(
            a_min=cfg.transforms.scale_intensity_range.a_min,
            a_max=cfg.transforms.scale_intensity_range.a_max,
            b_min=cfg.transforms.scale_intensity_range.b_min,
            b_max=cfg.transforms.scale_intensity_range.b_max,
            clip=cfg.transforms.scale_intensity_range.clip
        ))

    if cfg.transforms.norm is not None:
        if cfg.transforms.norm.type == "standard":
            t.append(NormalizeIntensity(
                subtrahend=cfg.transforms.norm.subtrahend,
                divisor=cfg.transforms.norm.divisor
            ))
        else:
            raise NotImplementedError("Normalization method not implemented")

    return Compose(t)


# ====== 构建数据集与加载器 ======
def make_datasets(cfg: TrainCfg, monai_transforms_train, monai_transforms_val):
    # 加载 + 影像级 transforms
    img_tfms_train = Compose([LoadImage(image_only=True), monai_transforms_train])
    img_tfms_val   = Compose([LoadImage(image_only=True), monai_transforms_val])

    train_items, label_cols = load_items_from_csv(cfg.train_csv, cfg.train_data_root)
    val_items,   _label2    = load_items_from_csv(cfg.val_csv,   cfg.valid_data_root)
    assert label_cols == _label2, "Train/Val 标签列不一致，请检查 CSV"

    if cfg.cache_rate and cfg.cache_rate > 0:
        train_ds = CacheDataset(train_items, transform=ImageOnlyTransform(img_tfms_train),
                                cache_rate=cfg.cache_rate, num_workers=cfg.num_workers)
        val_ds = CacheDataset(val_items, transform=ImageOnlyTransform(img_tfms_val),
                              cache_rate=min(cfg.cache_rate, 0.1), num_workers=cfg.num_workers)
    else:
        train_ds = Dataset(train_items, transform=ImageOnlyTransform(img_tfms_train))
        val_ds   = Dataset(val_items,   transform=ImageOnlyTransform(img_tfms_val))

    return train_ds, val_ds, label_cols


def make_dataloaders(train_ds, val_ds, cfg: TrainCfg):
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=bool(cfg.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=bool(cfg.num_workers > 0)
    )
    return train_loader, val_loader


# ====== 损失与后处理 ======
def get_loss_and_postproc(cfg: TrainCfg):
    if cfg.task == "multiclass":
        loss = nn.CrossEntropyLoss()
        def postproc(logits):
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            return probs, pred
    elif cfg.task == "multilabel":
        loss = nn.BCEWithLogitsLoss()
        def postproc(logits):
            probs = torch.sigmoid(logits)
            pred = (probs >= cfg.multilabel_threshold).float()
            return probs, pred
    else:
        raise ValueError(f"Unknown task: {cfg.task}")
    return loss, postproc


# ====== 指标 ======
def f1_score(pred, target, task: str):
    if task == "multiclass":
        C = int(max(int(pred.max().item()) + 1, int(target.max().item()) + 1))
        f1s = []
        for c in range(C):
            tp = ((pred == c) & (target == c)).sum().item()
            fp = ((pred == c) & (target != c)).sum().item()
            fn = ((pred != c) & (target == c)).sum().item()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            f1s.append(f1)
        return sum(f1s) / len(f1s)
    else:
        B, C = target.shape
        f1s = []
        for c in range(C):
            p = pred[:, c]
            t = target[:, c]
            tp = (p * t).sum().item()
            fp = (p * (1 - t)).sum().item()
            fn = ((1 - p) * t).sum().item()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            f1s.append(f1)
        return sum(f1s) / len(f1s)


def accuracy(pred, target, task: str):
    if task == "multiclass":
        return (pred == target).float().mean().item()
    else:
        # multilabel：per-label 平均
        return ((pred == target).float().mean(dim=1)).mean().item()


def build_metrics(cfg: TrainCfg, device):
    metrics = {}
    if _HAS_TORCHMETRICS:
        if cfg.task == "multiclass":
            metrics["auroc"] = MulticlassAUROC(num_classes=cfg.num_classes).to(device)
            metrics["map"]   = MulticlassAveragePrecision(num_classes=cfg.num_classes).to(device)
        else:
            metrics["auroc"] = MultilabelAUROC(num_labels=cfg.num_classes, average="macro").to(device)
            metrics["map"]   = MultilabelAveragePrecision(num_labels=cfg.num_classes, average="macro").to(device)
    return metrics


def update_metrics(metrics, probs, target, cfg: TrainCfg):
    """
    关键修复点：torchmetrics 对 multilabel 的 target 需要是 int/bool 类型。
    这里统一在内部转换，避免外部忘记强转导致报错。
    """
    if not metrics:
        return
    t = target
    if cfg.task == "multiclass":
        if t.ndim > 1:
            t = t.argmax(dim=-1)
        t = t.long()
        metrics["auroc"].update(probs, t)
        metrics["map"].update(probs, t)
    else:
        # 多标签：确保 target 为整型/布尔
        if t.dtype.is_floating_point:
            t = (t > 0.5).int()
        else:
            t = t.int()
        metrics["auroc"].update(probs, t)
        metrics["map"].update(probs, t)


def compute_metrics(metrics):
    out = {}
    if not metrics:
        return out
    for k, m in metrics.items():
        try:
            out[k] = float(m.compute().item())
            m.reset()
        except Exception:
            pass
    return out


# ====== 学习率调度 ======
def cosine_lr(optimizer, base_lr, final_lr, cur_epoch, max_epoch, warmup_epochs=0):
    if cur_epoch < warmup_epochs:
        lr = base_lr * (cur_epoch + 1) / max(1, warmup_epochs)
    else:
        t = (cur_epoch - warmup_epochs) / max(1, (max_epoch - warmup_epochs))
        lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# =========================
# 训练与验证（含进度条）
# =========================
def train_one_epoch(model, loader, optimizer, loss_fn, postproc, scaler, cfg: TrainCfg, device,
                    epoch: int = 0, max_epochs: int = 0):
    model.train()
    running_loss, running_acc, running_f1 = 0.0, 0.0, 0.0
    metrics_tm = build_metrics(cfg, device)
    optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, desc=f"Train [{epoch+1}/{max_epochs}]", leave=False) if _HAS_TQDM else loader

    for step, batch in enumerate(iterator, start=1):
        x = batch["image"].to(device)
        y = torch.as_tensor(batch["label"]).to(device)

        if cfg.task == "multiclass":
            if y.ndim > 1:
                y = y.argmax(dim=-1).long()
            else:
                y = y.long()

        with autocast(enabled=cfg.amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        if cfg.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            if cfg.amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        if step % cfg.accum_steps == 0:
            if cfg.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            probs, pred = postproc(logits.detach())
            running_loss += loss.item()
            running_acc  += accuracy(pred, y, cfg.task)
            y_for_f1 = y if cfg.task == "multiclass" else y.float()
            running_f1 += f1_score(pred.float(), y_for_f1, cfg.task)

            # 注意：这里把 y 原样传入，update_metrics 内部会做 dtype 兼容处理
            update_metrics(metrics_tm, probs, y, cfg)

        if _HAS_TQDM:
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    out = {"loss": running_loss / n, "acc": running_acc / n, "f1": running_f1 / n}
    out.update({k: v for k, v in compute_metrics(metrics_tm).items()})
    return out


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, postproc, cfg: TrainCfg, device,
                   epoch: int = 0, max_epochs: int = 0):
    model.eval()
    running_loss, running_acc, running_f1 = 0.0, 0.0, 0.0
    metrics_tm = build_metrics(cfg, device)

    iterator = tqdm(loader, desc=f"Valid [{epoch+1}/{max_epochs}]", leave=False) if _HAS_TQDM else loader

    for batch in iterator:
        x = batch["image"].to(device)
        y = torch.as_tensor(batch["label"]).to(device)

        if cfg.task == "multiclass":
            if y.ndim > 1:
                y = y.argmax(dim=-1).long()
            else:
                y = y.long()

        logits = model(x)
        loss = loss_fn(logits, y)
        probs, pred = postproc(logits)

        running_loss += loss.item()
        running_acc  += accuracy(pred, y, cfg.task)
        y_for_f1 = y if cfg.task == "multiclass" else y.float()
        running_f1 += f1_score(pred.float(), y_for_f1, cfg.task)
        update_metrics(metrics_tm, probs, y, cfg)

        if _HAS_TQDM:
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    out = {"loss": running_loss / n, "acc": running_acc / n, "f1": running_f1 / n}
    out.update({k: v for k, v in compute_metrics(metrics_tm).items()})
    return out


def save_checkpoint(state: Dict, is_best: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "last.pt")
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(state, os.path.join(out_dir, "best.pt"))


def train_loop(model: nn.Module, cfg: TrainCfg, monai_cfg):
    """
    model: 3D 分类模型
    monai_cfg: 外壳对象，内含 .transforms（你的增强参数）
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Transforms
    t_train = build_transformation_3D_M3D(monai_cfg, split="train")
    t_val   = build_transformation_3D_M3D(monai_cfg, split="val")

    # Datasets & Loaders
    train_ds, val_ds, label_cols = make_datasets(cfg, t_train, t_val)
    if cfg.task == "multiclass":
        cfg.num_classes = cfg.num_classes or int(len(label_cols))
    else:
        cfg.num_classes = len(label_cols)
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, cfg)

    # Loss & postproc
    loss_fn, postproc = get_loss_and_postproc(cfg)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    scaler = GradScaler(enabled=cfg.amp)

    best_metric_value = -1e9
    for epoch in range(cfg.epochs):
        cur_lr = cosine_lr(optimizer, base_lr=cfg.lr, final_lr=cfg.cosine_final_lr,
                           cur_epoch=epoch, max_epoch=cfg.epochs, warmup_epochs=cfg.warmup_epochs)

        train_log = train_one_epoch(model, train_loader, optimizer, loss_fn, postproc, scaler, cfg, device,
                                    epoch=epoch, max_epochs=cfg.epochs)
        val_log   = eval_one_epoch(model, val_loader,   loss_fn, postproc, cfg, device,
                                   epoch=epoch, max_epochs=cfg.epochs)

        # 选择 best 指标
        pick = cfg.save_best_metric
        metric_now = val_log.get(pick.replace("val_", ""), None) if pick.startswith("val_") else val_log.get(pick, None)
        if metric_now is None:
            key = pick[4:] if pick.startswith("val_") else pick
            metric_now = val_log.get(key, None)

        is_best = metric_now is not None and metric_now > best_metric_value
        if is_best:
            best_metric_value = metric_now

        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg.__dict__,
            "label_cols": label_cols,
            "val_log": val_log
        }, is_best=is_best, out_dir=cfg.out_dir)

        head = f"[{epoch+1}/{cfg.epochs}] lr={cur_lr:.2e}"
        tr   = " | ".join([f"train_{k}={v:.4f}" for k, v in train_log.items()])
        vl   = " | ".join([f"val_{k}={v:.4f}" for k, v in val_log.items()])
        print(f"{head}  {tr}  ||  {vl}")

    print(f"Training done. Best {cfg.save_best_metric} = {best_metric_value:.4f} (saved to {cfg.out_dir}/best.pt)")


# =========================
# 示例 transforms 配置
# =========================
class TransformsCfg:
    def __init__(self):
        self.rand_spatial_crop = type("X", (), {"roi_size": (224, 224, 112)})
        self.random_rotate90   = type("X", (), {"prob": 0.0, "spatial_axes": (0, 1)})
        self.random_flip       = type("X", (), {"axis_0_prob": 0.0, "axis_1_prob": 0.5, "axis_2_prob": None})
        self.random_scale_intensity = type("X", (), {"factors": 0.0, "prob": 0.5})
        self.random_shift_intensity = type("X", (), {"offsets": 0.0, "prob": 0.5})
        self.scale_intensity_range  = type("X", (), {"a_min": -1000, "a_max": 1000, "b_min": -1, "b_max": 1, "clip": True})
        self.norm = type("X", (), {"type": "standard", "subtrahend": 0.4978, "divisor": 0.2449})


# =========================
# 入口（示例）
# =========================
if __name__ == "__main__":
    # 1) 模型（如有自定义模型，替换这里）
    model = ResNet3D(
        spatial_dims=3, n_input_channels=1, num_classes=18,
        conv1_t_stride=2, block='bottleneck',
        layers=[2, 2, 2, 2], block_inplanes=[64, 128, 256, 512]
    )

    # 2) 训练配置
    cfg = TrainCfg(
        train_csv="/data4/haoranlai/Dataset/CT-RATE/Info/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv",
        val_csv="/data4/haoranlai/Dataset/CT-RATE/Info/multi_abnormality_labels/dataset_multi_abnormality_labels_valid_predicted_labels.csv",
        train_data_root="/data4/haoranlai/Dataset/CT-RATE/train_fixed_256_128_high/",
        valid_data_root="/data4/haoranlai/Dataset/CT-RATE/valid_fixed_256_128_high/",
        task="multilabel",
        num_classes=18,
        batch_size=64,
        num_workers=16,
        epochs=50,
        lr=1e-4,
        out_dir="./outputs_cls3d",
        transforms=None,   # 由 monai_cfg 控制，这里占位
    )

    # 3) transforms 外壳（把你的参数放进来）
    monai_cfg = type("C", (), {})()
    monai_cfg.transforms = TransformsCfg()

    # 4) 开训
    train_loop(model, cfg, monai_cfg)
