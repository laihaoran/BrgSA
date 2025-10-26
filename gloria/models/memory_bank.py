import torch
import torch.nn.functional as F
import torch.nn as nn


class MemoryBank(nn.Module):
    def __init__(self, cfg):
        super(MemoryBank, self).__init__()
        """
        初始化 Memory Bank
        Args:
            bank_size: int, Memory Bank 的大小 (e.g., 2048)
            feature_dim: int, 每个特征的维度 (e.g., 768)
            beta: float, 动量参数，用于动态更新特征 (e.g., 0.9)
            device: str, 使用的设备 (e.g., "cuda" or "cpu")
        """
        self.bank_size = cfg.model.MemoryBank.bank_size
        self.feature_dim = cfg.model.MemoryBank.feature_dim
        self.beta = cfg.model.MemoryBank.beta
        # Memory Bank 存储为可注册的缓冲区
        self.register_buffer("memory_bank", torch.zeros(self.bank_size, self.feature_dim))

        self.pointer = 0  # FIFO 指针

    def _initialize_memory_bank(self):
        """
        使用正交初始化生成 Memory Bank
        Returns:
            初始化后的 Memory Bank
        """
        memory_bank = torch.empty(self.bank_size, self.feature_dim)
        with torch.no_grad():  # 确保初始化不会记录梯度
            if self.bank_size >= self.feature_dim:
                # 如果行数大于等于列数，直接正交初始化
                nn.init.orthogonal_(memory_bank)
            else:
                # 如果行数小于列数，生成多余列，正交后裁剪
                temp = torch.empty(self.feature_dim, self.feature_dim)
                nn.init.orthogonal_(temp)
                memory_bank = temp[:self.bank_size]
        return memory_bank

    def update(self, image_features):
        """
        更新 Memory Bank
        Args:
            image_features: tensor of shape (batch_size, feature_dim), 当前 batch 的图像特征
        """
        batch_size = image_features.size(0)
        tail_indices = torch.arange(self.pointer, self.pointer + batch_size).to(image_features.device) % self.bank_size

        with torch.no_grad():  # 禁用梯度跟踪
            # 动态聚合更新
            for i, idx in enumerate(tail_indices):
                self.memory_bank[idx] = self.beta * self.memory_bank[idx] + (1 - self.beta) * image_features[i]

        # 更新 FIFO 指针
        self.pointer = (self.pointer + batch_size) % self.bank_size

    def retrieve(self):
        """
        获取 Memory Bank
        Returns:
            memory_bank: tensor, 当前 Memory Bank
        """
        return self.memory_bank


class TextFeatureRefiner(nn.Module):
    def __init__(self, memory_bank):
        super(TextFeatureRefiner, self).__init__()
        """
        初始化文本特征修正模块
        Args:
            memory_bank: MemoryBank 对象，动态存储图像特征
        """
        self.memory_bank = memory_bank

    def refine(self, text_features, alpha=0.1, temperature=0.07):
        """
        修正文本特征
        Args:
            text_features: tensor of shape (batch_size, feature_dim), 输入的文本特征
            alpha: float, 融合权重，控制文本特征与图像特征的平衡
            temperature: float, 温度系数，用于软化相似度分布
        Returns:
            refined_features: tensor of shape (batch_size, feature_dim), 修正后的文本特征
        """
        batch_size, feature_dim = text_features.size()
        memory_bank_features = self.memory_bank.retrieve()
        bank_size = memory_bank_features.size(0)

        # 计算文本特征与 Memory Bank 图像特征的余弦相似度
        text_norm = F.normalize(text_features, dim=1)
        memory_norm = F.normalize(memory_bank_features, dim=1)
        similarity = torch.mm(text_norm, memory_norm.T)  # (batch_size, bank_size)

        # 应用温度系数，计算 softmax 权重
        weights = F.softmax(similarity / temperature, dim=1)  # (batch_size, bank_size)

        # 使用权重聚合 Memory Bank 特征
        aggregated_features = torch.mm(weights, memory_bank_features)  # (batch_size, feature_dim)

        # 修正文本特征
        refined_features = alpha * aggregated_features + (1 - alpha) * text_features
        return refined_features
