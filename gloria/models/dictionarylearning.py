import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np



class SharedDictionaryLearning(nn.Module):
    def __init__(self, cfg):
        super(SharedDictionaryLearning, self).__init__()
        # 创建一个 nn.Embedding 作为共享的字典
        input_dim, dict_size, pretrained_features = cfg.model.dict.input_dim, cfg.model.dict.dict_size, cfg.model.dict.pretrained_features
        init_type = getattr(cfg.model.dict, "init_type", "orthogonal")  # 可选: "kmeans", "gaussian", "xavier", "uniform", "orthogonal"

        self.dictionary = nn.Embedding(dict_size, input_dim)
        
        if init_type == "kmeans" and pretrained_features is not None:
            self.initialize_with_kmeans(pretrained_features, dict_size, input_dim)
        elif init_type == "gaussian":
            nn.init.normal_(self.dictionary.weight, mean=0.0, std=1.0)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.dictionary.weight)
        elif init_type == "uniform":
            nn.init.uniform_(self.dictionary.weight, a=-0.1, b=0.1)
        else:  # 默认使用 orthogonal
            nn.init.orthogonal_(self.dictionary.weight)

        # 如果 `cfg.dict.freeze` 存在且为 True，则冻结字典权重；否则默认不冻结
        freeze_dictionary = getattr(cfg.model.dict, "freeze", False)
        if freeze_dictionary:
            self.dictionary.weight.requires_grad = False
            print("Dictionary no need requires grad")


    def initialize_with_kmeans(self, features_path, dict_size, input_dim):
        """
        使用 K-Means 聚类来初始化字典。
        
        参数:
        - features: 预训练特征, 形状为 [num_samples, input_dim]
        - dict_size: 字典的大小
        - input_dim: 特征的维度
        """
        # 使用 K-Means 聚类
        features = np.load(features_path)
        dictionary_init = torch.tensor(features, dtype=torch.float32)

        # 将聚类中心复制到字典的权重中
        self.dictionary.weight.data.copy_(dictionary_init)
        print("Load pretrained dictionary")


    # def normalize_dictionary(self):
    #     # 对字典基向量进行归一化
    #     with torch.no_grad():
    #         self.dictionary.weight.data = F.normalize(self.dictionary.weight.data, p=2, dim=-1)

    def forward(self, x, sparse=False, norm=False):
        # 每次前向传播时对字典进行归一化，确保其在单位球面上
        # self.normalize_dictionary()

        if norm:
            scale_factor = self.dictionary.weight.size(1) ** 0.5  # input_dim 的平方根
            z = torch.matmul(x, self.dictionary.weight.T) / scale_factor  # [batch_size, dict_size]
        else:
            # 计算稀疏编码系数
            z = torch.matmul(x, self.dictionary.weight.T)  # [batch_size, dict_size]
        
        # 稀疏编码，通过 softmax 将系数稀疏化，或使用其他稀疏化方法
        z = F.softmax(z, dim=-1)
        
        # 计算在共享典空间上的投影，并归一化
        x_reconstructed = torch.matmul(z, self.dictionary.weight)  # [batch_size, input_dim]
        
        if sparse:
            return x_reconstructed, z
        else:
            return x_reconstructed
    

