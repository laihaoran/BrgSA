import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def memory_diversity_loss(memory_bank):
    """
    计算记忆正则化损失，防止记忆单元冗余或过于相似。
    
    Args:
        memory_bank (torch.Tensor): 记忆单元，形状为 (N, D)，
                                    N 是记忆单元个数，D 是每个单元的特征维度。
    
    Returns:
        torch.Tensor: 记忆正则化损失值，标量。
    """
    # 归一化每个记忆单元以计算余弦相似度
    memory_bank_normalized = F.normalize(memory_bank, p=2, dim=1)  # (N, D)

    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(memory_bank_normalized, memory_bank_normalized.T)  # (N, N)

    # 排除对角线元素（即自相似），只考虑不同单元间的相似度
    # 创建对角矩阵
    eye_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    non_diagonal_similarity = similarity_matrix[~eye_mask.bool()].view(similarity_matrix.size(0), -1)

    # 平方和作为正则化损失
    diversity_loss = (non_diagonal_similarity ** 2).mean()
    
    return diversity_loss


def memory_diversity_info_loss(memory_bank, temperature=0.1):
    """
    使用 CrossEntropyLoss 优化记忆单元的分布，强化正样本，弱化负样本。

    Args:
        memory_bank (torch.Tensor): 记忆单元，形状为 (N, D)。
        temperature (float): 温度参数，用于平滑 logits。

    Returns:
        torch.Tensor: CrossEntropy 损失值，标量。
    """
    # 归一化每个记忆单元以计算余弦相似性
    memory_bank_normalized = F.normalize(memory_bank, p=2, dim=1)  # (N, D)

    # 计算相似性矩阵 (logits)
    similarity_matrix = torch.matmul(memory_bank_normalized, memory_bank_normalized.T)  # (N, N)

    # 调整相似性矩阵的尺度
    logits = similarity_matrix / temperature  # (N, N)

    # 构造目标标签，每个样本的正样本为自身
    targets = torch.arange(logits.size(0), device=logits.device)  # (N,)

    # 使用 CrossEntropyLoss
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)

    return loss