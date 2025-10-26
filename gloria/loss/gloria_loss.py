"""
Adapted from: https://github.com/mrlibw/ControlGAN
"""

import torch
import torch.nn as nn

import ipdb
from torch.autograd import Variable


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):  ## convert img2text
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)




def attention_fn_T(query, context, temp1):  ## convert img2text
    """
    query: batch x ndf x ih x iw (sourceL=ihxiw)
    context: batch x ndf x queryL
    mask: batch_size x sourceL
    """
    batch_size, queryL = context.size(0), context.size(2)
    ih, iw = query.size(2), query.size(3)
    sourceL = ih * iw
    # --> batch x sourceL x ndf
    query = query.view(batch_size, -1, sourceL)


    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * sourceL, queryL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn


def global_loss(cnn_code, rnn_code, eps=1e-8, temp3=10.0):
    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1


def global_organ_loss(cnn_code, rnn_code, labels, eps=1e-8, temp3=10.0):
    batch_size = cnn_code.shape[0]
    
    # positive_mask 和 negative_mask 用于区分正负样本
    positive_mask = labels.bool()
    negative_mask = ~positive_mask

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # 计算嵌入的范数
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    # 计算相似性分数
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # 去掉额外的维度
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)

    # 创建单位对角矩阵 E 作为标签矩阵
    E = torch.eye(batch_size).to(cnn_code.device)

    # 初始化正样本和负样本的 loss 为 0
    total_loss0, total_loss1 = 0, 0

    # 检查是否存在正样本
    if positive_mask.sum() > 0:
        # 提取正样本和所有样本之间的分数，并计算对应的标签
        scores0_positive = scores0[positive_mask, :]
        scores1_positive = scores1[positive_mask, :]
        
        # 生成正样本的目标标签
        pos_target_labels = E[positive_mask, :]
        pos_labels = pos_target_labels.argmax(dim=1)

        # 计算正样本的 Info Loss
        loss0_positive = nn.CrossEntropyLoss()(scores0_positive, pos_labels)
        loss1_positive = nn.CrossEntropyLoss()(scores1_positive, pos_labels)
        total_loss0 += loss0_positive
        total_loss1 += loss1_positive

    # # 针对负样本计算，只考虑它们与正样本的对比
    # if negative_mask.sum() > 0 and positive_mask.sum() > 0:

    #     # 获取所有负样本
    #     scores0_neg = scores0[negative_mask, :]  # 所有负样本与所有样本的对比
    #     scores1_neg = scores1[negative_mask, :]

    #     # 取出所有 positive 样本和 negative 样本之间的对比，以及每个负样本的自身对比
    #     scores0_neg_pos = torch.cat([scores0_neg[:, negative_mask].diagonal().unsqueeze(1), scores0_neg[:, positive_mask]], dim=1)
    #     scores1_neg_pos = torch.cat([scores1_neg[:, negative_mask].diagonal().unsqueeze(1), scores1_neg[:, positive_mask]], dim=1)

    #     # 生成负样本的标签，全为 0 表示每个负样本的第一个位置为自身
    #     neg_target_labels = torch.zeros(scores0_neg_pos.shape[0], dtype=torch.long).to(cnn_code.device)

    #     # 计算负样本的 Info Loss
    #     loss0_negative = nn.CrossEntropyLoss()(scores0_neg_pos, neg_target_labels)
    #     loss1_negative = nn.CrossEntropyLoss()(scores1_neg_pos, neg_target_labels)
    #     total_loss0 += loss0_negative
    #     total_loss1 += loss1_negative

    #     total_loss0 /= 2
    #     total_loss1 /= 2

    return total_loss0, total_loss1

def global_loss_for_ca(cnn_code, rnn_code, eps=1e-8, temp3=10.0):
    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    cnn_code = cnn_code.reshape(batch_size * batch_size, 1, -1)
    rnn_code = rnn_code.reshape(batch_size * batch_size, 1, -1)

    cnn_code_norm = cnn_code_norm.reshape(batch_size * batch_size, 1, -1)
    rnn_code_norm = rnn_code_norm.reshape(batch_size * batch_size, 1, -1)


    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    scores0 = scores0.reshape(batch_size, batch_size)

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1


def local_loss(
    img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):

    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        weiContext, attn = attention_fn(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [48, 48]

    labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

    loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return loss0, loss1, att_maps


def attention_fn_3d(query, context, temp1):  ## convert img2text
    """
    query: batch_size x ndf x queryL
    context: batch_size x ndf x depth x height x width (sourceL = depth * height * width)
    """
    batch_size, queryL = query.size(0), query.size(2)
    depth, height, width = context.size(2), context.size(3), context.size(4)
    sourceL = depth * height * width

    # --> batch_size x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()  # --> batch_size x sourceL x ndf

    # Get attention
    # (batch_size x sourceL x ndf)(batch_size x ndf x queryL)
    # --> batch_size x sourceL x queryL
    attn = torch.bmm(contextT, query)  # --> batch_size x sourceL x queryL

    # Flatten and apply softmax
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)  # Apply softmax over queryL dimension

    # Reshape to batch_size x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)

    # Transpose and reshape to batch_size * queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    # Apply temperature scaling and softmax again
    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)

    # Reshape back to batch_size x queryL x sourceL
    attn = attn.view(batch_size, queryL, sourceL)

    # Transpose for batch_size x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch_size x ndf x sourceL)(batch_size x sourceL x queryL)
    # --> batch_size x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    # Reshape attn back to original 3D structure for visualization or further processing
    return weightedContext, attn.view(batch_size, queryL, depth, height, width)


def local_loss_3d(
    img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):
    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []

    # Loop through each text description in the batch
    for i in range(words_emb.shape[0]):

        # Get the number of words in the current caption
        words_num = cap_lens[i]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, embedding_dim, words_num]
        word = word.repeat(batch_size, 1, 1)  # Repeat for each image in the batch [batch_size, embedding_dim, words_num]
        
        # Context now refers to 3D image features
        context = img_features  # [batch_size, channels, depth, height, width]

        # Apply attention between the words and the 3D image features
        weiContext, attn = attention_fn_3d(
            word, context, temp1
        )  # attn will be of shape [batch_size, words_num, depth, height, width]

        # Save the attention map for the current sample
        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # [words_num, depth, height, width]

        # Transpose word embeddings and weiContext to match dimensions for cosine similarity calculation
        word = word.transpose(1, 2).contiguous()  # [batch_size, words_num, embedding_dim]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [batch_size, words_num, embedding_dim]

        # Flatten for cosine similarity
        word = word.view(batch_size * words_num, -1)  # [batch_size * words_num, embedding_dim]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [batch_size * words_num, embedding_dim]

        # Calculate cosine similarity
        row_sim = cosine_similarity(word, weiContext)  # [batch_size * words_num]
        row_sim = row_sim.view(batch_size, words_num)  # Reshape back to [batch_size, words_num]

        # Scale the similarities and aggregate (sum or mean)
        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # Sum over words dimension
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # Average over words dimension
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    # Concatenate all similarities across the batch
    similarities = torch.cat(similarities, 1)  # [batch_size, batch_size]
    similarities = similarities * temp3  # Scale the final similarities
    similarities1 = similarities.transpose(0, 1)  # Transpose for second cross-entropy loss

    # Generate labels for cross-entropy
    labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

    # Compute two cross-entropy losses (original and transposed)
    loss0 = nn.CrossEntropyLoss()(similarities, labels)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)

    return loss0, loss1, att_maps