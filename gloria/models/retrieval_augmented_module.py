import torch
import torch.nn as nn
import numpy as np
from .transformer import TransformerDecoderLayer,  TransformerDecoder, TransformerTrippleDecoderLayer, TransformerTrippleDecoder
# import faiss
import ipdb


# class CrossAttentionBlock(nn.Module):
#     def __init__(self, d_model, num_heads, num_layer):
#         super(CrossAttentionBlock, self).__init__()
#         self.attentions = nn.ModuleList([
#             nn.MultiheadAttention(d_model, num_heads) for _ in range(num_layer)
#         ])
#         self.linear = nn.Linear(d_model, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)
        
#     def forward(self, q, k, v):
#         # ipdb.set_trace()
#         x = q
#         for attention in self.attentions:
#             # Apply cross attention
#             x, _ = attention(query=x, key=k, value=v)
        
#         # Residual connection and layer normalization
#         x = self.layer_norm(x + q)
        
#         # Feed-forward layer
#         x = self.linear(x)
        
#         # Residual connection and layer normalization
#         x = self.layer_norm(x + q)
        
#         return x    



# class CrossAttentionBlock(nn.Module):
#     def __init__(self, d_model, num_heads, num_layer):
#         super(CrossAttentionBlock, self).__init__()
#         self.attentions = nn.ModuleList([
#             nn.MultiheadAttention(d_model, num_heads) for _ in range(num_layer)
#         ])

#         self.feedforward = nn.Linear(d_model, d_model)
#         self.layerNorm1 = nn.LayerNorm(d_model)
#         self.layerNorm2 = nn.LayerNorm(d_model)

        
#     def forward(self, q, k, v):

#         x = q
#         for attention in self.attentions:
#             # Apply cross attention
#             x, _ = attention(query=x, key=k, value=v)
        
#         # Residual connection and layer normalization
#         x = self.layerNorm1(x + q)
        
#         # Feed-forward layer
#         x = self.feedforward(x)
        
#         # Residual connection and layer normalization
#         x = self.layerNorm2(x + q)
        
#         return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_layer):
        super(CrossAttentionBlock, self).__init__()
        decoder_layer = TransformerTrippleDecoderLayer(d_model, num_heads, 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerTrippleDecoder(decoder_layer, num_layer , decoder_norm,
                                  return_intermediate=False)
        
    def forward(self, q, k, v):

        features,ws = self.decoder(q, k, v,
            memory_key_padding_mask=None, pos=None, query_pos=None)
        
        return features


# q = torch.ones(4, 16, 768).cuda()
# k = torch.ones(4, 16, 768).cuda()
# v = torch.ones(4, 16, 768).cuda()
# net = CrossAttentionBlock(768, 4, 6).cuda()

# net(q,k,v)


class cross_attention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.cross_attention_img_blocks = CrossAttentionBlock(embed_dim, num_heads, num_layers)
        self.cross_attention_text_blocks = CrossAttentionBlock(embed_dim, num_heads, num_layers)


    def forward(self, features, img_features, text_features):
     
        features = features.unsqueeze(0)
        img_features = img_features.permute(1, 0, 2)
        text_features = text_features.permute(1, 0, 2)

        ram_img = self.cross_attention_img_blocks(features, img_features, text_features)
        
        ram_text = self.cross_attention_text_blocks(features, text_features, img_features)
        out = features + ram_img + ram_text
        return out.squeeze(0)

# # 设置PyTorch设备为GPU\
# import torch.distributed as dist
# dist.init_process_group(backend='nccl') 
# device = torch.device('cuda')
# torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())  # 设置当前进程的GPU设备


class retrieval_augmentend(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.embedd_dim = cfg.model.ram.embed_dim
        self.image_dict_vectors = np.load(cfg.model.ram.images_dict_vectors_path)
        self.text_dict_vectors = np.load(cfg.model.ram.text_dict_vectors_path)

        # # 构建FAISS索引 for image
        # index = faiss.IndexFlatL2(self.embedd_dim)  # 使用欧氏距离作为相似性度量
        # self.img_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)  # 将索引移动到 GPU
        # self.img_index.add(self.image_dict_vectors)  # 将字典向量添加到索引中

        # chose top k 
        self.K = cfg.model.ram.K

        # cross attention
        self.crossattention = cross_attention(cfg.model.ram.embed_dim, cfg.model.ram.num_heads, cfg.model.ram.num_layer)

    # def obation_faiss(self, device):
    #     # 构建FAISS索引 for image
    #     index = faiss.IndexFlatL2(self.embedd_dim)  # 使用欧氏距离作为相似性度量
    #     img_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), device.index, index)  # 将索引移动到 GPU
    #     img_index.add(self.image_dict_vectors)  # 将字典向量添加到索引中
    #     return img_index

    def forward(self, img_indices, featrues):

        device = featrues.device 
        # ipdb.set_trace()
        ## 实时生成查询器
        # img_index = self.obation_faiss(device)

        # 进行批量查询
        # _, img_indices = self.img_index.search(query_vectors.cpu().numpy(), self.K)
        img_result_vectors = self.image_dict_vectors[img_indices.flatten()].reshape(-1, self.K, self.image_dict_vectors.shape[1])
        img_result_vectors = torch.from_numpy(img_result_vectors).to(device)

        text_result_vectors = self.text_dict_vectors[img_indices.flatten()].reshape(-1, self.K, self.text_dict_vectors.shape[1])
        text_result_vectors = torch.from_numpy(text_result_vectors).to(device)
        # featrues.unsqueeze(1).repeat(1, self.K, 1)
        out = self.crossattention(featrues, img_result_vectors, text_result_vectors)
        return out

