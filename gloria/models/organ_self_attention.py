import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from functools import partial

class SelfAttentionModule(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True, dropout=0.0, attn_drop=0.0, drop_path=0.0):
        super(SelfAttentionModule, self).__init__()
        
        # 初始化 cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # 使用 Block 作为 Self-Attention 模块
        self.block = Block(
            dim=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

    def forward(self, x):
        # 将 cls_token 添加到输入中
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.block(x)
        # Forward pass 通过 Block 处理输入
        return x[:, 0]

    def initialize_from_pretrained(self, pretrained_state_dict):
        """
        从预训练的权重中提取 cls_token 和 Block 权重，并加载到当前 Self-Attention 模块。
        """
        # 提取预训练权重中的 cls_token
        self.cls_token.data.copy_(pretrained_state_dict['cls_token'])
        
        # 提取预训练权重中的最后一个 Block 的参数
        block_prefix = 'blocks.11.'  # 假设最后一个 Block 的前缀是 blocks.11
        block_state_dict = {k[len(block_prefix):]: v for k, v in pretrained_state_dict.items() if k.startswith(block_prefix)}
        
        # 将权重加载到 self.block 中
        self.block.load_state_dict(block_state_dict)
        print("Self-Attention module initialized with pretrained weights and cls_token.")

# 加载预训练模型权重并初始化 Self-Attention 模块
def initialize_self_attention_from_pretrained(pretrained_model_path):
    # 加载预训练的权重文件
    pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu')
    
    # 创建自定义 SelfAttentionModule 并初始化
    hidden_size = pretrained_state_dict['cls_token'].shape[-1]
    num_heads = 12
    mlp_ratio = pretrained_state_dict['blocks.11.mlp.fc1.weight'].shape[0] / hidden_size
    
    self_attention_module = SelfAttentionModule(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        dropout=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    )
    
    # 使用预训练权重进行初始化
    self_attention_module.initialize_from_pretrained(pretrained_state_dict)
    
    return self_attention_module

# 使用示例
def make_self_attention(cfg):
    self_attention = initialize_self_attention_from_pretrained('/haoranlai/Project/M3D/VisionModel/vit_b16_3D_epoch_116.pth')
    return self_attention
