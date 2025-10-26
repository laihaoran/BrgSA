from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block, Attention
import numpy as np

# =========================
#  Attention / Block (自定义)
# =========================

class MaskAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            # mask: (B, N) or (B, 1, 1, N) 都兼容处理为 (B, num_heads, N, N)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
            mask = mask.expand(-1, self.num_heads, N, -1)  # (B,heads,N,N)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(mask == 0, float('-inf'))
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MaskBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = MaskAttention(
            dim=self.attn.qkv.in_features,
            num_heads=self.attn.num_heads,
            qkv_bias=self.attn.qkv.bias is not None,
            qk_norm=kwargs.get('qk_norm', False),
            attn_drop=self.attn.attn_drop.p,
            proj_drop=self.attn.proj_drop.p,
            norm_layer=type(self.norm1),
        )

    def forward(self, x, mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

# class CustomAttention(Attention):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)

#         if self.fused_attn:
#             x_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
#             attn_weights = None
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x_out = attn @ v
#             attn_weights = attn

#         x_out = x_out.transpose(1, 2).reshape(B, N, C)
#         x_out = self.proj(x_out)
#         x_out = self.proj_drop(x_out)
#         return x_out, attn_weights

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Attention

class CustomAttention(Attention):
    """
    与 timm 的 Attention 兼容的子类：
    - 在非 fused 路径下保存 softmax 后的注意力到 self.last_attn （形状 [B, H, T, T]）
    - 调用了 retain_grad()，方便在一次 backward 后从 self.last_attn.grad 读取 dS/dA
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     # fused 路径通常无法直接拿到注意力权重
        #     x_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        #     attn_weights = None
        # else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)           # [B, H, T, T]
        attn = attn.softmax(dim=-1)              # softmax 后的注意力
        # 保存 softmax 后注意力，并允许后向获取梯度
        self.last_attn = attn
        self.last_attn.retain_grad()

        attn = self.attn_drop(attn)
        x_out = attn @ v                          # [B, H, T, D]
        attn_weights = attn

        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out, attn_weights


class CustomBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = CustomAttention(
            dim=self.attn.qkv.in_features,
            num_heads=self.attn.num_heads,
            qkv_bias=self.attn.qkv.bias is not None,
            qk_norm=kwargs.get('qk_norm', False),
            attn_drop=self.attn.attn_drop.p,
            proj_drop=self.attn.proj_drop.p,
            norm_layer=type(self.norm1),
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, attn_weights

# =========================
#  Utils: 3D sin-cos PE
# =========================

def get_3d_sincos_pos_embed(embed_dim, grid_d, grid_h, grid_w, cls_token=False):
    """
    生成 3D 正余弦位置编码，返回形状：(grid_d*grid_h*grid_w (+1), embed_dim)
    """
    grid_z = np.linspace(0, 1, num=grid_d, dtype=np.float32)
    grid_y = np.linspace(0, 1, num=grid_h, dtype=np.float32)
    grid_x = np.linspace(0, 1, num=grid_w, dtype=np.float32)
    # 'ij' 对应 (x,y,z) = (w,h,d)，这里统一用 (x,y,z)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid = np.stack(grid, axis=-1).reshape(-1, 3)

    pos_embed = []
    dim_per_axis = embed_dim // 3
    for i in range(3):
        pos = grid[:, i]
        omega = np.power(10000, -np.arange(0, dim_per_axis // 2) / (dim_per_axis // 2))
        sin_emb = np.sin(pos[:, None] * omega[None, :])
        cos_emb = np.cos(pos[:, None] * omega[None, :])
        pos_embed.append(np.concatenate([sin_emb, cos_emb], axis=1))

    pos_embed = np.concatenate(pos_embed, axis=1)

    if pos_embed.shape[1] > embed_dim:
        pos_embed = pos_embed[:, :embed_dim]
    elif pos_embed.shape[1] < embed_dim:
        extra_dims = embed_dim - pos_embed.shape[1]
        pos_embed = np.concatenate([pos_embed, np.zeros((pos_embed.shape[0], extra_dims))], axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim)), pos_embed], axis=0)

    return pos_embed

# =========================
#  3D PatchEmbed
# =========================

class PatchEmbed3D(nn.Module):
    """Compute 3D patch embeddings for non-cubic volumes."""
    def __init__(self, img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 运行时缓存当前网格尺寸 (D',H',W')，forward 时更新
        self.grid_size = (img_size[2] // patch_size[2],
                          img_size[1] // patch_size[1],
                          img_size[0] // patch_size[0])

    def forward(self, x):
        x = self.proj(x)  # (B, C, D', H', W')
        B, C, Dp, Hp, Wp = x.shape
        self.grid_size = (Dp, Hp, Wp)  # 更新当前网格
        x = x.flatten(2).transpose(1, 2)  # (B, D'H'W', C)
        return x

# =========================
#  ViT3D (带动态PE插值)
# =========================

class ViT3D(nn.Module):
    """Vision Transformer (ViT) with 3D support and dynamic abs-PE resizing."""
    def __init__(
        self,
        in_channels: int,
        img_size: tuple,
        patch_size: tuple,
        hidden_size: int = 1536,
        mlp_dim: int = 6144,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = True,
        save_attn: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_channels, hidden_size)

        # 基准网格（由构造时 img_size / patch_size 决定），保证与预训练权重对齐
        D0 = self.patch_embed.img_size[2] // self.patch_embed.patch_size[2]
        H0 = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        W0 = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
        num_patches = D0 * H0 * W0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_dim // hidden_size, qkv_bias=qkv_bias,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.initialize_weights()

        if self.classification:
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)

    def initialize_weights(self):
        D0 = self.patch_embed.img_size[2] // self.patch_embed.patch_size[2]
        H0 = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        W0 = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], D0, H0, W0, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def _interpolate_pos_encoding(self, grid_size):
        """
        将“基准 pos_embed”（构造时 img_size 对齐）插值到当前 grid_size=(D',H',W')。
        返回 (1, 1 + D'H'W', C)
        """
        Dp, Hp, Wp = grid_size
        Ntgt = Dp * Hp * Wp

        cls_pos = self.pos_embed[:, :1, :]     # (1,1,C)
        patch_pos = self.pos_embed[:, 1:, :]   # (1,N0,C)

        # 基准网格
        D0 = self.patch_embed.img_size[2] // self.patch_embed.patch_size[2]
        H0 = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        W0 = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
        N0 = D0 * H0 * W0
        assert patch_pos.shape[1] == N0, "pos_embed 必须与基准网格一致，才能进行运行时插值"

        C = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, D0, H0, W0, C).permute(0, 4, 1, 2, 3)  # (1,C,D0,H0,W0)
        patch_pos = F.interpolate(patch_pos, size=(Dp, Hp, Wp), mode='trilinear', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 4, 1).reshape(1, Ntgt, C)        # (1,Ntgt,C)

        return torch.cat([cls_pos, patch_pos], dim=1)

    # def TokenizerImage(self, x):
    #     x = self.patch_embed(x)  # (B, N, C)。内部会更新 self.patch_embed.grid_size=(D',H',W')
    #     pos_embed_cur = self._interpolate_pos_encoding(self.patch_embed.grid_size)
    #     cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((cls_token, x), dim=1)
    #     x = x + pos_embed_cur
    #     return x
    
    def TokenizerImage(self, x):
        x = self.patch_embed(x)  # (B, N, C)。内部会更新 self.patch_embed.grid_size=(D',H',W')
        # 记录这次前向对应的网格，外部需要它把 token map 回 3D
        self.last_grid_size = self.patch_embed.grid_size
        pos_embed_cur = self._interpolate_pos_encoding(self.patch_embed.grid_size)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + pos_embed_cur
        return x


    def self_attention_token(self, x):
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward(self, x, select_layer=-1):
        
        x = self.TokenizerImage(x)
        if select_layer == -1:
            hidden_states_out = []
            for blk in self.blocks:
                x = blk(x)
                hidden_states_out.append(x)
            x = self.norm(x)
            return x[:, 0], x[:, 1:]
        else:
            hidden_states_out = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i == select_layer:
                    hidden_states_out.append(x)
            x = self.norm(x)
            return x[:, 0], x[:, 1:], hidden_states_out[0]

# =========================
#  Custom / Mask ViT3D
# =========================

# class CustomViT3D(ViT3D):
#     def __init__(
#         self,
#         in_channels: int,
#         img_size: tuple,
#         patch_size: tuple,
#         hidden_size: int = 1536,
#         mlp_dim: int = 6144,
#         num_layers: int = 12,
#         num_heads: int = 12,
#         pos_embed: str = "conv",
#         classification: bool = False,
#         num_classes: int = 2,
#         dropout_rate: float = 0.0,
#         spatial_dims: int = 3,
#         post_activation: str = "Tanh",
#         qkv_bias: bool = True,
#         save_attn: bool = False,
#         **kwargs
#     ):
#         super().__init__(in_channels, img_size, patch_size, hidden_size, mlp_dim, num_layers, num_heads,
#                          pos_embed, classification, num_classes, dropout_rate, spatial_dims,
#                          post_activation, qkv_bias, save_attn)

#         # 重置 blocks 为自定义 block
#         self.blocks = nn.ModuleList([
#             CustomBlock(hidden_size, num_heads, mlp_dim // hidden_size, qkv_bias=qkv_bias,
#                         norm_layer=partial(nn.LayerNorm, eps=1e-6))
#             for _ in range(num_layers)
#         ])

#     def forward(self, x, get_local=True):
#         x = super().TokenizerImage(x)  # 使用父类的动态PE
#         last_attn_weights = None
#         for blk in self.blocks:
#             x, attn_weights = blk(x)
#             last_attn_weights = attn_weights
#         if last_attn_weights is not None:
#             last_attn_weights = last_attn_weights[:, :, 1:, 1:].mean(dim=1)
#         x = self.norm(x)
#         if self.classification:
#             x = self.classification_head(x[:, 0])
#         return x[:, 0], x[:, 1:], last_attn_weights


import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
from typing import List, Optional, Tuple

# 假设你已在同文件里定义了：
# - ViT3D 基类（含 TokenizerImage / _interpolate_pos_encoding 等）
# - CustomBlock（其内部使用了上面的 CustomAttention）

class CustomViT3D(ViT3D):
    def __init__(
        self,
        in_channels: int,
        img_size: tuple,
        patch_size: tuple,
        hidden_size: int = 1536,
        mlp_dim: int = 6144,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation: str = "Tanh",
        qkv_bias: bool = True,
        save_attn: bool = False,
        **kwargs
    ):
        super().__init__(in_channels, img_size, patch_size, hidden_size, mlp_dim, num_layers, num_heads,
                         pos_embed, classification, num_classes, dropout_rate, spatial_dims,
                         post_activation, qkv_bias, save_attn)

        # 用自定义 Block（内部是上面定义的 CustomAttention）
        self.blocks = nn.ModuleList([
            CustomBlock(hidden_size, num_heads, mlp_dim // hidden_size, qkv_bias=qkv_bias,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(num_layers)
        ])

    def forward(self, x, get_local: bool = True, return_all_attn: bool = False):
        """
        返回：
          - 常规： (cls, patches, last_attn_weights)
          - 若 return_all_attn=True： (cls, patches, last_attn_weights, all_attn)
              其中 all_attn 是长度为 L 的列表，每层元素形状为 [B, H, T, T]（softmax 后）
        另外会把当前 token 网格记录到 self.last_grid_size = (Dz, Hy, Wx)
        """
        # 使用父类的 tokenizer（含动态 3D 位置编码插值）
        x = super().TokenizerImage(x)

        last_attn_weights = None
        all_attn: List[torch.Tensor] = []

        for blk in self.blocks:
            x, attn_weights = blk(x)
            last_attn_weights = attn_weights  # 这通常是“最后一层”的 attn（可用于 quick 可视化）

            if return_all_attn and hasattr(blk.attn, "last_attn") and blk.attn.last_attn is not None:
                # 收集每层 softmax 后的注意力权重 [B, H, T, T]
                all_attn.append(blk.attn.last_attn)

        # last_attn_weights: [B, H, T, T] -> 平均 head，且去掉 CLS 行/列，得到 patch×patch
        if last_attn_weights is not None:
            last_attn_weights = last_attn_weights[:, :, 1:, 1:].mean(dim=1)  # [B, P, P]

        x = self.norm(x)

        # 记录本次前向对应的 patch 网格，供外部映射回 (Dz,Hy,Wx)
        # 注意：TokenizerImage 调用了 PatchEmbed3D.forward，会在其中更新 grid_size
        self.last_grid_size = self.patch_embed.grid_size  # (Dz, Hy, Wx)

        if self.classification:
            x = self.classification_head(x[:, 0])

        cls_token = x[:, 0]   # [B, C]
        patches  = x[:, 1:]   # [B, P, C]


        if return_all_attn:
            return cls_token, patches, last_attn_weights, all_attn
        else:
            return cls_token, patches, last_attn_weights




class MaskViT3D(ViT3D):
    def __init__(
        self,
        in_channels: int,
        img_size: tuple,
        patch_size: tuple,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation: str = "Tanh",
        qkv_bias: bool = True,
        save_attn: bool = False,
        **kwargs
    ):
        super().__init__(in_channels, img_size, patch_size, hidden_size, mlp_dim, num_layers, num_heads,
                         pos_embed, classification, num_classes, dropout_rate, spatial_dims,
                         post_activation, qkv_bias, save_attn,)

        self.blocks = nn.ModuleList([
            MaskBlock(hidden_size, num_heads, mlp_dim // hidden_size, qkv_bias=qkv_bias,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(num_layers)
        ])

    def forward(self, x, select_layer=-1, mask=None):
        x = super().TokenizerImage(x)  # 使用父类的动态PE
        if select_layer == -1:
            hidden_states_out = []
            for blk in self.blocks:
                x = blk(x, mask=mask)
                hidden_states_out.append(x)
            x = self.norm(x)
            return x[:, 0], x[:, 1:]
        else:
            hidden_states_out = []
            for i, blk in enumerate(self.blocks):
                x = blk(x, mask=mask)
                if i == select_layer:
                    hidden_states_out.append(x)
            x = self.norm(x)
            return x[:, 0], x[:, 1:], hidden_states_out[0]

# =========================
#  ViT3DMAETower (保持不变)
# =========================

class ViT3DMAETower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = ViT3D(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="conv",
            spatial_dims=len(self.config.patch_size),
            classification=False
        )

    def forward(self, images):
        last_feature, hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

# =========================
#  权重加载（含PE到基准网格的插值）
# =========================

def _infer_3d_grid(n_tokens: int):
    # 简单因子分解找 (D,H,W)
    c = round(n_tokens ** (1/3))
    if c * c * c == n_tokens:
        return (c, c, c)
    for d in range(1, int(round(n_tokens ** (1/3))) + 3):
        if n_tokens % d: 
            continue
        rem = n_tokens // d
        for h in range(1, int(round(rem ** 0.5)) + 3):
            if rem % h: 
                continue
            w = rem // h
            return (d, h, w)
    return (c, c, max(1, n_tokens // (c*c)))

def _resize_pos_embed_to_base(ckpt_pos_embed: torch.Tensor, target_grid):
    # ckpt_pos_embed: (1, Nsrc+1, C) -> (1, Ntgt+1, C)
    cls = ckpt_pos_embed[:, :1, :]
    pos = ckpt_pos_embed[:, 1:, :]
    Nsrc, C = pos.shape[1], pos.shape[2]
    D0, H0, W0 = _infer_3d_grid(Nsrc)

    pos = pos.reshape(1, D0, H0, W0, C).permute(0, 4, 1, 2, 3)  # (1,C,D0,H0,W0)
    Dt, Ht, Wt = target_grid
    pos = F.interpolate(pos, size=(Dt, Ht, Wt), mode='trilinear', align_corners=False)
    pos = pos.permute(0, 2, 3, 4, 1).reshape(1, Dt*Ht*Wt, C)
    return torch.cat([cls, pos], dim=1)

def load_weight(model, path):
    pretrain_dict = torch.load(path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()

    # 先处理 pos_embed：把 ckpt 的 pos_embed 插到“基准网格”（构造时 img_size/patch_size）
    if 'pos_embed' in pretrain_dict and 'pos_embed' in model_dict:
        pe_ckpt = pretrain_dict['pos_embed']
        pe_model = model_dict['pos_embed']
        if pe_ckpt.shape != pe_model.shape:
            Dt = model.patch_embed.img_size[2] // model.patch_embed.patch_size[2]
            Ht = model.patch_embed.img_size[1] // model.patch_embed.patch_size[1]
            Wt = model.patch_embed.img_size[0] // model.patch_embed.patch_size[0]
            pretrain_dict['pos_embed'] = _resize_pos_embed_to_base(pe_ckpt, (Dt, Ht, Wt))

    # 正常过滤加载
    filtered_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    mismatched_keys = {k: (v.size(), model_dict[k].size()) for k, v in pretrain_dict.items()
                       if k in model_dict and v.size() != model_dict[k].size()}

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)

    if mismatched_keys:
        print("Mismatched keys:")
        for key, sizes in mismatched_keys.items():
            print(f"{key}: checkpoint shape {sizes[0]}, model shape {sizes[1]}")
    return model

# =========================
#  工厂函数
# =========================

def make_3d_model(pretrained=True):
    config = {
        'in_channels': 1,
        'img_size': (224, 224, 112),   # 基准尺寸（与权重对齐）
        'patch_size': (16, 16, 8),
        'hidden_size': 768,
        'mlp_dim': 3072,
        'num_layers': 12,
        'num_heads': 12,
        'pos_embed': 'conv',
        'classification': False,
        'num_classes': 2,
        'dropout_rate': 0.0,
        'spatial_dims': 3,
        'post_activation': 'Tanh',
        'qkv_bias': True,
        'save_attn': False,
    }
    model = ViT3D(**config)
    # model = CustomViT3D(**config)
    if pretrained:
        model = load_weight(model, '/data2/haoranlai/Project/gloria/Pretrain_model/vit_b16_3D_m3ae_epoch_2.pth')
        print("image pretrained model loaded")
    return model, config['in_channels'], config['hidden_size']

def make_3d_atten_model(pretrained=True):
    config = {
        'in_channels': 1,
        'img_size': (224, 224, 112),   # 基准尺寸（与权重对齐）
        'patch_size': (16, 16, 8),
        'hidden_size': 768,
        'mlp_dim': 3072,
        'num_layers': 12,
        'num_heads': 12,
        'pos_embed': 'conv',
        'classification': False,
        'num_classes': 2,
        'dropout_rate': 0.0,
        'spatial_dims': 3,
        'post_activation': 'Tanh',
        'qkv_bias': True,
        'save_attn': False,
    }
    model = CustomViT3D(**config)
    # if pretrained:
    #     model = load_weight(model, '/haoranlai/Project/M3D/VisionModel/vit_b16_3D_epoch_116.pth')
    #     print("image pretrained model loaded")
    return model, config['in_channels'], config['hidden_size']

def make_3d_mask_model(pretrained=True):
    config = {
        'in_channels': 1,
        'img_size': (224, 224, 112),
        'patch_size': (16, 16, 8),
        'hidden_size': 768,
        'mlp_dim': 3072,
        'num_layers': 12,
        'num_heads': 12,
        'pos_embed': 'conv',
        'classification': False,
        'num_classes': 2,
        'dropout_rate': 0.0,
        'spatial_dims': 3,
        'post_activation': 'Tanh',
        'qkv_bias': True,
        'save_attn': False,
    }
    model = MaskViT3D(**config)
    if pretrained:
        model = load_weight(model, '/haoranlai/Project/M3D/VisionModel/vit_b16_3D_epoch_116.pth')
        print("image pretrained model loaded")
    return model, config['in_channels'], config['hidden_size']
