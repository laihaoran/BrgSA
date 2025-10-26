from functools import partial
import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block, Attention
import numpy as np
import torch.nn.functional as F

# 自定义 Attention 类，继承自原 Attention 类
class MaskAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            # Apply the mask, making attention weights -inf for masked positions
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(mask == 0, float('-inf'))
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn_weights = attn  # 保存注意力权重
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x  # 返回注意力权重

# 自定义 Block 类，继承自原 Block 类
class MaskBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 使用自定义的 Attention
        self.attn = MaskAttention(
            dim=self.attn.qkv.in_features,
            num_heads=self.attn.num_heads,
            qkv_bias=self.attn.qkv.bias is not None,
            qk_norm=kwargs.get('qk_norm', False),
            attn_drop=self.attn.attn_drop.p,
            proj_drop=self.attn.proj_drop.p,
            norm_layer=type(self.norm1),  # 获取原始的 norm_layer
        )

    def forward(self, x, mask=None):
        # 从自定义 Attention 获取注意力权重
        attn_output = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path1(self.ls1(attn_output))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x  # 返回注意力权重



def get_3d_sincos_pos_embed(embed_dim, grid_d, grid_h, grid_w, cls_token=False):
    """
    Create 3D sine-cosine positional embeddings ensuring exact embedding dimensions.
    grid_d, grid_h, grid_w: dimensions of the grid depth, height, and width.
    cls_token: whether to include a class token.
    """
    # Create a 3D grid for position encoding
    grid_z = np.linspace(0, 1, num=grid_d, dtype=np.float32)
    grid_y = np.linspace(0, 1, num=grid_h, dtype=np.float32)
    grid_x = np.linspace(0, 1, num=grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # Change to ij for consistency
    grid = np.stack(grid, axis=-1).reshape(-1, 3)  # Flatten grid

    # Calculate position encoding for each dimension
    pos_embed = []
    dim_per_axis = embed_dim // 3  # Divide the dimensions equally among the axes
    for i in range(3):
        pos = grid[:, i]
        omega = np.power(10000, -np.arange(0, dim_per_axis // 2) / (dim_per_axis // 2))
        sin_emb = np.sin(pos[:, None] * omega[None, :])
        cos_emb = np.cos(pos[:, None] * omega[None, :])
        pos_embed.append(np.concatenate([sin_emb, cos_emb], axis=1))

    pos_embed = np.concatenate(pos_embed, axis=1)

    # Ensure the positional embedding exactly matches the embedding dimension
    if pos_embed.shape[1] > embed_dim:
        pos_embed = pos_embed[:, :embed_dim]
    elif pos_embed.shape[1] < embed_dim:
        extra_dims = embed_dim - pos_embed.shape[1]
        extra_emb = np.zeros((pos_embed.shape[0], extra_dims))
        pos_embed = np.concatenate([pos_embed, extra_emb], axis=1)

    if cls_token:
        # Prepend a class token if required
        cls_embed = np.zeros((1, embed_dim))
        pos_embed = np.concatenate([cls_embed, pos_embed], axis=0)

    return pos_embed

class PatchEmbed3D(nn.Module):
    """Compute 3D patch embeddings for non-cubic volumes."""
    def __init__(self, img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Calculate the number of patches along each dimension
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        x = self.proj(x)  # (N, C, D', H', W')
        x = x.flatten(2)  # (N, C, D'*H'*W')
        x = x.transpose(1, 2)  # (N, D'*H'*W', C)
        return x


class ViT3D(nn.Module):
    """Vision Transformer (ViT) with 3D support."""
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
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_dim // hidden_size, qkv_bias=qkv_bias,  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.initialize_weights()

        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.img_size[0] // self.patch_embed.patch_size[0],
                                            self.patch_embed.img_size[1] // self.patch_embed.patch_size[1],
                                            self.patch_embed.img_size[2] // self.patch_embed.patch_size[2], cls_token=True)
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

    def TokenizerImage(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return x
    
    def self_attention_token(self, x):
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        x = self.norm(x)
        # return x, hidden_states_out
        return x[:, 0], x[:, 1:]

    def forward(self, x, select_layer=-1):
    
        x = self.TokenizerImage(x)

        if select_layer == -1:
            hidden_states_out = []
            for blk in self.blocks:
                x = blk(x)
                hidden_states_out.append(x)

            x = self.norm(x)
            # return x, hidden_states_out
            return x[:, 0], x[:, 1:]


# 自定义 ViT3D 类，继承自原 ViT3D 类
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
        **kwargs  # 捕获任何其他额外的参数
    ):
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=classification,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            **kwargs  # 传递额外的参数给父类
        )

        # 使用自定义的 Block 类
        self.blocks = nn.ModuleList([
            MaskBlock(
                hidden_size, num_heads, mlp_dim // hidden_size, qkv_bias=qkv_bias,  norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ) for _ in range(self.blocks.__len__())
        ])

    def forward(self, x, mask=None):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x, mask=mask)  # 获取每一层的注意力权重

        x = self.norm(x)
        if self.classification:
            x = self.classification_head(x[:, 0])

        return x[:, 0], x[:, 1:]  # 返回注意力权重



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


# 加载权重函数
def load_weight(model, path):
    pretrain_dict = torch.load(path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()

    # 过滤掉尺寸不匹配的参数
    filtered_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    mismatched_keys = {k: (v.size(), model_dict[k].size()) for k, v in pretrain_dict.items() if k in model_dict and v.size() != model_dict[k].size()}
    
    # 更新model_dict
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    
    # 打印不匹配的权重
    if mismatched_keys:
        print("Mismatched keys:")
        for key, sizes in mismatched_keys.items():
            print(f"{key}: checkpoint shape {sizes[0]}, model shape {sizes[1]}")

    return model



def make_3d_model(pretrained=True):
    # 定义模型配置
    config = {
        'in_channels': 1,
        'img_size': (224, 224, 112),
        'patch_size': (16, 16, 8),
        'hidden_size': 768,  # 1536
        'mlp_dim': 3072,   # 6144
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

    if pretrained:
        # write new weight
        # model = load_weight(model, '/haoranlai/Project/M3D/VisionModel/vit_b16_3D_embedding1536_epoch_434.pth')
        model = load_weight(model, '/haoranlai/Project/M3D/VisionModel/vit_b16_3D_epoch_116.pth')
        print("image pretrained model loaded")

    return model, config['in_channels'], config['hidden_size']


def make_3d_mask_model(pretrained=True):
    # 定义模型配置
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
        # write new weight
        model = load_weight(model, '/haoranlai/Project/M3D/VisionModel/vit_b16_3D_embedding1536_epoch_434.pth')
        print("image pretrained model loaded")

    return model, config['in_channels'], config['hidden_size']




