import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
import ipdb
import numpy as np
from einops import rearrange

from collections.abc import Sequence
from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

class UnetrPrUpBlock(nn.Module):
    # Initialization and other methods remain the same

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, num_layer: int,
                 kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int],
                 upsample_kernel_size: Union[Sequence[int], int], norm_name: Union[tuple, str], conv_block: bool = False,
                 res_block: bool = False, skip_upsample_in_blocks: bool = False) -> None:
        """
        skip_upsample_in_blocks: Flag to skip upsampling in blocks (used to prevent double upsampling)
        """
        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride,
            conv_only=True, is_transposed=True,
        )
        
        self.skip_upsample_in_blocks = skip_upsample_in_blocks  # Store the flag

        if conv_block:
            # Add the condition to avoid upsampling in blocks
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            UnetResBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ) if self.skip_upsample_in_blocks else
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            )
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            UnetBasicBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ) if self.skip_upsample_in_blocks else
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            )
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class MRMWithSwin3D(SwinTransformer):
    """
    Masked Autoencoder with 3D Swin Transformer backbone for medical images.
    This class inherits from SwinTransformer and adds the masking and decoder generation functionality.
    """
    def __init__(self, patch_size=(16, 16, 8), in_chans=1,
                 embed_dim=768, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, **kwargs):
        super(MRMWithSwin3D, self).__init__(
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=(8, 8, 8),  # Use 3D window size
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            spatial_dims=3,  # Ensure 3D processing
            **kwargs
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm_pix_loss = norm_pix_loss


        # feature_size = embed_dim
        # spatial_dims = 3
        # norm_name = "instance"

        # self.decoder5 = UnetrPrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=16 * feature_size,
        #     out_channels=8 * feature_size,
        #     num_layer=1,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     stride=1,
        #     norm_name=norm_name,
        #     conv_block=True,
        #     res_block=True,
        #     skip_upsample_in_blocks=True
        # )

        # self.decoder4 = UnetrPrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size * 8,
        #     out_channels=feature_size * 4,
        #     num_layer=1,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     stride=1,
        #     norm_name=norm_name,
        #     conv_block=True,
        #     res_block=True,
        #     skip_upsample_in_blocks=True
        # )

        # self.decoder3 = UnetrPrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size * 4,
        #     out_channels=feature_size * 2,
        #     num_layer=1,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     stride=1,
        #     norm_name=norm_name,
        #     conv_block=True,
        #     res_block=True,
        #     skip_upsample_in_blocks=True
        # )
        # self.decoder2 = UnetrPrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size * 2,
        #     out_channels=feature_size,
        #     num_layer=1,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     stride=2,
        #     norm_name=norm_name,
        #     conv_block=True,
        #     res_block=True,
        #     skip_upsample_in_blocks=True
        # )

        # self.decoder1 = UnetrPrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size,
        #     out_channels=feature_size,
        #     num_layer=1,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     stride=1,
        #     norm_name=norm_name,
        #     conv_block=True,
        #     res_block=True,
        #     skip_upsample_in_blocks=True
        # )

        # # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=1)

        # self.norm_up = norm_layer(embed_dim)
        # self.decoder_pred = nn.Linear(embed_dim, patch_size[0]*patch_size[1]*patch_size[2] * in_chans, bias=True)




        # self.initialize_weights()

    # def initialize_weights(self):
    #     # Initialize weights for positional embedding and decoder
    #     grid_d, grid_h, grid_w = (self.img_size[0] // self.patch_size[0],
    #                               self.img_size[1] // self.patch_size[1],
    #                               self.img_size[2] // self.patch_size[2])
    #     pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], grid_d, grid_h, grid_w, cls_token=True)
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    #     torch.nn.init.normal_(self.cls_token, std=.02)
    #     self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform random masking by selecting a portion of the input sequence to be masked.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # Noise to shuffle the samples
        ids_shuffle = torch.argsort(noise, dim=1)  # Shuffle the samples
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Create a mask where 1 is removed and 0 is kept
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def window_masking(self, x: torch.Tensor, mask_ratio: float, remove: bool = False):
        """
        3D Masking on embedded patches
        x: [B, C, D, H, W], the input image tensor after patch embedding
        remove: whether to remove the mask patch or just replace with mask token
        """
        B, C, D, H, W = x.shape  # B: batch size, C: channels, D: depth, H: height, W: width
        
        # Flatten the spatial dimensions (D, H, W) into a single dimension (patches)
        L = D * H * W  # Total number of patches (D * H * W)
        
        # Generate random noise for shuffling and masking patches
        noise = torch.rand(B, L, device=x.device)  # noise for shuffling
        sparse_shuffle = torch.argsort(noise, dim=1)  # Sort the noise to decide which patches to mask
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        
        # Keep patches (non-masked patches) based on the mask ratio
        num_keep = int(L * (1 - mask_ratio))  # Number of patches to keep
        sparse_keep = sparse_shuffle[:, :num_keep]
        
        # Create mask (0 is keep, 1 is remove)
        mask = torch.ones([B, L], device=x.device)
        mask[:, :num_keep] = 0  # Set the mask for kept patches to 0
        mask = torch.gather(mask, dim=1, index=sparse_restore)
        
        # Apply the mask
        x_masked = torch.clone(x)  # Clone the original tensor to modify
        if not remove:
            # For each batch, replace the masked patches with mask_token
            # Reshape x to [B, L, C] to facilitate indexing by patch
            x_reshaped = rearrange(x, 'b c d h w -> b (d h w) c')  # Reshape to [B, L, C]
            for i in range(B):
                # Get the indices for the masked patches and replace them with mask_token
                masked_indices = sparse_shuffle[i, num_keep:]  # Get the indices of the patches to mask
                x_reshaped[i, masked_indices, :] = self.mask_token.to(x_reshaped.dtype)  # Replace with mask_token
            
            # Reshape it back to the original shape [B, C, D, H, W]
            x_masked = rearrange(x_reshaped, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)

        return x_masked, mask


    def patchify(self, imgs):
        """
        imgs: (N, C, D, H, W)
        x: (N, L, patch_size[0]*patch_size[1]*patch_size[2]*C)
        """
        # 获取每个维度的patch大小
        pd, ph, pw = self.patch_embed.patch_size
        assert imgs.shape[2] % pd == 0 and imgs.shape[3] % ph == 0 and imgs.shape[4] % pw == 0
        
        d = imgs.shape[2] // pd
        h = imgs.shape[3] // ph
        w = imgs.shape[4] // pw
        x = imgs.reshape(imgs.shape[0], imgs.shape[1], d, pd, h, ph, w, pw)
        # x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # Reorder dimensions
        x = torch.einsum('ncdohpwq->ndhwopqc', x)
        x = x.reshape(imgs.shape[0], d * h * w, pd * ph * pw * imgs.shape[1])  # Flatten
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size[0]*patch_size[1]*patch_size[2]*C)
        imgs: (N, C, D, H, W)
        """
        pd, ph, pw = self.patch_embed.patch_size
        N, L, D = x.shape
        d = h = w = int(L ** (1/3))  # Assuming cubic patches for simplicity
        assert d * h * w == L, "Number of patches does not match expected volume"
        
        x = x.reshape(shape=(N, d, h, w, pd, ph, pw, self.in_chans))
        # x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()  # Reorder dimensions back
        x = torch.einsum('ndhwopqc->ncdohpwq', x)
        imgs = x.reshape(N, self.in_chans, d * pd, h * ph, w * pw)  # Reshape back to original dimensions
        return imgs



    def forward(self, x, mask_ratio=0.75):
        # Step 1: Pass through Swin3D encoder (Patch embedding and initial transformer layers)
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)  # Dropout
        # x0_out = self.proj_out(x0, normalize)

        # Step 2: Apply Masking after Patch Embedding
        # x_masked, mask = self.window_masking(x0, mask_ratio, remove=False)
        # Add class token
        # cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
        # x_masked = torch.cat([cls_token, x_masked], dim=1)

        # Step 3: Transformer blocks
        x1 = self.layers1[0](x0.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x4 = self.layers4[0](x3.contiguous())

        # # decoder
        # dec3 = self.decoder5(x4)
        # dec2 = self.decoder4(dec3)
        # dec1 = self.decoder3(dec2)
        # dec0 = self.decoder2(dec1)
        # out = self.decoder1(dec0)
        
        # # reshape and normalize
        # out = out.permute(0, 2, 3, 4, 1)
        # out = self.norm_up(out)
        # pred = self.decoder_pred(out)

        
        # # Step 5: Compute loss
        # loss = self.forward_loss(x, pred, mask)

        return x4
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute the reconstruction loss between predicted and target (masked) images.
        """
        target = self.patchify(imgs)
        pred = rearrange(pred, 'b d h w c -> b (d h w) c')  # Reshape to [B, L, C]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # Apply mask ratio weight
        return loss

# 加载权重函数
def load_weight(model, path):
    # 加载预训练权重
    pretrain_dict = torch.load(path, map_location=torch.device('cpu'))['model']
    model_dict = model.state_dict()

    # 打印模型的总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params}")

    # 过滤掉尺寸不匹配的参数
    filtered_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    mismatched_keys = {k: (v.size(), model_dict[k].size()) for k, v in pretrain_dict.items() if k in model_dict and v.size() != model_dict[k].size()}

    # 打印成功加载的参数数目
    print(f"Loaded parameters: {len(filtered_dict)} / {len(model_dict)}")

    # 找出未加载的参数（包括预训练中不存在的key）
    missing_keys = set(model_dict.keys()) - set(filtered_dict.keys())

    # 更新模型权重
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)  # 使用 strict=False 避免报错

    # 打印未加载的参数
    if missing_keys:
        print("Missing keys (not loaded from checkpoint):")
        for key in sorted(missing_keys):
            print(f"  {key}")

    # 打印维度不匹配的参数
    if mismatched_keys:
        print("Mismatched keys (size mismatch):")
        for key, (ckpt_shape, model_shape) in mismatched_keys.items():
            print(f"  {key}: checkpoint shape {ckpt_shape}, model shape {model_shape}")

    return model


def swin3d_base(pretrained=True, **kwargs):
    in_chans = 1  # 输入通道数（例如，单通道CT图像）
    # img_size = (256, 256, 128)  # 图像的尺寸：高度，宽度，深度
    patch_size = (4, 4, 2)  # 每个patch的尺寸

    # 选择模型参数
    embed_dim = 48
    depths = [2, 2, 6, 2]  # 每个阶段的深度
    num_heads = [3, 6, 12, 24]  # 每个阶段的头数

    # 创建模型实例
    model = MRMWithSwin3D(
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        **kwargs
    )
    if pretrained:
        # load pretrain model
        model = load_weight(model, "/data2/haoranlai/Project/gloria/Pretrain_model/swin_vit/checkpoint-250.pth")
    
    return model, embed_dim * (2 ** len(depths)), embed_dim * (2 ** len(depths))
