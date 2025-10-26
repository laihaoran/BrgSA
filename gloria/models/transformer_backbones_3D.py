# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial


import torch
import torchvision
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block
import numpy as np
import ipdb


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




class MRM(nn.Module):
    """Masked Autoencoder with 3D VisionTransformer backbone."""
    def __init__(self, img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        # define value
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        #defien function
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        
        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]*patch_size[1]*patch_size[2]*in_chans, bias=True)
        
        # self.norm_pix_loss = norm_pix_loss
        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights as needed, especially the 3D positional embeddings
        grid_d, grid_h, grid_w = (self.img_size[0] // self.patch_size[0], 
                                  self.img_size[1] // self.patch_size[1], 
                                  self.img_size[2] // self.patch_size[2])
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], grid_d, grid_h, grid_w, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_embed_dim, grid_d, grid_h, grid_w, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Other weight initialization as before
        # Initialize nn.Linear and nn.LayerNorm here as well, similar to previous initialization logic
          # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def image_encoder(self, x):
 
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return  x[:, 0, :], x[:, 1:,:]
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        # imgs = imgs.cuda()

        global_latent, local_latent = self.image_encoder(imgs)
        # pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # ipdb.set_trace()
        # loss = self.forward_loss(imgs, pred, mask)

        return local_latent
    
    
    def forward_features(self, imgs):
        global_latent, local_latent = self.image_encoder(imgs)
        return local_latent


def update_weights(model, pretrained_dict):
    model_dict = model.state_dict()
    
    # 过滤掉不匹配的权重，确保键名和形状都匹配
    filtered_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        # 如果键名存在于模型字典中，并且形状相同
        stripped_k = k[len('vision_encoder.'):] if k.startswith('vision_encoder.') else k
        if stripped_k in model_dict and model_dict[stripped_k].shape == v.shape:
            filtered_pretrained_dict[stripped_k] = v
    
    # 打印匹配的键的数量
    print('matched keys:', len(filtered_pretrained_dict))
    
    # 使用过滤后的预训练字典更新模型的参数字典
    model_dict.update(filtered_pretrained_dict)
    return model_dict


def load_weight(model, path):
    pretrain_dict = torch.load(path, map_location=torch.device('cpu'))
    update_dict = update_weights(model, pretrain_dict['model'])
    # update_dict = update_weights(model, pretrain_dict['state_dict'])
    model.load_state_dict(update_dict)
    return model


def mrm_vit_b16(**kwargs):
    model = MRM(
        img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1, embed_dim=1536, depth=12, num_heads=12,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model = load_weight(model, '/haoranlai/Project/MRM_pretrain_vit/output_vit_b16_3D_embedding768_6gpus/checkpoint-116.pth')
    model = load_weight(model, "/haoranlai/Project/MRM_pretrain_vit/output_vit_b16_3D_embedding1536_4gpus_stable_c/checkpoint-176.pth")
    return model

def mrm_vit_l16(**kwargs):
    model = MRM(
        img_size=(224, 224, 112), patch_size=(16, 16, 8), in_chans=1, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model   



# if __name__ == "__main__":
#     import ipdb
#     model = mrm_vit_b16().cuda()
#     print(model)
#     print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

#     imgs = torch.randn(2, 1, 224, 224, 112).cuda()
#     loss = model({"image": imgs})
#     print(loss)
#     ipdb.set_trace()
