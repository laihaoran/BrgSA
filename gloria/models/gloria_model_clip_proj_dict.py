import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics

from PIL import Image
from .. import builder
from .. import loss
from .. import utils
from transformers import AutoTokenizer, BertTokenizer
from nltk.tokenize import RegexpTokenizer
import nibabel as nib
import ipdb
from skimage.transform import resize
import torch.nn.functional as F



class CLIPProjDict(nn.Module):
    def __init__(self, cfg):
        super(CLIPProjDict, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_proj_model(cfg)  #close proj
        self.img_encoder = builder.build_img_model(cfg)
        self.dictionarylearning = builder.build_dictionary_model(cfg)
        # self.dictionarylearning_d = builder.build_dictionary_model(cfg)
        self.local_loss = loss.gloria_loss.local_loss_3d
        self.global_loss = loss.gloria_loss.global_loss
        # self.diversity_loss = loss.diversity.memory_diversity_loss
        self.local_loss_weight = self.cfg.model.gloria.local_loss_weight
        self.global_loss_weight = self.cfg.model.gloria.global_loss_weight
        self.global_recon_loss_weight = self.cfg.model.gloria.global_recon_loss_weight
        self.mse_loss_weight = self.cfg.model.gloria.mse_loss_weight
        self.temp1 = self.cfg.model.gloria.temp1
        self.temp2 = self.cfg.model.gloria.temp2
        self.temp3 = self.cfg.model.gloria.temp3
        self.batch_size = self.cfg.train.batch_size

        # self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model.text.bert_type, trust_remote_code=True)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_emb_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(    #close norm
            img_emb_g, img_emb_l
        )

        # img_emb_g = F.normalize(img_emb_g, dim=-1)
        # img_emb_l = F.normalize(img_emb_l, dim=-1)
        return img_emb_l, img_emb_g

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents):
        
        # ipdb.set_trace()
        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, attn_maps = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1

    def calc_loss(self, img_emb_g_recon, img_emb_g, text_emb_g_recon, text_emb_g, sents):

        # l_loss0, l_loss1, attn_maps = self._calc_local_loss(
        #     img_emb_l, text_emb_l, sents
        # )

        # reconstruction loss
        img_recon_loss = F.mse_loss(img_emb_g_recon, img_emb_g)
        text_recon_loss = F.mse_loss(text_emb_g_recon, text_emb_g)

        img_emb_g = F.normalize(img_emb_g, dim=-1)
        text_emb_g = F.normalize(text_emb_g, dim=-1)
        img_emb_g_recon = F.normalize(img_emb_g_recon, dim=-1)
        text_emb_g_recon = F.normalize(text_emb_g_recon, dim=-1)

        # global loss
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g) # correct here
        g_rec_loss0, g_rec_loss1 = self._calc_global_loss(img_emb_g_recon, text_emb_g_recon)

        # diversity loss
        # d_loss = self.diversity_loss(self.dictionarylearning.dictionary.weight) 

        # weighted loss
        loss = 0

        loss += (g_loss0 + g_loss1) * self.global_loss_weight
        loss += ( img_recon_loss + text_recon_loss ) * self.mse_loss_weight
        loss += (g_rec_loss0 + g_rec_loss1) * self.global_recon_loss_weight 
        # loss += d_loss * self.global_loss_weight 
        return loss

    def forward(self, x):
        # img encoder branch
        # x["imgs"] = x["imgs"].repeat(1, 3, 1, 1, 1).permute(0, 1, 4, 3, 2)
        img_emb_l, img_emb_g = self.image_encoder_forward(x["imgs"])

        # img_emb_l = img_emb_l.reshape(img_emb_l.shape[0], 14, 14, 14, -1).permute(0, 4, 1, 2, 3)
        
        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        # text_emb_l, text_emb_g, sents = self.text_encoder_forward(
        #     x["caption_ids"], x["attention_mask"], []
        # )
        # 

        img_emb_g_recon = self.dictionarylearning(img_emb_g)
        text_emb_g_recon = self.dictionarylearning(text_emb_g)


        return img_emb_g_recon, img_emb_g, text_emb_g_recon, text_emb_g, sents

    def get_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, cap_lens):


        batch_size = img_emb_l.shape[0]
        similarities = []

        for i in range(len(text_emb_l)):
            words_num = cap_lens[i]
            
            # Extract the word embeddings for the current caption
            word = text_emb_l[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()  # [1, embedding_dim, words_num]
            word = word.repeat(batch_size, 1, 1)  # Repeat for each image in the batch [batch_size, embedding_dim, words_num]
            
            # Context is now 3D image embeddings [batch_size, embedding_dim, depth, height, width]
            context = img_emb_l
            
            # Apply 3D attention function
            weiContext, attn = loss.gloria_loss.attention_fn_3d(
                word, context, 4.0
            )  # [batch_size, embedding_dim, words_num], [batch_size, words_num, depth, height, width]

            # Transpose word and weiContext to match dimensions for cosine similarity calculation
            word = word.transpose(1, 2).contiguous()  # [batch_size, words_num, embedding_dim]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [batch_size, words_num, embedding_dim]

            # Flatten the word embeddings and the weighted context for cosine similarity calculation
            word = word.view(batch_size * words_num, -1)  # [batch_size * words_num, embedding_dim]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [batch_size * words_num, embedding_dim]

            # Calculate cosine similarity
            row_sim = loss.gloria_loss.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # Reshape back to [batch_size, words_num]

            # Scale the similarities and take the max along the word dimension
            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [batch_size, 1]

            # Apply logarithm to the max similarity
            row_sim = torch.log(row_sim)

            # Append the calculated similarities for the current sample
            similarities.append(row_sim)

        # Concatenate all similarities across the batch
        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities

    def get_attn_maps(self, img_emb_l, text_emb_l, sents):
        _, _, attn_maps = self._calc_local_loss(img_emb_l, text_emb_l, sents)
        return attn_maps

    def plot_attn_maps(self, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

        img_set, _ = utils.build_attention_images(
            imgs,
            attn_maps,
            max_word_num=self.cfg.data.text.word_num,
            nvis=self.cfg.train.nvis,
            rand_vis=self.cfg.train.rand_vis,
            sentences=sents,
        )

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = (
                f"{self.cfg.output_dir}/"
                f"attention_maps_epoch{epoch_idx}_"
                f"{batch_idx}.png"
            )
            im.save(fullpath)

    def process_text(self, text, device):

        if type(text) == str:
            text = [text]

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_class_prompts(self, class_prompts, device):

        cls_2_processed_txt = {}
        for k, v in class_prompts.items():
            
            # cls_2_processed_txt[v[0]] = self.process_text(v, device)
            cls_2_processed_txt[k] = self.process_text(v, device)

        return cls_2_processed_txt

    def process_img(self, path, device):

        # transform = builder.build_transformation_3D(self.cfg, "test")
        transform = builder.build_transformation_3D_M3D(self.cfg, "test")
        nii_img = nib.load(str(path))
        img = nii_img.get_fdata()

        # if self.cfg.data.image.imsize is not None:
        #     # transform images
        #     img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
            # img = resize(img, (224,224,64), mode='reflect', anti_aliasing=True)

        if transform is not None:
            img = transform(img)

        return img
    
    def process_single_img(self, paths):

        transform = builder.build_transformation(self.cfg, split="test")
        x = cv2.imread(str(paths), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        img = transform(img)

        return img



    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
    

    # ===== GWAR 工具函数（类内） =====
    @staticmethod
    def _row_norm(M: torch.Tensor) -> torch.Tensor:
        return M / (M.sum(dim=-1, keepdim=True) + 1e-6)

    def _grad_weighted_rollout(self, attn_list: list) -> torch.Tensor:
        """
        attn_list: List[L] of [B, H, T, T]，每个张量已在前向中 retain_grad()
        返回: [B, T, T]
        """
        assert len(attn_list) > 0, "attn_list 为空，检查是否 return_all_attn=True 且未走 fused 路径"
        mats = []
        for A in attn_list:
            G = A.grad  # [B, H, T, T]
            if G is None:
                # 这层可能没被反传到，跳过
                continue
            A = A.mean(dim=1)                  # [B, T, T] 头均值
            G = G.mean(dim=1).clamp_min(0.0)   # 只取正梯度更稳
            M = A * G
            I = torch.eye(M.size(-1), device=M.device).unsqueeze(0)
            M = self._row_norm(M + I)          # 残差 + 行归一
            mats.append(M)

        assert len(mats) > 0, "没有可用的梯度加权注意力矩阵，请确认未使用 no_grad/inference_mode 且 fused_attn=False。"
        R = mats[0]
        for M in mats[1:]:
            R = R @ M
        return R  # [B, T, T]

    @torch.no_grad()
    def _upsample_to_image_like(self, heat_DzHyWx: torch.Tensor, img_zyx: torch.Tensor) -> torch.Tensor:
        """
        heat_DzHyWx: (Dz,Hy,Wx)
        img_zyx: (1,1,Z,Y,X) 或 (Z,Y,X)；用于确定上采样目标体素大小
        返回: (Z,Y,X) 的浮点热图（0~1，不做 min-max 放大）
        """
        if img_zyx.dim() == 3:
            img_zyx = img_zyx.unsqueeze(0).unsqueeze(0)
        Dz, Hy, Wx = heat_DzHyWx.shape
        Z, Y, X = img_zyx.shape[-3:]
        x = heat_DzHyWx[None, None]  # (1,1,Dz,Hy,Wx)
        out = F.interpolate(x, size=(Z, Y, X), mode="trilinear", align_corners=False)
        out = out[0, 0]  # (Z,Y,X)
        m = out.max()
        if float(m) > 1e-12:
            out = out / m
        else:
            out = torch.zeros_like(out)
        return out

    @torch.enable_grad()
    def compute_gwar_heatmap(self, batch):
        """
        输入 batch 至少包含:
          - "imgs": [B, 1, Z, Y, X]
          - "caption_ids", "attention_mask", "token_type_ids"
        输出:
          - heatmaps: [B, Dz, Hy, Wx]  （L1 归一）
          - grid: (Dz, Hy, Wx)
        说明:
          - 需确保图像编码器走的是非 fused 注意力路径；且 forward 时传 return_all_attn=True
          - 不会改变模型权重；只做一次 forward + backward
        """
        device = next(self.parameters()).device
        imgs = batch["imgs"].to(device)
        caption_ids   = batch["caption_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        # ===== 视觉前向：拿到每层 softmax 后注意力 =====
        # 你的 img_encoder.forward 签名需支持 return_all_attn=True（见前面已改好的 CustomViT3D）
        # 常见返回: cls_token, patches, last_attn_pp, all_attn
        cls_token, patches, last_attn_pp, all_attn = self.img_encoder(imgs, get_local=True, return_all_attn=True)

        cls_token, patches = self.img_encoder.generate_embeddings(    #close norm
            cls_token, patches
        )
        grid = self.img_encoder.model.last_grid_size # 若你的 img_encoder 外面包了 tower
        # import ipdb; ipdb.set_trace()
        # # 取当前 patch 网格
        # try:
            
        # except AttributeError:
        #     # 若没有 tower，直接从内部 vit 取
        #     grid = self.img_encoder.patch_embed.grid_size  # (Dz, Hy, Wx)

        # ===== 文本前向 =====
        _, text_emb_g, _ = self.text_encoder_forward(caption_ids, attention_mask, token_type_ids)

        # ===== 以训练一致方式计算全局相似度分数 S =====
        img_g = F.normalize(cls_token, dim=-1)         # [B, D]
        txt_g = F.normalize(text_emb_g, dim=-1)        # [B, D]
        S = F.cosine_similarity(img_g, txt_g, dim=-1).sum()

        # ===== 反传：把梯度打到 all_attn[i] 上 =====
        self.zero_grad(set_to_none=True)
        S.backward(retain_graph=False)

        # ===== GWAR：梯度加权 attention rollout =====
        R = self._grad_weighted_rollout(all_attn)      # [B, T, T]
        cls_to_all = R[:, 0, :]                        # [B, T]
        patch_scores = cls_to_all[:, 1:]               # 去掉 CLS -> [B, P]
        patch_scores = patch_scores / (patch_scores.sum(dim=1, keepdim=True) + 1e-6)

        # ===== reshape 成 (Dz,Hy,Wx) =====
        Dz, Hy, Wx = grid
        B, P = patch_scores.shape
        assert P == Dz * Hy * Wx, f"token 数 {P} 与网格 {Dz}x{Hy}x{Wx} 不匹配"
        heatmaps = patch_scores.reshape(B, Dz, Hy, Wx).contiguous()
        return heatmaps, grid
