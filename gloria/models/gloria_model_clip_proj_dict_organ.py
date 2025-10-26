import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics
import random
from PIL import Image
from .. import builder
from .. import loss
from .. import utils
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
import nibabel as nib
import ipdb
from skimage.transform import resize
import torch.nn.functional as F
from collections import defaultdict


class CLIPProjDictOrgan(nn.Module):
    def __init__(self, cfg):
        super(CLIPProjDictOrgan, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_proj_model(cfg)
        self.img_encoder = builder.build_img_model(cfg)
        # self.organ_attention = builder.build_organ_attention(cfg)
        self.dictionarylearning = builder.build_dictionary_model(cfg)
        
        self.local_loss = loss.gloria_loss.local_loss_3d
        self.global_loss = loss.gloria_loss.global_loss
        self.local_loss_weight = self.cfg.model.gloria.local_loss_weight
        self.global_loss_weight = self.cfg.model.gloria.global_loss_weight
        self.global_recon_loss_weight = self.cfg.model.gloria.global_recon_loss_weight
        self.mse_loss_weight = self.cfg.model.gloria.mse_loss_weight

        self.temp1 = self.cfg.model.gloria.temp1
        self.temp2 = self.cfg.model.gloria.temp2
        self.temp3 = self.cfg.model.gloria.temp3
        self.batch_size = self.cfg.train.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_emb_g, img_emb_l, hidden_state = self.img_encoder(imgs, get_local=True, select_layer=10)

        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(
            img_emb_g, img_emb_l
        )
        # img_emb_g = F.normalize(img_emb_g, dim=-1)
        # img_emb_l = F.normalize(img_emb_l, dim=-1)
        return img_emb_l, img_emb_g, hidden_state

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

    def calc_loss(self, img_emb_g, img_emb_g_recon, all_organ_img_features, all_recon_organ_img_features, text_emb_g, text_emb_g_recon,  all_organ_text_features, all_recon_organ_text_features):
        
        # reconstruction loss
        img_recon_loss = F.mse_loss(img_emb_g_recon, img_emb_g)
        text_recon_loss = F.mse_loss(text_emb_g_recon, text_emb_g)

        # clip loss for global features
        g_loss0, g_loss1 = self._calc_global_loss(F.normalize(img_emb_g, dim=-1), F.normalize(text_emb_g, dim=-1))
        g_rec_loss0, g_rec_loss1 = self._calc_global_loss(F.normalize(img_emb_g_recon, dim=-1), F.normalize(text_emb_g_recon, dim=-1))


        # clip loss for local features
        if len(all_organ_img_features) > 0:
            organ_img_recon_loss = 0
            organ_text_recon_loss = 0
            g_loss0_organ = 0
            g_loss1_organ = 0
            g_rec_loss0_organ = 0
            g_rec_loss1_organ = 0
            # calculate the organ loss
            for organ_img_features, organ_img_features_recon, organ_text_features, organ_text_features_recon in zip(all_organ_img_features, all_recon_organ_img_features, all_organ_text_features, all_recon_organ_text_features):
                organ_img_recon_loss += F.mse_loss(organ_img_features_recon, organ_img_features)
                organ_text_recon_loss += F.mse_loss(organ_text_features_recon, organ_text_features)

                g_loss0_organ_, g_loss1_organ_ = self._calc_global_loss(F.normalize(organ_img_features, dim=-1), F.normalize(organ_text_features, dim=-1))
                g_loss0_organ += g_loss0_organ_
                g_loss1_organ += g_loss1_organ_

                g_rec_loss0_organ_, g_rec_loss1_organ_ = self._calc_global_loss(F.normalize(organ_img_features_recon, dim=-1), F.normalize(organ_text_features_recon, dim=-1))
                g_rec_loss0_organ += g_rec_loss0_organ_
                g_rec_loss1_organ += g_rec_loss1_organ_
        
        # weighted loss
        loss = 0
        loss += (g_loss0 + g_loss1) * self.global_loss_weight
        loss += ( img_recon_loss + text_recon_loss ) * self.mse_loss_weight
        loss += (g_rec_loss0 + g_rec_loss1) * self.global_loss_weight    #( wrong all the time)
        if len(all_organ_img_features) > 0:
            loss += (g_loss0_organ + g_loss1_organ) / len(all_organ_img_features) * self.global_loss_weight
            loss += (organ_img_recon_loss + organ_text_recon_loss) / len(all_organ_img_features) * self.mse_loss_weight
            loss += (g_rec_loss0_organ + g_rec_loss1_organ) / len(all_organ_img_features) * self.global_loss_weight
        return loss

    def forward(self, x):
        # img encoder branch
        img_emb_l, img_emb_g, hidden_state = self.image_encoder_forward(x["imgs"])

        hidden_global, hidden_local = hidden_state[:, 0:1], hidden_state[:, 1:]
    
        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )
        
          # text encoder branch for shot_caps
        shot_text_emb_l, shot_text_emb_g, shot_sents = self.text_encoder_forward(
            x["shot_caption_ids"], x["shot_attention_mask"], x["shot_token_type_ids"]
        )

        # organ for shot caps
        organ = x['shot_organ_lists']
        shot_organ_indices = x['shot_organ_token_maps']
   
        organ_groups = self.get_organ_groups_with_indices(organ, shot_organ_indices)

        all_organ_img_features = []  # 图像特征配对列表
        all_organ_text_features = []  # 文本特征配对列表
        all_recon_organ_img_features = []  # 图像特征配对列表
        all_recon_organ_text_features = []  # 文本特征配对列表

        # # add one loss
        # total_choise = len(organ_groups)
        # choise = torch.randint(0, total_choise, (1,)).item()
        # organ_group, organ_name = organ_groups[choise]

        # if len(organ_groups) > 0:
        #     # 遍历每个器官组
        #     organ_img_features = []
        #     organ_text_features = []
        #     # 提取该器官的图像和文本特征
        #     for sample_idx, organ_idx in organ_group:
        #         # 提取图像特征
        #         organ_token_indices = shot_organ_indices[sample_idx][organ_idx]
        #         organ_token_embeddings = hidden_local[sample_idx][organ_token_indices]
        #         organ_token_embeddings = torch.cat([hidden_global[sample_idx], organ_token_embeddings], dim=0)
        #         organ_img_feature = self.img_encoder.model.blocks[11](organ_token_embeddings.unsqueeze(0))
        #         organ_img_feature = self.img_encoder.model.norm(organ_img_feature)
        #         organ_img_feature_g, organ_img_feature_l = self.img_encoder.generate_embeddings(
        #         organ_img_feature[:, 0], organ_img_feature[:, 1:])
                
        #         # 提取对应文本特征
        #         organ_text_feature = shot_text_emb_g[sample_idx * self.cfg.data.text.organ_number  + organ_idx]
                
        #         # 将图像和文本特征加入配对列表
        #         organ_img_features.append(organ_img_feature_g.squeeze(0))
        #         organ_text_features.append(organ_text_feature)
        #     all_organ_img_features.append(torch.stack(organ_img_features))
        #     all_organ_text_features.append(torch.stack(organ_text_features))
        #     all_recon_organ_img_features.append(self.dictionarylearning(torch.stack(organ_img_features)))
        #     all_recon_organ_text_features.append(self.dictionarylearning(torch.stack(organ_text_features))) 

        if len(organ_groups) > 0:
            # 遍历每个器官组
            for organ_group, organ_name in organ_groups:
                organ_img_features = []
                organ_text_features = []
                # 提取该器官的图像和文本特征
                for sample_idx, organ_idx in organ_group:
                    # 提取图像特征
                    organ_token_indices = shot_organ_indices[sample_idx][organ_idx]
                    organ_token_embeddings = hidden_local[sample_idx][organ_token_indices]
                    organ_token_embeddings = torch.cat([hidden_global[sample_idx], organ_token_embeddings], dim=0)
                    organ_img_feature = self.img_encoder.model.blocks[11](organ_token_embeddings.unsqueeze(0))
                    organ_img_feature = self.img_encoder.model.norm(organ_img_feature)
                    organ_img_feature_g, organ_img_feature_l = self.img_encoder.generate_embeddings(
                    organ_img_feature[:, 0], organ_img_feature[:, 1:])
                    
                    # 提取对应文本特征
                    organ_text_feature = shot_text_emb_g[sample_idx * self.cfg.data.text.organ_number  + organ_idx]
                    
                    # 将图像和文本特征加入配对列表
                    organ_img_features.append(organ_img_feature_g.squeeze(0))
                    organ_text_features.append(organ_text_feature)
                all_organ_img_features.append(torch.stack(organ_img_features))
                all_organ_text_features.append(torch.stack(organ_text_features))
                all_recon_organ_img_features.append(self.dictionarylearning(torch.stack(organ_img_features)))
                all_recon_organ_text_features.append(self.dictionarylearning(torch.stack(organ_text_features))) 

        img_emb_g_recon = self.dictionarylearning(img_emb_g)
        text_emb_g_recon = self.dictionarylearning(text_emb_g)

        # Example of returning the pairs for contrastive learning
        return img_emb_g, img_emb_g_recon, all_organ_img_features, all_recon_organ_img_features, text_emb_g, text_emb_g_recon,  all_organ_text_features, all_recon_organ_text_features, sents

    # def get_organ_groups_with_indices(self, shot_organ_lists, shot_organ_indices):
    #     # 初始化字典来记录每个器官的样本索引和器官在样本内的具体位置
    #     organ_to_samples = defaultdict(list)
        
    #     # 遍历每个样本和其包含的器官
    #     for sample_idx, organs in enumerate(shot_organ_lists):
    #         # 检查 shot_organ_indices 对应位置是否为非空列表
    #         # 遍历该样本的每个器官
    #         for organ_idx, organ in enumerate(organs):
    #             # 检查该器官的索引列表是否非空
    #             if shot_organ_indices[sample_idx] and shot_organ_indices[sample_idx][organ_idx]:
    #                 # 如果该器官在 shot_organ_indices 中存在有效索引，记录到 organ_to_samples 中
    #                 organ_to_samples[organ].append((sample_idx, organ_idx))
    
        
    #     # 构造器官样本组
    #     organ_groups = []
    #     for organ, samples in organ_to_samples.items():
    #         # 仅处理出现次数大于1的器官
    #         if len(samples) > 1:
    #             # 将包含相同器官的所有样本作为一个组合
    #             organ_groups.append((samples, organ))

    #     return organ_groups
    
    def get_organ_groups_with_indices(self, shot_organ_lists, shot_organ_indices):
        # 初始化字典来记录每个器官的样本索引和器官在样本内的具体位置
        organ_to_samples = defaultdict(list)
        
        # 遍历每个样本和其包含的器官
        for sample_idx, organs in enumerate(shot_organ_lists):
            # 用于追踪该样本中已处理的器官
            processed_organs = set()
            
            # 遍历该样本的每个器官
            for organ_idx, organ in enumerate(organs):
                # 检查该器官的索引列表是否非空且尚未处理过该器官
                if (organ not in processed_organs 
                    and shot_organ_indices[sample_idx] 
                    and shot_organ_indices[sample_idx][organ_idx]):
                    
                    # 如果该器官在 shot_organ_indices 中存在有效索引，记录到 organ_to_samples 中
                    organ_to_samples[organ].append((sample_idx, organ_idx))
                    
                    # 标记该器官已处理
                    processed_organs.add(organ)
        
        # 构造器官样本组
        organ_groups = []
        for organ, samples in organ_to_samples.items():
            # 仅处理出现次数大于1的器官
            if len(samples) > 1:
                # 将包含相同器官的所有样本作为一个组合
                organ_groups.append((samples, organ))

        return organ_groups


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
            cls_2_processed_txt[k] = self.process_text(v, device)

        return cls_2_processed_txt

    def process_img(self, path, device):

        # transform = builder.build_transformation_3D(self.cfg, "test")
        transform = builder.build_transformation_3D_M3D(self.cfg, "test")
        nii_img = nib.load(str(path))
        img = nii_img.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # transform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)

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