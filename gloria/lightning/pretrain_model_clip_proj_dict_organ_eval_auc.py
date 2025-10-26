import torch

from PIL import Image
from .. import builder
from .. import loss
from .. import utils

from pytorch_lightning.core import LightningModule
from torch.autograd import Variable


class PretrainCLIPProjDictOrganModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.gloria = builder.build_CLIP_proj_dict_organ_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.gloria)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss, sents = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        # 逐个batch计算loss并累积预测值
        logist, labels, self.get_organ_similarities(self.gloria, batch, txts)
        return {"logits": logits, "labels": labels}

    def validation_epoch_end(self, outputs):
        # 汇总所有 batch 的预测和标签
        all_logits = torch.cat([x["logits"] for x in outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in outputs], dim=0)

        # 计算整个验证集的 AUC
        val_auc = self.val_auc(all_logits, all_labels.int())
        self.log("val_auc", val_auc, prog_bar=True)


    def shared_step(self, batch, split):
        if split == "train":
            """Similar to traning step"""
            img_emb_g, img_emb_g_recon, all_organ_img_features, all_recon_organ_img_features, text_emb_g, text_emb_g_recon,  all_organ_text_features, all_recon_organ_text_features, sents = self.gloria(batch)
            loss = self.gloria.calc_loss(
                img_emb_g, img_emb_g_recon, all_organ_img_features, all_recon_organ_img_features, text_emb_g, text_emb_g_recon,  all_organ_text_features, all_recon_organ_text_features
            )
            # log training progress
            log_iter_loss = True if split == "train" else False
            self.log(
                f"{split}_loss",
                loss,
                on_epoch=True,
                on_step=log_iter_loss,
                logger=True,
                prog_bar=True,
            )
            return loss, sents
        else:
            """Similar to validation step"""
            simi = self.get_organ_similarities(self.gloria, batch, txts)
            auc = self.gloria.calc_auc(
                img_emb_g, img_emb_g_recon, all_organ_img_features, all_recon_organ_img_features, text_emb_g, text_emb_g_recon,  all_organ_text_features, all_recon_organ_text_features
            )
            # log training progress
            log_iter_loss = True if split == "train" else False
            self.log(
                f"{split}_auc",
                auc,
                on_epoch=True,
                on_step=log_iter_loss,
                logger=True,
                prog_bar=True,
            )
            return auc, sents

    def get_organ_similarities(gloria_model, batch, txts):

        imgs, organ_index, organ_name = batch['images'], batch['organ_dict'], batch['organ_map']
        # get global and local image features
        with torch.no_grad():
            img_emb_l, img_emb_g, hidden_state = gloria_model.image_encoder_forward(imgs)
            hidden_global, hidden_local = hidden_state[:, 0:1], hidden_state[:, 1:]
            text_emb_l, text_emb_g, _ = gloria_model.text_encoder_forward(
                txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
            )
            
            if len(organ_index) ==  0:
                organ_img_feature = img_emb_g
            else:
                organ_token_embeddings = hidden_local[0, organ_index]
                organ_token_embeddings = torch.cat([hidden_global[0], organ_token_embeddings], dim=0)
                organ_img_feature = gloria_model.img_encoder.model.blocks[11](organ_token_embeddings.unsqueeze(0))
                organ_img_feature = gloria_model.img_encoder.model.norm(organ_img_feature)
                organ_img_feature, organ_img_feature_l = gloria_model.img_encoder.generate_embeddings(
                organ_img_feature[:, 0], organ_img_feature[:, 1:])
            organ_img_feature = F.normalize(organ_img_feature, dim=-1)
            text_emb_g = F.normalize(text_emb_g, dim=-1)

        # get similarities
        global_similarities = gloria_model.get_global_similarities(organ_img_feature, text_emb_g)


        return global_similarities.detach().cpu().numpy()



