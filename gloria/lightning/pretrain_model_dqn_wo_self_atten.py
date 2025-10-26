import torch

from PIL import Image
from .. import builder
from .. import loss
from .. import utils

from pytorch_lightning.core import LightningModule
from torch.autograd import Variable
import ipdb

class PretrainDQNWOSAModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.gloria = builder.build_gloria_dqn_wo_self_atten_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        if self.cfg.model.pretrain is not None:
            self.load_xray_pretrain(self.cfg.model.pretrain)
    
    def load_xray_pretrain(self, path):
        # 加载预训练的权重
        ckpt = torch.load(path, map_location='cpu')
        ckpt_dict = ckpt["state_dict"]
        model_weights = self.gloria.state_dict()

        fixed_ckpt_dict = {}
        for k, v in ckpt_dict.items():
            new_key = k.split("gloria.")[-1]  # 调整键以匹配模型的键
            if new_key in model_weights:
                # 检查权重形状是否一致
                if model_weights[new_key].shape == v.shape:
                    fixed_ckpt_dict[new_key] = v
                else:
                    print(f"Shape mismatch for '{new_key}': model shape {model_weights[new_key].shape}, checkpoint shape {v.shape}")

        # 使用检查过形状的权重字典更新模型
        self.gloria.load_state_dict(fixed_ckpt_dict, strict=False)
        print("Pretrained weights loaded with shape verification.")

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.gloria)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        return loss

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls = self.gloria(batch)
        loss = self.gloria.calc_loss(
            img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls
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

        return loss
