import torch

from PIL import Image
from .. import builder
from .. import loss
from .. import utils

from pytorch_lightning.core import LightningModule
from torch.autograd import Variable
import ipdb

class PretrainDQNWOSAGLModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.gloria = builder.build_gloria_dqn_wo_self_atten_gl_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        if self.cfg.model.pretrain is not None:
                self.load_xray_pretrain(self.cfg.model.pretrain)
    
    def load_xray_pretrain(self, path):
        self.gloria.load_xray_pretrain(path)

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.gloria)
        # ipdb.set_trace()
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
