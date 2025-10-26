import torch

from PIL import Image
from .. import builder
from .. import loss
from .. import utils

from pytorch_lightning.core import LightningModule
from torch.autograd import Variable
import ipdb



class PretrainDQNWOSAMLPGLCLIPOPENAIModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.gloria = builder.build_gloria_dqn_wo_self_atten_gl_open_clip_model(cfg)
        self.load_weights_from_pretrained_model()
        self.lr = cfg.lightning.trainer.lr
        self.dm = None


    def load_weights_from_pretrained_model(self):
        ckpt = torch.load(self.cfg.model.pretrain_path, map_location='cpu')
        ckpt_dict = ckpt["state_dict"]
        model_weights = self.gloria.state_dict()
        fixed_ckpt_dict = {}
        for k, v in ckpt_dict.items():
            new_key = k.split("gloria.")[-1]
            if new_key in model_weights:
                fixed_ckpt_dict[new_key] = v
        ckpt_dict = fixed_ckpt_dict
        self.gloria.load_state_dict(ckpt_dict, strict=False)

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.gloria)
        # ipdb.set_trace()
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")

        # # get attention map image
        # if self.cfg.train.update_interval is not None:
        #     if batch_idx % self.cfg.train.update_interval == 0:
        #         imgs = batch["imgs"].cpu()
        #         self.gloria.plot_attn_maps(
        #             attn_maps, imgs, sents, self.current_epoch, batch_idx
        #         )
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
