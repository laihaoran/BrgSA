import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from . import models
from . import lightning
from . import datasets
from . import loss
from gloria.constants import *
from monai.transforms import (
    Compose, Rand3DElastic, RandAffine, RandFlip, RandCropByPosNegLabel, AddChannel, AddChanneld, ToTensor, ToTensord, LoadImaged, RandSpatialCropd, CenterSpatialCropd, RandRotate90d, RandFlipd, RandScaleIntensityd, RandShiftIntensityd, ScaleIntensityRanged, NormalizeIntensityd,
    ToTensor, NormalizeIntensity, ScaleIntensity, EnsureType, RandSpatialCrop, CenterSpatialCrop, LoadImage, ScaleIntensityRanged, Resize, RandRotate90, RandScaleIntensity, RandShiftIntensity, ScaleIntensityRange
)
import numpy as np
import ipdb

def build_data_module(cfg):
    data_module = datasets.DATA_MODULES[cfg.data.dataset.lower()]
    return data_module(cfg)


def build_lightning_model(cfg, dm):
    module = lightning.LIGHTNING_MODULES[cfg.phase.lower()]
    module = module(cfg)
    module.dm = dm
    return module


def build_gloria_model(cfg):
    gloria_model = models.gloria_model.GLoRIA(cfg)
    return gloria_model


def build_CLIP_model(cfg):
    gloria_model = models.glorai_model_clip.CLIP(cfg)
    return gloria_model

def build_CLIP_proj_model(cfg):
    gloria_model = models.gloria_model_clip_proj.CLIPProj(cfg)
    return gloria_model


def build_CLIP_dict_mse_model(cfg):
    gloria_model = models.gloria_model_mse.CLIPProjDictMSE(cfg)
    return gloria_model

def build_CLIP_proj_dqn_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dqn.CLIPProjDQN(cfg)
    return gloria_model

def build_organ_cls_token_model(cfg):
    gloria_model = models.multi_organ_token.create_multi_organ_cls_token(cfg)
    return gloria_model

def build_CLIP_proj_organ_model(cfg):
    gloria_model = models.gloria_model_clip_proj_organ.CLIPProjOrgan(cfg)
    return gloria_model


def build_CLIP_proj_organ_block_model(cfg):
    gloria_model = models.gloria_model_clip_proj_organ_block.CLIPProjOrganB(cfg)
    return gloria_model

def build_CLIP_proj_dict_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict.CLIPProjDict(cfg)
    return gloria_model


def build_CLIP_proj_kmb_model(cfg):
    gloria_model = models.gloria_model_clip_proj_kmb.CLIPProjKMB(cfg)
    return gloria_model

def build_CLIP_proj_memory_model(cfg):
    gloria_model = models.gloria_model_clip_proj_memory.CLIPProjMem(cfg)
    return gloria_model


def build_CLIP_proj_dict_organ_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_organ.CLIPProjDictOrgan(cfg)
    return gloria_model

def build_CLIP_proj_dict_organ_fast_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_organ_fast.CLIPProjDictOrganFast(cfg)
    return gloria_model


def build_CLIP_proj_dict_organ_neg_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_organ_neg.CLIPProjDictOrganNeg(cfg)
    return gloria_model

def build_CLIP_proj_dict_organ_neg_fast_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_organ_neg_fast.CLIPProjDictOrganNegFast(cfg)
    return gloria_model

def build_CLIP_proj_dict_organ_large_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_organ_large.CLIPProjDictOrganLarge(cfg)
    return gloria_model

def build_CLIP_proj_dict_organ_cls_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_organ_cls.CLIPProjDictOrganCLS(cfg)
    return gloria_model

def build_CLIP_proj_dict_organ_one_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_organ_one.CLIPProjDictOrganOne(cfg)
    return gloria_model

def build_CLIP_proj_dict_organ_one_large_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_organ_one_large.CLIPProjDictOrganOneLarge(cfg)
    return gloria_model

def build_CLIP_proj_dict_gl_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_gl.CLIPProjGLDict(cfg)
    return gloria_model

def build_CLIP_proj_dict_add_model(cfg):
    gloria_model = models.gloria_model_clip_proj_dict_add.CLIPProjDictAdd(cfg)
    return gloria_model

def build_CLIP_proj_gloabl_local_model(cfg):
    gloria_model = models.gloria_model_clip_proj_global_local.CLIPProjGL(cfg)
    return gloria_model

def build_goria_model(cfg):
    gloria_model = models.gloria_model_global.GoRIA(cfg)
    return gloria_model


def build_goria_dict_model(cfg):
    gloria_model = models.gloria_model_global_dict.GoRIADict(cfg)
    return gloria_model


def build_organ_attention(cfg):
    fusion = models.organ_self_attention.make_self_attention(cfg)
    return fusion



def build_gloria_dqn_model(cfg):
    gloria_model = models.gloria_model_dqn.GLoRIADQN(cfg)
    return gloria_model

def build_gloria_dqn_M3AE_model(cfg):
    gloria_model = models.gloria_model_dqn_copy_M3AE.GLoRIADQNM3AE(cfg)
    return gloria_model

def build_gloria_dqn_wo_self_atten_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten.GLoRIADQNWOSA(cfg)
    return gloria_model

def build_gloria_dqn_wo_self_atten_global_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten_global.GLoRIADQNWOSAG(cfg)
    return gloria_model


def build_gloria_dqn_wo_self_atten_gl_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten_gl.GLoRIADQNWOSAGL(cfg)
    return gloria_model



def build_gloria_dqn_wo_self_atten_gl_open_clip_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten_gl_mlp_clip_openai.GLoRIADQNWOSAGLMLPCLIPOPENAI(cfg)
    return gloria_model


def build_gloria_dqn_wo_self_atten_gl_wo_head_cos_proj_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten_gl_cos_proj.GLoRIADQNWOSAGLWOHProjCOS(cfg)
    return gloria_model


def build_gloria_dqn_wo_self_atten_mlp_gl_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten_gl_mlp.GLoRIADQNWOSAGLMLP(cfg)
    return gloria_model

def build_gloria_dqn_wo_self_atten_mlp_gl_aug_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten_gl_mlp_aug.GLoRIADQNWOSAGLMLPAUG(cfg)
    return gloria_model

def build_gloria_dqn_wo_self_atten_gl_proj_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten_gl_proj.GLoRIADQNWOSAGLP(cfg)
    return gloria_model

def build_gloria_dqn_wo_self_atten_wo_add_model(cfg):
    gloria_model = models.gloria_model_dqn_wo_self_atten_wo_add.GLoRIADQNWOSAWOADD(cfg)
    return gloria_model

def build_gloria_dqn_self_atten_local_model(cfg):
    gloria_model = models.gloria_model_dqn_self_atten_local.GLoRIADQNSAL(cfg)
    return gloria_model

def build_gloria_dqn_self_atten_double_model(cfg):
    gloria_model = models.gloria_model_dqn_self_atten_double.GLoRIADQNSAD(cfg)
    return gloria_model

def build_gloria_dqn_random_mask_model(cfg):
    gloria_model = models.gloria_model_dqn_random_mask.GLoRIADQNRM(cfg)
    return gloria_model

def build_gloria_dqn_local_model(cfg):
    gloria_model = models.gloria_model_dqn_local.GLoRIADQNLOCAL(cfg)
    return gloria_model

def build_gloria_dqn_atten_model(cfg):
    gloria_model = models.gloria_model_dqn_atten.GLoRIADQNATT(cfg)
    return gloria_model

def build_gloria_dqn_atten_local_model(cfg):
    gloria_model = models.gloria_model_dqn_atten_local.GLoRIAIINATTL(cfg)
    return gloria_model

def build_gloria_dqn_atten_local_proj_model(cfg):
    gloria_model = models.gloria_model_car_proj.CLIPCARProj(cfg)
    return gloria_model


def build_gloria_dqn_atten_local_global_model(cfg):
    gloria_model = models.gloria_model_dqn_atten_local_global.GLoRIAIINATTLG(cfg)
    return gloria_model


def build_gloria_dqn_double_model(cfg):
    gloria_model = models.gloria_model_dqn_double.GLoRIADQND(cfg)
    return gloria_model

def build_gloria_dqn_llm_model(cfg):
    gloria_model = models.gloria_model_dqn_iin.GLoRIADQNIIN(cfg)
    return gloria_model

def build_gloria_iin_model(cfg):
    gloria_model = models.gloria_model_iin.GLoRIAIIN(cfg)
    return gloria_model

def build_gloria_iins_model(cfg):
    gloria_model = models.gloria_model_iinsahare.GLoRIAIINS(cfg)
    return gloria_model


def build_gloria_iins_wo_self_atten_model(cfg):
    gloria_model = models.gloria_model_iinshare_wo_self_atten.GLoRIAIINSWOSA(cfg)
    return gloria_model


def build_gloria_dqngl_model(cfg):
    gloria_model = models.gloria_model_dqn_gl.GLoRIADQNGL(cfg)
    return gloria_model

def build_gloria_tripple_model(cfg):
    gloria_model = models.gloria_tripple_model.GLoRIATripple(cfg)
    return gloria_model

def build_universal_CRD_model(cfg):
    gloria_model = models.universal_cxr_model.UniversalCXR(cfg)
    return gloria_model

def build_universal_CRDP_model(cfg):
    gloria_model = models.universal_cxr_prompts_model.UniversalCXRP(cfg)
    return gloria_model

def build_universal_CRDPS_model(cfg):
    gloria_model = models.universal_cxr_prompts_model_soft.UniversalCXRPS(cfg)
    return gloria_model


def build_universal_CRDCP_model(cfg):
    gloria_model = models.universal_cxr_cat_prompts_model.UniversalCXRCP(cfg)
    return gloria_model

def build_universal_CRDCPDALI_model(cfg):
    gloria_model = models.universal_cxr_cat_prompts_DALI_model.UniversalCXRCPDALI(cfg)
    return gloria_model



def build_universal_CRDCat_model(cfg):
    gloria_model = models.universal_cxr_cat_model.UniversalCXRCAT(cfg)
    return gloria_model

def build_universal_RAM_model(cfg):
    gloria_model = models.universal_ram_model.UniversalRAM(cfg)
    return gloria_model


def build_query(cfg):
    df = pd.read_csv(CHEXPERT_MASTER_LABRL_CSV_V5)
    key = df.keys()[2:]
    query = list(key)
    # query = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    return query

def build_report_query(cfg):
    df = pd.read_csv(CHEXPERT_MASTER_CSV_V5)
    key = df.keys()[5:17]
    query = list(key)
    
    return query

def build_prompts(cfg):
    prompts = [   
    "{}",
    "A disease of {}",
    "This patient has a disease of {}",
    "Finding suggesting {}",
    "The primary symptom of {}",
    "{} diagnosis",
    "Symptoms related to {}",
    "Treatment for {}",
    "Common causes of {}",
    "Prevention of {}",
    "{} management",
    "Prognosis of {}",
    "Living with {}"
    ]
    return prompts


def build_fusion_module(cfg):
    fusion = models.fusion_module.Fusion(cfg)
    return fusion

def build_dqn_module(cfg):
    fusion = models.dqn.TQN_Model(cfg)
    return fusion

def build_dqn_M3AE_module(cfg):
    fusion = models.dqn_M3AE.DQN_M3AE(cfg)
    return fusion

def build_dqn_wo_self_atten_module(cfg):
    fusion = models.dqn_wo_self_atten.TQN_Model(cfg)
    return fusion

def build_dqn_wo_self_atten_wo_head_module(cfg):
    fusion = models.dqn_wo_self_atten_wo_head.TQN_Model(cfg)
    return fusion


def build_dqn_wo_self_atten_mlp_module(cfg):
    fusion = models.dqn_wo_self_atten_mlp.TQN_Model(cfg)
    return fusion
def build_dqn_wo_self_atten_wo_add_module(cfg):
    fusion = models.dqn_wo_self_atten_wo_add.TQN_Model(cfg)
    return fusion

def build_dqn_self_atten_local_module(cfg):
    fusion = models.dqn_self_atten_local.TQN_Model(cfg)
    return fusion

def build_dqn_self_atten_double_module(cfg):
    fusion = models.dqn_self_atten_double.TQN_Model(cfg)
    return fusion


def build_dqn_local_module(cfg):
    fusion = models.dqn_local.TQNLocal_Model(cfg)
    return fusion

def build_iin_module(cfg):
    fusion = models.iin.IIN_Model(cfg)
    return fusion

def build_iin_share_module(cfg):
    fusion = models.iin_share.IINS_Model(cfg)
    return fusion


def build_iin_share_wo_self_atten_module(cfg):
    fusion = models.iin_share_wo_self_atten.IINS_Model(cfg)
    return fusion

def build_ram_module(cfg):
    ram = models.retrieval_augmented_module.retrieval_augmentend(cfg)
    return ram

def build_ram_extract_module():
    extract = models.mrm_pretrain_model.mrm_vit_b16()
    return extract

def build_gloria_from_ckpt(ckpt):

    ckpt = torch.load(ckpt)
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    gloria_model = build_gloria_model(cfg)
    gloria_model.load_state_dict(ckpt_dict)

    return gloria_model


def build_img_model(cfg):
    image_model = models.IMAGE_MODELS[cfg.phase.lower()]
    return image_model(cfg)


def build_text_model(cfg):
    return models.text_model.BertEncoder(cfg)


def build_dictionary_model(cfg):
    return models.dictionarylearning.SharedDictionaryLearning(cfg)

def build_memory_model_global(cfg):
    return models.memory_bank.MemoryBank(cfg)

def build_Refiner_model_global(memory_bank):
    return models.memory_bank.TextFeatureRefiner(memory_bank)

# def build_image_projector_model(cfg):
#     return models.image_model.ImageProjector(cfg)

def build_text_proj_model(cfg):
    return models.text_model.BertEncoderWithProj(cfg)

def build_text_proj_attention_model(cfg):
    return models.text_model.BertEncoderWithProjAtten(cfg)

def build_text_simple_model(cfg):
    return models.text_model.BertEncoderSimple(cfg)

def build_gpt_model(cfg):
    return models.gpt_model.EmbeddingFusing(cfg)


def build_optimizer(cfg, lr, model):

    # get params for optimization
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    # define optimizers
    if cfg.train.optimizer.name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, momentum=cfg.train.optimizer.momentum, weight_decay=cfg.train.optimizer.weight_decay
        )
    elif cfg.train.optimizer.name == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )
    elif cfg.train.optimizer.name == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=cfg.train.optimizer.weight_decay
        )
    

def build_scheduler(cfg, optimizer, dm=None):

    if cfg.train.scheduler.name == "warmup":

        def lambda_lr(epoch):
            if epoch <= 3:
                return 0.001 + epoch * 0.003
            if epoch >= 22:
                return 0.01 * (1 - epoch / 200.0) ** 0.9
            return 0.01

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    elif cfg.train.scheduler.name == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif cfg.train.scheduler.name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
    elif cfg.train.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    elif cfg.train.scheduler.name == "stepbyepoch":
        def lambda_lr(epoch):
            if epoch > 10:
                return 0.5
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    else:
        scheduler = None

    if cfg.lightning.trainer.val_check_interval is not None:
        cfg.train.scheduler.interval = "step"
        num_iter = len(dm.train_dataloader().dataset)
        if type(cfg.lightning.trainer.val_check_interval) == float:
            frequency = int(num_iter * cfg.lightning.trainer.val_check_interval)
            cfg.train.scheduler.frequency = frequency
        else:
            cfg.train.scheduler.frequency = cfg.lightning.trainer.val_check_interval

    scheduler = {
        "scheduler": scheduler,
        "monitor": cfg.train.scheduler.monitor,
        "interval": cfg.train.scheduler.interval,
        "frequency": cfg.train.scheduler.frequency,
    }

    return scheduler


def build_loss(cfg):

    if cfg.train.loss_fn.type == "DiceLoss":
        return loss.segmentation_loss.DiceLoss()
    elif cfg.train.loss_fn.type == "FocalLoss":
        return loss.segmentation_loss.FocalLoss()
    elif cfg.train.loss_fn.type == "MixedLoss":
        return loss.segmentation_loss.MixedLoss(alpha=cfg.train.loss_fn.alpha)
    elif cfg.train.loss_fn.type == "BCE":
        if cfg.train.loss_fn.class_weights is not None:
            weight = torch.Tensor(cfg.train.loss_fn.class_weights)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn
    else:
        raise NotImplementedError(f"{cfg.train.loss_fn} not implemented yet")


def build_transformation(cfg, split):

    t = []
    if split == "train":

        if cfg.transforms.random_crop is not None:
            t.append(transforms.RandomCrop(cfg.transforms.random_crop.crop_size))

        if cfg.transforms.random_horizontal_flip is not None:
            t.append(
                transforms.RandomHorizontalFlip(p=cfg.transforms.random_horizontal_flip)
            )

        if cfg.transforms.random_affine is not None:
            t.append(
                transforms.RandomAffine(
                    cfg.transforms.random_affine.degrees,
                    translate=[*cfg.transforms.random_affine.translate],
                    scale=[*cfg.transforms.random_affine.scale],
                )
            )

        if cfg.transforms.color_jitter is not None:
            t.append(
                transforms.ColorJitter(
                    brightness=[*cfg.transforms.color_jitter.bightness],
                    contrast=[*cfg.transforms.color_jitter.contrast],
                )
            )
    else:
        if cfg.transforms.random_crop is not None:
            t.append(transforms.CenterCrop(cfg.transforms.random_crop.crop_size))

    t.append(transforms.ToTensor())
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif cfg.transforms.norm == "half":
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        elif cfg.transforms.norm == "CXR_MAE":
            t.append(transforms.Normalize(mean=[0.4978], std=[0.2449]))
        elif cfg.transforms.norm == "CLIP":
            t.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))
        else:
            raise NotImplementedError("Normaliation method not implemented")

    return transforms.Compose(t)


def build_transformation_3D(cfg, split):
    t = []
    
    # 将所有图像转换为张量
    t.append(ToTensor())
    t.append(AddChannel())

    if split == "train":
        # 随机弹性形变，模拟不规则形状变化
        if cfg.transforms.random_elastic is not None:
            t.append(Rand3DElastic(
                prob=cfg.transforms.random_elastic.prob, sigma_range=cfg.transforms.random_elastic.sigma_range,
                magnitude_range=cfg.transforms.random_elastic.magnitude_range,
                spatial_size=cfg.transforms.random_elastic.spatial_size
            ))

        # 随机翻转，适用于3D图像
        if cfg.transforms.random_flip is not None:
            t.append(RandFlip(
                prob=cfg.transforms.random_flip.prob, 
                spatial_axis=cfg.transforms.random_flip.spatial_axis
            ))

        # 随机仿射变换，包括旋转和缩放
        if cfg.transforms.random_affine is not None:
            t.append(RandAffine(
                prob=cfg.transforms.random_affine.prob, rotate_range=cfg.transforms.random_affine.rotate_range,
                translate_range=cfg.transforms.random_affine.translate_range,
                scale_range=cfg.transforms.random_affine.scale_range,
                padding_mode="border"
            ))

     # 随机空间裁剪
        if cfg.transforms.random_crop is not None:
            t.append(CenterSpatialCrop(
                roi_size=cfg.transforms.random_crop.crop_size
            ))

    else:
        # 测试时使用确定性的裁剪，如中心裁剪
        if cfg.transforms.random_crop is not None:
            t.append(CenterSpatialCrop(
                roi_size=cfg.transforms.random_crop.crop_size
            ))


    # 正则化处理
    # since the data is CT image, moreover, is has be save to jpg, we use the mean and std of the CT image
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm.type == "standard":
            t.append(NormalizeIntensity(
                subtrahend=cfg.transforms.norm.subtrahend, 
                divisor=cfg.transforms.norm.divisor
            ))
        elif cfg.transforms.norm.type == "scale":
            t.append(ScaleIntensity(
                minv=cfg.transforms.norm.minv, 
                maxv=cfg.transforms.norm.maxv
            ))
        else:
            raise NotImplementedError("Normalization method not implemented")

    return Compose(t)

def build_transformation_3D_M3D(cfg, split):
    t = []

    # 将所有图像转换为张量
    t.append(ToTensor(dtype=torch.float))
    t.append(AddChannel())

    if split == "train":
        # 随机裁剪3D图像到固定大小
        if cfg.transforms.rand_spatial_crop is not None:
            t.append(RandSpatialCrop(roi_size=cfg.transforms.rand_spatial_crop.roi_size, random_size=False))
            # t.append(CenterSpatialCrop(roi_size=cfg.transforms.rand_spatial_crop.roi_size))

        # 随机旋转90度，适用于3D图像
        if cfg.transforms.random_rotate90 is not None:
            t.append(RandRotate90(
                prob=cfg.transforms.random_rotate90.prob, 
                spatial_axes=cfg.transforms.random_rotate90.spatial_axes
            ))

        # 随机翻转
        if cfg.transforms.random_flip is not None:
            if cfg.transforms.random_flip.axis_0_prob is not None:
                t.append(RandFlip(prob=cfg.transforms.random_flip.axis_0_prob, spatial_axis=0))
            if cfg.transforms.random_flip.axis_1_prob is not None:
                t.append(RandFlip(prob=cfg.transforms.random_flip.axis_1_prob, spatial_axis=1))
            if cfg.transforms.random_flip.axis_2_prob is not None:
                t.append(RandFlip(prob=cfg.transforms.random_flip.axis_2_prob, spatial_axis=2))

        # 随机强度缩放
        if cfg.transforms.random_scale_intensity is not None:
            t.append(RandScaleIntensity(
                factors=cfg.transforms.random_scale_intensity.factors, 
                prob=cfg.transforms.random_scale_intensity.prob
            ))

        # 随机强度偏移
        if cfg.transforms.random_shift_intensity is not None:
            t.append(RandShiftIntensity(
                offsets=cfg.transforms.random_shift_intensity.offsets, 
                prob=cfg.transforms.random_shift_intensity.prob
            ))

    # else:
    #     # 测试时使用确定性的大小调整
    #     if cfg.transforms.rand_spatial_crop is not None:
    #         t.append(CenterSpatialCrop(roi_size=cfg.transforms.rand_spatial_crop.roi_size))

    # 强度范围归一化
    if cfg.transforms.scale_intensity_range is not None:
        t.append(ScaleIntensityRange(
            a_min=cfg.transforms.scale_intensity_range.a_min, 
            a_max=cfg.transforms.scale_intensity_range.a_max,
            b_min=cfg.transforms.scale_intensity_range.b_min, 
            b_max=cfg.transforms.scale_intensity_range.b_max, 
            clip=cfg.transforms.scale_intensity_range.clip
        ))

    # 正则化处理
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm.type == "standard":
            t.append(NormalizeIntensity(
                subtrahend=cfg.transforms.norm.subtrahend, 
                divisor=cfg.transforms.norm.divisor
            ))
        else:
            raise NotImplementedError("Normalization method not implemented")

    return Compose(t)

def build_transformation_3D_M3D_with_mask(cfg, split):
    keys = ["image", "mask"]
    t = []

    # 载入图像和mask
    t.append(ToTensord(keys=keys, dtype=torch.float))
    # 添加通道维度
    t.append(AddChanneld(keys=keys))

    if split == "train":
        # 随机裁剪3D图像和应对的mask到固定大小
        if cfg.transforms.rand_spatial_crop is not None:
            t.append(RandSpatialCropd(keys=keys, roi_size=cfg.transforms.rand_spatial_crop.roi_size, random_size=False))

        # 随机旋转90度，适用3D图像和mask
        if cfg.transforms.random_rotate90 is not None:
            t.append(RandRotate90d(
                keys=keys,
                prob=cfg.transforms.random_rotate90.prob,
                spatial_axes=cfg.transforms.random_rotate90.spatial_axes
            ))

        # 随机翻转图像和mask
        if cfg.transforms.random_flip is not None:
            if cfg.transforms.random_flip.axis_0_prob is not None:
                t.append(RandFlipd(keys=keys, prob=cfg.transforms.random_flip.axis_0_prob, spatial_axis=0))
            if cfg.transforms.random_flip.axis_1_prob is not None:
                t.append(RandFlipd(keys=keys, prob=cfg.transforms.random_flip.axis_1_prob, spatial_axis=1))
            if cfg.transforms.random_flip.axis_2_prob is not None:
                t.append(RandFlipd(keys=keys, prob=cfg.transforms.random_flip.axis_2_prob, spatial_axis=2))

        # 随机强度缩放（只对图像）
        if cfg.transforms.random_scale_intensity is not None:
            t.append(RandScaleIntensityd(
                keys=["image"],
                factors=cfg.transforms.random_scale_intensity.factors,
                prob=cfg.transforms.random_scale_intensity.prob
            ))

        # 随机强度偏移（只对图像）
        if cfg.transforms.random_shift_intensity is not None:
            t.append(RandShiftIntensityd(
                keys=["image"],
                offsets=cfg.transforms.random_shift_intensity.offsets,
                prob=cfg.transforms.random_shift_intensity.prob
            ))

    else:
        # 测试时使用确定性的大小调整
        if cfg.transforms.rand_spatial_crop is not None:
            t.append(CenterSpatialCropd(keys=keys, roi_size=cfg.transforms.rand_spatial_crop.roi_size))

    # 强度范围平比（只对图像）
    if cfg.transforms.scale_intensity_range is not None:
        t.append(ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg.transforms.scale_intensity_range.a_min,
            a_max=cfg.transforms.scale_intensity_range.a_max,
            b_min=cfg.transforms.scale_intensity_range.b_min,
            b_max=cfg.transforms.scale_intensity_range.b_max,
            clip=cfg.transforms.scale_intensity_range.clip
        ))

    # 正则化处理（只对图像）
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm.type == "standard":
            t.append(NormalizeIntensityd(
                keys=["image"],
                subtrahend=cfg.transforms.norm.subtrahend,
                divisor=cfg.transforms.norm.divisor
            ))
        else:
            raise NotImplementedError("Normalization method not implemented")

    # 转换为库置式数据
    t.append(ToTensord(keys=keys))

    return Compose(t)

from torchvision.transforms.functional import InterpolationMode
def build_ram_transformation(cfg):
    transform_test = transforms.Compose([
    # transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
    # transforms.RandomHorizontalFlip(),
    # transforms.ToPILImage(),
    # transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    # transforms.Grayscale(num_output_channels=3),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.4978], std=[0.2449])])
    return transform_test