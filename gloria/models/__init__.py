from . import text_model
from . import bert_model
from . import vision_model
from . import unet
from . import dqn
from . import iin
from . import dqn_wo_self_atten
from . import dqn_wo_self_atten_mlp
from . import dqn_wo_self_atten_wo_add
from . import dqn_wo_self_atten_wo_head
from . import dqn_self_atten_local
from . import dqn_self_atten_double
from . import dqn_local
from . import iin_share
from . import dqn_M3AE
from . import iin_share_wo_self_atten
from . import gloria_model
from . import gloria_model_dqn
from . import retrival_model
from . import cnn_backbones
from . import gloria_tripple_model
from . import gpt_model
from . import universal_cxr_model
from . import fusion_module
from . import universal_ram_model
from . import retrieval_augmented_module
from . import mrm_pretrain_model
from . import universal_cxr_cat_model
from . import universal_cxr_prompts_model
from . import universal_cxr_cat_prompts_model
from . import universal_cxr_cat_prompts_DALI_model
from . import universal_cxr_prompts_model_soft
from . import gloria_model_dqn_gl
from . import gloria_model_global
from . import gloria_model_iin
from . import gloria_model_dqn_iin
from . import gloria_model_iinsahare
from . import gloria_model_dqn_double
from . import gloria_model_dqn_local
from . import gloria_model_dqn_atten
from . import gloria_model_dqn_atten_local
from . import gloria_model_dqn_atten_local_global
from . import gloria_model_dqn_random_mask
from . import gloria_model_dqn_wo_self_atten
from . import gloria_model_dqn_self_atten_local
from . import gloria_model_dqn_self_atten_double
from . import gloria_model_dqn_wo_self_atten_wo_add 
from . import gloria_model_iinshare_wo_self_atten
from . import gloria_model_dqn_copy_M3AE
from . import gloria_model_dqn_wo_self_atten_gl
from . import gloria_model_dqn_wo_self_atten_gl_proj
from . import gloria_model_dqn_wo_self_atten_global
from . import gloria_model_dqn_wo_self_atten_gl_mlp
from . import gloria_model_dqn_wo_self_atten_gl_cos_proj
from . import gloria_model_dqn_wo_self_atten_gl_mlp_clip_openai
from . import gloria_model_dqn_wo_self_atten_gl_mlp_aug
from . import stander_3D_vit
from . import stander_3D_vit_relate_pe
from . import glorai_model_clip
from . import gloria_model_clip_proj
from . import gloria_model_car_proj
from . import gloria_model_clip_proj_global_local
from . import gloria_model_clip_proj_dict
from . import dictionarylearning
from . import gloria_model_global_dict
from . import gloria_model_clip_proj_dict_add
from . import gloria_model_clip_proj_dict_gl
from . import gloria_model_clip_proj_organ
from . import gloria_model_clip_proj_dqn
from . import organ_self_attention
from . import gloria_model_clip_proj_dict_organ
from . import gloria_model_clip_proj_organ_block
from . import gloria_model_clip_proj_dict_gl
from . import multi_organ_token
from . import gloria_model_clip_proj_dict_organ_cls
from . import gloria_model_clip_proj_dict_organ_one
from . import gloria_model_clip_proj_dict_organ_large
from . import gloria_model_clip_proj_dict_organ_one_large
from . import gloria_model_clip_proj_dict_organ_neg
from . import gloria_model_clip_proj_dict_organ_neg_fast
from . import gloria_model_clip_proj_dict_organ_fast
from . import gloria_model_clip_proj_kmb
from . import gloria_model_clip_proj_memory
from . import memory_bank
from . import gloria_model_mse
from . import cnn_backbone_3d

IMAGE_MODELS = {
    "pretrain": vision_model.ImageEncoder,
    "clip3d": vision_model.ImageEncoder,
    "clip3d_proj": vision_model.ImageEncoder,
    "clip3d_proj_dqn": vision_model.ImageEncoder,
    "clip3d_proj_organ": vision_model.ImageEncoder,
    "clip3d_proj_organ_block": vision_model.ImageEncoder,
    "clip3d_proj_dict_organ": vision_model.ImageEncoder,
    "clip3d_proj_dict_organ_fast": vision_model.ImageEncoder,
    "clip3d_proj_dict_organ_large": vision_model.ImageEncoder,
    "clip3d_proj_dict_organ_one": vision_model.ImageEncoder,
    "clip3d_proj_dict_organ_one_large": vision_model.ImageEncoder,
    "clip3d_proj_dict_organ_cls": vision_model.ImageEncoder,
    "clip3d_proj_dict_organ_neg": vision_model.ImageEncoder,
    "clip3d_proj_dict_organ_neg_fast": vision_model.ImageEncoder,
    "clip3d_proj_dict": vision_model.ImageEncoder,
    "clip3d_proj_kmb": vision_model.ImageEncoder,
    "clip3d_proj_memory": vision_model.ImageEncoder,
    "clip3d_proj_dict_gl": vision_model.ImageEncoder,
    "clip3d_proj_dict_add": vision_model.ImageEncoder,
    "clip3d_conv_proj": vision_model.ImageEncoder,
    "clip3d_proj_global_local": vision_model.ImageEncoder,
    "vit2d_dict_mse": vision_model.ImageEncoder,
    "car3d_proj": vision_model.ImageEncoder,
    "pretrain_global": vision_model.ImageEncoder,
    "pretrain_global_dict": vision_model.ImageEncoder,
    "pretrain_dqn": vision_model.ImageEncoder,
    "pretrain_llm_dqn_random_mask": vision_model.ImageEncoder,
    "pretrain_llm": vision_model.ImageEncoder,
    "pretrain_llm_v1": vision_model.ImageEncoder,
    "pretrain_llm_dqn": vision_model.ImageEncoder,
    "pretrain_llm_dqn_m3ae": vision_model.ImageEncoder,
    "pretrain_llm_dqn_local": vision_model.ImageEncoder,
    "pretrain_llm_dqn_local_global": vision_model.ImageEncoder,
    "pretrain_llm_dqn_atten": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_global": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_gl": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl_clip_openai": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_gl_wo_head_cos_proj": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl_aug": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_gl_proj": vision_model.ImageEncoder,
    "pretrain_llm_dqn_wo_self_atten_wo_add": vision_model.ImageEncoder,
    "pretrain_llm_dqn_self_atten_local": vision_model.ImageEncoder,
    "pretrain_llm_dqn_self_atten_double": vision_model.ImageEncoder,
    "pretrain_llm_dqn_atten_local": vision_model.ImageEncoder,
    "pretrain_llm_dqn_atten_local_global": vision_model.ImageEncoder,
    "pretrain_llm_dqn_double": vision_model.ImageEncoder,
    "pretrain_llm_iin": vision_model.ImageEncoder,
    "pretrain_llm_iins": vision_model.ImageEncoder,
    "pretrain_llm_iins_wo_self_atten": vision_model.ImageEncoder,
    "pretrain_llm_dqn_iin": vision_model.ImageEncoder,
    "pretrain_llm_v1_dqn": vision_model.ImageEncoder,
    "pretrain_llm_dqn_gl": vision_model.ImageEncoder,
    "pretrain_llm_dqn_large": vision_model.ImageEncoder,
    "pretrain_llm_dqn_fast": vision_model.ImageEncoder,
    "pretrain_tripple": vision_model.ImageEncoder,
    "classification": vision_model.ImageClassifier,
    "segmentation": unet.ResnetUNet,
    "universal_cxr_reports_diseases": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_llm": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_cat_prompts_llm": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_v2": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_v3": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_llm_v2": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_cat_prompts_llm_v2": vision_model.ImageEncoder,
    "universal_cxr_ram": vision_model.ImageEncoder,
    "universal_cxr_cat_reports_diseases": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_prompts": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_prompts_llm": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_prompts_llm_v4": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_prompts_llm_v5": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_prompts_llm_v6": vision_model.ImageEncoder,
    "universal_cxr_reports_diseases_cat_prompts": vision_model.ImageEncoder,
     "universal_cxr_reports_diseases_cat_prompts_dali": vision_model.ImageEncoder,
}
