from .pretrain_model import PretrainModel
from .pretrain_model_global import PretrainGlobalModel
from .pretrain_model_dqn import PretrainDQNModel
from .classification_model import ClassificationModel
from .classification_model_best import ClassificationBestModel
from .segmentation_model import SegmentationModel
from .pretrain_model_tripple import PretrainTrippleModel
from .universal_cxr_reports_diseases import PretrainUniversalCRDModel
from .universal_cxr_reports_ram import PretrainUniversalRAMModel
from .universal_cxr_cat_reports_diseases import PretrainUniversalCRDCATModel
from .universal_cxr_reports_diseases_prompts import PretrainUniversalCRDPModel
from .universal_cxr_reports_diseases_cat_prompts import PretrainUniversalCRDCPModel
from .universal_cxr_reports_diseases_cat_prompts_DALI import PretrainUniversalCRDCPDALIModel
from .universal_cxr_reports_diseases_prompts_soft import PretrainUniversalCRDPSModel
from .pretrain_model_dqn_gl import PretrainDQNGLModel
from .pretrain_model_iin import PretrainIINModel
from .pretrain_model_dqn_iin import PretrainDQNIINModel
from .pretrain_model_iins import PretrainIINSModel
from .pretrain_model_dqn_double import PretrainDQNDModel
from .pretrain_model_dqn_local import PretrainDQNLOCALModel
from .pretrain_model_dqn_atten import PretrainDQNATTModel
from .pretrain_model_dqn_atten_local import PretrainDQNIINLOCALModel
from .pretrain_model_dqn_atten_local_global import PretrainDQNIINLOCALGLOBALModel
from .pretrain_model_dqn_random_mask import PretrainDQNRMModel
from .pretrain_model_dqn_wo_self_atten import PretrainDQNWOSAModel
from .pretrain_model_dqn_self_atten_local import PretrainDQNSALModel
from .pretrain_model_dqn_self_atten_double import PretrainDQNSADModel
from .pretrain_model_dqn_wo_self_atten_wo_add import PretrainDQNWOSAWOADDModel
from .pretrain_model_iins_wo_self_atten import PretrainIINSWOSAModel
from .pretrain_model_dqn_M3AE import PretrainDQNM3AEModel
from .pretrain_model_dqn_wo_self_atten_gl import PretrainDQNWOSAGLModel 
from .pretrain_model_dqn_wo_self_atten_gl_proj import PretrainDQNWOSAGLPModel
from .pretrain_model_dqn_wo_self_atten_global import PretrainDQNWOSAGModel
from .pretrain_model_dqn_wo_self_atten_mlp_gl import PretrainDQNWOSAMLPGLModel
from .pretrain_model_dqn_wo_self_atten_gl_cos_proj import PretrainDQNWOSAGLWOHProjCOSModel
from .pretrain_model_dqn_wo_self_atten_gl_clip_openai import PretrainDQNWOSAMLPGLCLIPOPENAIModel
from .pretrain_model_dqn_wo_self_atten_mlp_gl_aug import PretrainDQNWOSAMLPGLAUGModel
from .pretrain_model_clip import PretrainCLIPModel
from .pretrain_model_clip_proj import PretrainCLIPProjModel
from .pretrain_model_dqn_atten_local_proj import PretrainDQNIINLOCALProjModel
from .pretrain_model_clip_proj_global_local import PretrainCLIPProjGLModel
from .pretrain_model_clip_proj_dict import PretrainCLIPProjDictModel
from .pretrain_model_global_dict import PretrainGlobalDictModel
from .pretrain_model_clip_proj_dict_add import PretrainCLIPProjDictAddModel
from .pretrain_model_clip_proj_dict_gl import PretrainCLIPProjDictGLModel
from .pretrain_model_clip_proj_organ import PretrainCLIPProjOrganModel
from .pretrain_model_clip_proj_dqn import PretrainCLIPProjDQNModel
from .pretrain_model_clip_proj_dict_organ import PretrainCLIPProjDictOrganModel
from .pretrain_model_clip_proj_organ_block import PretrainCLIPProjOrganBModel
from .pretrain_model_clip_proj_dict_organ_cls import PretrainCLIPProjDictOrganCLSModel
from .pretrain_model_clip_proj_dict_organ_one import PretrainCLIPProjDictOrganOneModel
from .pretrain_model_clip_proj_dict_organ_large import PretrainCLIPProjDictOrganLargeModel
from .pretrain_model_clip_proj_dict_organ_one_large import PretrainCLIPProjDictOrganOneLargeModel
from .pretrain_model_clip_proj_dict_organ_neg import PretrainCLIPProjDictOrganNegModel
from .pretrain_model_clip_proj_dict_organ_neg_fast import PretrainCLIPProjDictOrganNegFastModel
from .pretrain_model_clip_proj_dict_organ_fast import PretrainCLIPProjDictOrganFastModel
from .pretrain_model_clip_proj_kmb import PretrainCLIPProjKMBModel
from .pretrain_model_clip_proj_memory import PretrainCLIPProjMemModel
from .pretrain_model_mse import PretrainMSEModel

LIGHTNING_MODULES = {
    "pretrain": PretrainModel,
    "clip3d": PretrainCLIPModel,
    "clip3d_proj": PretrainCLIPProjModel,
    "clip3d_proj_organ": PretrainCLIPProjOrganModel,
    "clip3d_proj_organ_block": PretrainCLIPProjOrganBModel,
    "clip3d_proj_dict_organ": PretrainCLIPProjDictOrganModel,
    "clip3d_proj_dict_organ_fast": PretrainCLIPProjDictOrganFastModel,
    "clip3d_proj_dict_organ_neg": PretrainCLIPProjDictOrganNegModel,
    "clip3d_proj_dict_organ_neg_fast": PretrainCLIPProjDictOrganNegFastModel,
    "clip3d_proj_dict_organ_large": PretrainCLIPProjDictOrganLargeModel,
    "clip3d_proj_dict_organ_one": PretrainCLIPProjDictOrganOneModel,
    "clip3d_proj_dict_organ_one_large": PretrainCLIPProjDictOrganOneLargeModel,
    "clip3d_proj_dict_organ_cls": PretrainCLIPProjDictOrganCLSModel,
    "clip3d_proj_dqn": PretrainCLIPProjDQNModel,
    "clip3d_proj_dict": PretrainCLIPProjDictModel,
    "clip3d_proj_kmb": PretrainCLIPProjKMBModel,
    "clip3d_proj_memory": PretrainCLIPProjMemModel,
    "clip3d_proj_dict_gl": PretrainCLIPProjDictGLModel,
    "clip3d_proj_dict_add": PretrainCLIPProjDictAddModel,
    "clip3d_proj_global_local": PretrainCLIPProjGLModel,
    "car3d_proj": PretrainDQNIINLOCALProjModel,
    "vit2d_dict_mse": PretrainMSEModel,
    "pretrain_global":PretrainGlobalModel,
    "pretrain_global_dict": PretrainGlobalDictModel,
    "pretrain_dqn": PretrainDQNModel,
    "pretrain_llm_dqn_random_mask": PretrainDQNRMModel,
    "pretrain_llm": PretrainModel,
    "pretrain_llm_v1": PretrainModel,
    "pretrain_llm_dqn": PretrainDQNModel,
    "pretrain_llm_dqn_m3ae": PretrainDQNM3AEModel,
    "pretrain_llm_dqn_local": PretrainDQNLOCALModel,
    "pretrain_llm_dqn_atten": PretrainDQNATTModel,
    "pretrain_llm_dqn_atten_local": PretrainDQNIINLOCALModel,
    "pretrain_llm_dqn_wo_self_atten": PretrainDQNWOSAModel,
    "pretrain_llm_dqn_wo_self_atten_gl": PretrainDQNWOSAGLModel,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl": PretrainDQNWOSAMLPGLModel,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl_aug": PretrainDQNWOSAMLPGLAUGModel,
    "pretrain_llm_dqn_wo_self_atten_mlp_gl_clip_openai": PretrainDQNWOSAMLPGLCLIPOPENAIModel,
    "pretrain_llm_dqn_wo_self_atten_gl_wo_head_cos_proj": PretrainDQNWOSAGLWOHProjCOSModel,
    "pretrain_llm_dqn_wo_self_atten_global": PretrainDQNWOSAGModel,
    "pretrain_llm_dqn_wo_self_atten_gl_proj": PretrainDQNWOSAGLPModel,
    "pretrain_llm_dqn_wo_self_atten_wo_add": PretrainDQNWOSAWOADDModel,
    "pretrain_llm_dqn_self_atten_local": PretrainDQNSALModel,
    "pretrain_llm_dqn_atten_local_global": PretrainDQNIINLOCALGLOBALModel,
    "pretrain_llm_dqn_self_atten_double": PretrainDQNSADModel,
    "pretrain_llm_dqn_double": PretrainDQNDModel,
    "pretrain_llm_iin": PretrainIINModel,
    "pretrain_llm_iins":PretrainIINSModel,
    "pretrain_llm_iins_wo_self_atten" : PretrainIINSWOSAModel,
    "pretrain_llm_dqn_iin": PretrainDQNIINModel,
    "pretrain_llm_v1_dqn":PretrainDQNModel,
    "pretrain_llm_dqn_gl": PretrainDQNGLModel,
    "pretrain_llm_dqn_large": PretrainDQNModel,
    "pretrain_llm_dqn_fast":PretrainDQNModel,
    "pretrain_tripple": PretrainTrippleModel,
    
    "classification": ClassificationModel,
    "classification_best": ClassificationBestModel,

    "segmentation": SegmentationModel,
    "universal_cxr_reports_diseases": PretrainUniversalCRDModel,
    "universal_cxr_reports_diseases_llm": PretrainUniversalCRDModel,
    "universal_cxr_reports_diseases_cat_prompts_llm":PretrainUniversalCRDCPModel,
    "universal_cxr_reports_diseases_v2": PretrainUniversalCRDModel,
    "universal_cxr_reports_diseases_v3": PretrainUniversalCRDModel,
    "universal_cxr_reports_diseases_llm_v2": PretrainUniversalCRDModel,
    "universal_cxr_reports_diseases_cat_prompts_llm_v2": PretrainUniversalCRDCPModel,
    "universal_cxr_reports_diseases_prompts": PretrainUniversalCRDPModel,
    "universal_cxr_reports_diseases_prompts_llm": PretrainUniversalCRDPModel,
    "universal_cxr_reports_diseases_prompts_llm_dqn": PretrainUniversalCRDPModel,
    "universal_cxr_reports_diseases_prompts_llm_v4": PretrainUniversalCRDPModel,
    "universal_cxr_reports_diseases_prompts_llm_v5": PretrainUniversalCRDPModel,
    "universal_cxr_reports_diseases_prompts_llm_v6": PretrainUniversalCRDPSModel,
    "universal_cxr_reports_diseases_cat_prompts": PretrainUniversalCRDCPModel,
     "universal_cxr_reports_diseases_cat_prompts_dali": PretrainUniversalCRDCPDALIModel,
    "universal_cxr_ram": PretrainUniversalRAMModel,
    "universal_cxr_cat_reports_diseases": PretrainUniversalCRDCATModel
}
