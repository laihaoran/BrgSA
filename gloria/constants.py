from pathlib import Path

# CheXpert constants
CHEXPERT_DATA_DIR = Path("/data2/haoranlai/Project/BrgSA/Dataset")
CHEXPERT_SPLIT_DATA_DIR = Path("/haoranlai/Project/BrgSA/Dataset/CheXpert")
CHEXPERT_ORIGINAL_TRAIN_CSV = CHEXPERT_SPLIT_DATA_DIR / "train.csv"
CHEXPERT_TRAIN_CSV = CHEXPERT_SPLIT_DATA_DIR / "train_split.csv"  # train split from train.csv
CHEXPERT_VALID_CSV = CHEXPERT_SPLIT_DATA_DIR / "valid_split.csv"  # valid split from train.csv
CHEXPERT_TEST_CSV = (
    CHEXPERT_SPLIT_DATA_DIR / "valid.csv"
)  # using validation set as test set (test set label hidden)
CHEXPERT_MASTER_CSV = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-chexpertformat.csv"
)  # contains patient information, not PHI conplient

CHEST3D_MASTER_CSV = (
    CHEXPERT_DATA_DIR / "CTRG-Chest-548K-reports.csv"
)  # contains patient information, not PHI conplient

CT_RATE3D_MASTER_CSV = (
    CHEXPERT_DATA_DIR / "CT-RATE-Chest-25K_report.csv"
)  # contains patient information, not PHI conplient



CT_RATE3D_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report.json"
)  # contains patient information, not PHI conplient


CT_RATE3D_ORGAN_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_organ.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_ORGAN_ALL_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_organ_all.json"
)  # contains patient information, not PHI conplient


CT_RATE3D_ORGAN_WLABEL_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_organ_label.json"
)  # contains patient information, not PHI conplient


CT_RATE3D_WLABEL_CLASS1_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_label_class1.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_WLABEL_CLASS0_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_label_class0.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_ORGAN_WLABEL_ALL_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_organ_label_all.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_ORGAN_WLGPTM_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_organ_GPT4Modify.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_ORGAN_WLGPTM_ALL_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_organ_GPT4Modify_all.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_WLABEL_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_label.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_WLABEL01_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_label_0.1.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_WLABEL02_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_label_0.2.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_WLGPT_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_gpt4.json"
)  # contains patient information, not PHI conplient

CT_RATE3D_WLGPTM_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_gpt4modify.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLLLAMA3_8B_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_llama3.1_8b.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLQWEN25_7B_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_qwen2.5_7b.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient


CT_RATE3D_WLMEDGEMMA_27B_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_medgemma-27b-text-it.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLMEDGEMMA_4B_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_medgemma_4b.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLLINSHU_7B_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_linshu_7b.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLDEEPSEEK_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_deepseek.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLMEDGEMMA_27B_REWRITE_SHORT_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_medgemma-27b-text-rewrite_short.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient


CT_RATE3D_WLMEDGEMMA_27B_REWRITE_LONG_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_medgemma-27b-text-rewrite_long.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient



CT_RATE3D_WLDEEPSEEK_REWRITE_SHORT_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_deepseek-rewrite_short.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient


CT_RATE3D_WLDEEPSEEK_REWRITE_LONG_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_deepseek-rewrite_long.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLDEEPSEEK_REWRITE_LACK_0_4_SENTENCE_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_deepseek-lack_0.4_sentence.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLDEEPSEEK_REWRITE_LACK_0_2_SENTENCE_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_deepseek-lack_0.2_sentence.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLDEEPSEEK_REWRITE_GRAMM_ERROR_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_deepseek-gramm_error.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient



CT_RATE3D_WLDEEPSEEK_REWRITE_LACK_0_1_SENTENCE_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_deepseek-lack_0.1_sentence.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient


CT_RATE3D_WLDEEPSEEK_REWRITE_LACK_0_3_SENTENCE_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_deepseek-lack_0.3_sentence.json"   #use for my brgsa 3d
)  # contains patient information, not PHI conplient

CT_RATE3D_WLGPTM_NOBRAIN_MASTER_JSON = (
    CHEXPERT_DATA_DIR / "ctrate_train_unique_report_gpt4modify_nobrain.json"
)  # contains patient information, not PHI conplient



CHEXPERT_MASTER_LABRL_CSV = (
    CHEXPERT_DATA_DIR / "all-data-cxr-disease.csv"
)  # contains patient information, not PHI conplient

CHEXPERT_MASTER_CSV_V2 = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-chexpertformat-v2.csv"
)  # contains patient information, not PHI conplient
CHEXPERT_MASTER_LABRL_CSV_V2 = (
    CHEXPERT_DATA_DIR / "all-data-cxr-disease_v2.csv"
)

CHEXPERT_MASTER_CSV_V3 = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-chexpertformat-v3.csv"
)  # contains patient information, not PHI conplient
CHEXPERT_MASTER_LABRL_CSV_V3 = (
    CHEXPERT_DATA_DIR / "all-data-cxr-disease.csv"
)


CHEXPERT_MASTER_CSV_V4 = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-chexpertformat.csv"
)  # contains patient information, not PHI conplient
CHEXPERT_MASTER_LABRL_CSV_V4 = (
    CHEXPERT_DATA_DIR / "all-data-cxr-disease-v4.csv"
)  # contains patient information, not PHI conplient


CHEXPERT_MASTER_CSV_V5 = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-chexpertformat-v5.csv"
)  # contains patient information, not PHI conplient
CHEXPERT_MASTER_LABRL_CSV_V5 = (
    CHEXPERT_DATA_DIR / "all-data-cxr-disease-v5.csv"
)  # contains patient information, not PHI conplient


CHEXPERT_MASTER_CSV_V6 = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-chexpertformat.csv"
)  # contains patient information, not PHI conplient
CHEXPERT_MASTER_LABRL_CSV_V6 = (
    CHEXPERT_DATA_DIR / "all-data-cxr-disease-v6.csv"
)  # contains patient information, not PHI conplient


CHEXPERT_MASTER_CSV_V7 = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report_v1-chexpertformat.csv"
)  # contains patient information, not PHI conplient


CHEXPERT_MASTER_CSV_XH = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-xinhuo-chexpertformat.csv"
)  # contains patient information, not PHI conplient

CHEXPERT_MASTER_CSV_XH_ADO = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-xinhuo-ADO-chexpertformat.csv"
)  # contains patient information, not PHI conplient


CHEXPERT_MASTER_CSV_XH_keyword = (
    CHEXPERT_DATA_DIR / "mimic-cxr-label-LLM_report-xinhuo-withkeywordchexpertformat.csv"
)  # contains patient information, not PHI conplient


CHEXPERT_MASTER_CSV_XH_GPT4TEMPLATE = (
    CHEXPERT_DATA_DIR / "mimic-cxr-XH_GPT4template_report-chexpertformat.csv"
)  # contains patient information, not PHI conplient
### add extract label dataset
# (1) nih training set
# (2) vinBig training set
# (3) Pneumonia
#（4）

### add extract reports dataset
# (1) padchest training set
###


CHEXPERT_TRAIN_DIR = CHEXPERT_DATA_DIR / "train"
CHEXPERT_TEST_DIR = CHEXPERT_DATA_DIR / "valid"
CHEXPERT_5x200 = CHEXPERT_DATA_DIR / "chexpert_5x200_newpath.csv"
# LTNIHBalance = CHEXPERT_DATA_DIR / "nih-cxr-lt_single-balance_image.csv


CHEXPERT_VALID_NUM = 5000
CHEXPERT_VIEW_COL = "Frontal/Lateral"
CHEXPERT_PATH_COL = "Path"
CHEXPERT_SPLIT_COL = "Split"
CHEXPERT_REPORT_COL = "Report Impression"
CHEXPERT_LLM_REPORT_COL = "LLM Report Impression"
CHEXPERT_XH_REPORT_COL = "xinhuo"
CHEXPERT_LLM_REPORT_V1_COL = "LLM Report v1 Impression"
CHEXPERT_DataFlag_COL = "Data Flag"
CHEXPERT_RAMINDEX_COL = "Index"
CHEXPERT_Original_VIEW_COL = "OriginalView"

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# baseed on original chexpert paper
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

# SIIM Pneumothorax
PNEUMOTHORAX_DATA_DIR = Path("/data4/siim")
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train.csv"
PNEUMOTHORAX_VALID_CSV = PNEUMOTHORAX_DATA_DIR / "valid.csv"
PNEUMOTHORAX_TEST_CSV = PNEUMOTHORAX_DATA_DIR / "test.csv"
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "dicom-images-train"
PNEUMOTHORAX_IMG_SIZE = 1024
PNEUMOTHORAX_TRAIN_PCT = 0.7

# RSNA Pneumonia
PNEUMONIA_DATA_DIR = Path("/data4/rsna_pneumonia/")  # aimi0/rsna_pneumonia')
PNEUMONIA_ORIGINAL_TRAIN_CSV = PNEUMONIA_DATA_DIR / "stage_2_train_labels.csv"
PNEUMONIA_TRAIN_CSV = PNEUMONIA_DATA_DIR / "train.csv"
PNEUMONIA_VALID_CSV = PNEUMONIA_DATA_DIR / "val.csv"
PNEUMONIA_TEST_CSV = PNEUMONIA_DATA_DIR / "test.csv"
PNEUMONIA_IMG_DIR = PNEUMONIA_DATA_DIR / "stage_2_train_images"
PNEUMONIA_TRAIN_PCT = 0.7


CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}
