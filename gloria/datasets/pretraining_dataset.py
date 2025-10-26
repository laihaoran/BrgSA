import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random  # for one sentencese
import torch
import torch.utils.data as data
import json
from PIL import Image
from monai.data import CacheDataset
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer, BertTokenizer
from gloria.constants import *
import nibabel as nib
from skimage.transform import resize
import ipdb



class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class Multimodal3DPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEST3D_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_3D.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path))
        x = nii_img.get_fdata()
     
        # tranform images
        img = resize(x, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove



class Multimodal3DCT_RATEPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CT_RATE3D_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_ct_rate_3D.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path))
        img = nii_img.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # tranform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove



class Multimodal3DCT_RATE_Unique_PretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):
        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        json_path = os.path.join(CHEXPERT_DATA_DIR, CT_RATE3D_WLGPTM_MASTER_JSON)
        # Load the JSON file containing report-image mapping
        with open(json_path, 'r', encoding='utf-8') as f:
            self.report_to_images = json.load(f)[split]

        # Create list of all reports (keys in the JSON file)
        self.reports = list(self.report_to_images.keys())
        # ipdb.set_trace()

        # Create BERT tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model.text.bert_type, trust_remote_code=True)

    def get_caption(self, report):
        series_sents = report

        if len(series_sents) == 0:
            print(report)
            raise Exception("no sentence for report")

        # ipdb.set_trace()
        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        elif self.cfg.data.text.merge_num == 1:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]
        else:
            # 随机聚合n个句子
            n_sentences = min(self.cfg.data.text.merge_num, len(series_sents))  # 确保不超过句子总数
            selected_sents = list(np.random.choice(series_sents, n_sentences, replace=False))
            sent = " ".join(selected_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path))
        img = nii_img.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # transform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):
        # Select a report (key)
        report = self.reports[index]
        # Randomly select an image from the list of images corresponding to this report
        # Try to get the image
        img_path = random.choice(self.report_to_images[report]['images'])
        imgs = self.get_imgs(img_path, self.transform)
        # Randomly select a sentence (caption) corresponding to this report
        caps, cap_len = self.get_caption(self.report_to_images[report]['processed_report'])
        return imgs, caps, cap_len, img_path


    def __len__(self):
        return len(self.reports)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove



class Multimodal3DCT_RATE_Unique_Organ_PretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):
        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        json_path = os.path.join(CHEXPERT_DATA_DIR, CT_RATE3D_ORGAN_WLABEL_MASTER_JSON)
        # Load the JSON file containing report-image mapping
        with open(json_path, 'r', encoding='utf-8') as f:
            self.report_to_images = json.load(f)[split]

        # Create list of all reports (keys in the JSON file)
        self.reports = list(self.report_to_images.keys())
        # ipdb.set_trace()

        # Create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def get_caption(self, organ_to_report):
        # Step 1: Select as many unique organs as possible, up to N
        available_organs = [organ for organ, reports in organ_to_report.items() if reports]
        random.shuffle(available_organs)
        n_organs = min(self.cfg.data.text.merge_num, len(available_organs))
        selected_organs = available_organs[:n_organs]

        # Step 2: Extract one sentence from each selected organ
        selected_sents = []
        organ_list = []
        used_sentences = set()
        for organ in selected_organs:
            reports = organ_to_report[organ]
            available_reports = [report for report in reports if report not in used_sentences]
            if available_reports:
                selected_sentence = random.choice(available_reports)
                selected_sents.append(selected_sentence)
                organ_list.append(organ)
                used_sentences.add(selected_sentence)

        # Step 3: If the number of selected sentences is less than N, continue extracting from already selected organs
        while len(selected_sents) < self.cfg.data.text.merge_num:
            organ = random.choice(selected_organs)
            reports = organ_to_report[organ]
            available_reports = [report for report in reports if report not in used_sentences]
            if available_reports:
                selected_sentence = random.choice(available_reports)
                selected_sents.append(selected_sentence)
                organ_list.append(organ)
                used_sentences.add(selected_sentence)

        # Step 4: Combine the selected sentences
        sent = " ".join(selected_sents)

        # Step 5: Tokenize each selected sentence individually
        individual_tokens = []
        for s in selected_sents:
            tokens = self.tokenizer(
                s,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            individual_tokens.append(tokens)

        # Step 6: Tokenize the combined sentence
        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len, individual_tokens, organ_list


    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path[0]))
        img = nii_img.get_fdata()

        mask = nib.load(str(img_path[1]))
        mask = mask.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # transform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
            mask = resize(mask, self.cfg.data.image.imsize, mode='reflect', order=0, anti_aliasing=True)


        if transform is not None:
            img = {"image": img, "mask": mask}
            transformed_data = transform(img)
        return transformed_data["image"], transformed_data["mask"]

    def __getitem__(self, index):
        # Select a report (key)
        report = self.reports[index]
        # Randomly select an image from the list of images corresponding to this report
        # Try to get the imageq
        img_path = random.choice(self.report_to_images[report]['images'])
        imgs, mask = self.get_imgs(img_path, self.transform)
        # Randomly select a sentence (caption) corresponding to this report
        caps, cap_len, shot_caps, organ_list  = self.get_caption(self.report_to_images[report]['organ_to_report'])
        organ_token_map = self.organ_mapping(mask.squeeze(0), organ_list)
        return imgs, caps, cap_len, img_path, shot_caps, organ_list, organ_token_map

    def __len__(self):
        return len(self.reports)

    def organ_mapping(self, mask, organ_list):
        organ_mask_label = {
            "spleen": [1],
            "kidney": [2, 3, 23, 24],
            "gallbladder": [4],
            "liver": [5],
            "stomach": [6],
            "pancreas": [7],
            "adrenal_gland": [8, 9],
            "lung": [10, 11, 12, 13, 14],
            "esophagus": [15],
            "trachea": [16],
            "thyroid_gland": [17],
            "small_bowel": [18, 19],
            "colon": [20],
            "urinary_bladder": [21],
            "prostate": [22],
            "vertebrae": [
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
                45, 46, 47, 48, 49, 50
            ],
            "heart": [51, 61],
            "blood_vessels": [
                52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 
                63, 64, 65, 66, 67, 68
            ],
            "humerus": [69, 70],
            "scapula": [71, 72],
            "clavicula": [73, 74],
            "femur": [75, 76],
            "hip": [77, 78],
            "spinal_cord": [79],
            "gluteus": [
                80, 81, 82, 83, 84, 85
            ],
            "autochthon": [86, 87],
            "iliopsoas": [88, 89],
            "brain": [90],
            "skull": [91],
            "ribs_left": [
                92, 93, 94, 95, 96, 97, 98, 99, 100, 
                101, 102, 103
            ],
            "ribs_right": [
                104, 105, 106, 107, 108, 109, 110, 111, 
                112, 113, 114, 115
            ],
            "sternum": [116],
            "costal_cartilages": [117],
            "outline": [118]
        }
        
        result = []
        patch_shape = (16, 16, 8)
        mask_shape = mask.shape
        
        # Calculate the number of patches along each dimension
        num_patches = (
            mask_shape[0] // patch_shape[0],
            mask_shape[1] // patch_shape[1],
            mask_shape[2] // patch_shape[2]
        )
        
        organ_indices = {organ: [] for organ in organ_list}
        sequence_index = 0
        
        for z in range(num_patches[2]):
            for y in range(num_patches[1]):
                for x in range(num_patches[0]):
                    # Extract the patch
                    patch = mask[
                        x * patch_shape[0]:(x + 1) * patch_shape[0],
                        y * patch_shape[1]:(y + 1) * patch_shape[1],
                        z * patch_shape[2]:(z + 1) * patch_shape[2]
                    ]
                    
                    unique_patch_values = np.unique(patch)
                    
                    for organ in organ_list:
                        labels = organ_mask_label.get(organ, [])
                        # Check if any of the labels for the organ are present in the patch
                        if any(label in unique_patch_values for label in labels):
                            organ_indices[organ].append(sequence_index)
                    
                    sequence_index += 1
        
        result = [organ_indices[organ] for organ in organ_list]
        return result



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove



class Multimodal3DCT_RATE_Unique_Organ_All_PretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):
        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        json_path = os.path.join(CHEXPERT_DATA_DIR, CT_RATE3D_ORGAN_WLABEL_ALL_MASTER_JSON)
        # Load the JSON file containing report-image mapping
        with open(json_path, 'r', encoding='utf-8') as f:
            self.report_to_images = json.load(f)[split]

        # Create list of all reports (keys in the JSON file)
        self.reports = list(self.report_to_images.keys())
        # ipdb.set_trace()

        # Create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def get_caption(self, organ_to_report, split_reports):
        
        #  extract for total image
        series_sents = split_reports
        if len(series_sents) == 0:
            print(split_reports)
            raise Exception("no sentence for report")
        # ipdb.set_trace()
        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        elif self.cfg.data.text.merge_num == 1:
            sent_ix = random.randint(0, len(series_sents) - 1)
            sent = series_sents[sent_ix]
        else:
            # 随机聚合n个句子
            n_sentences = min(self.cfg.data.text.merge_num, len(series_sents))  # 确保不超过句子总数
            selected_sents = random.sample(series_sents, n_sentences)
            sent = " ".join(selected_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        # Step 1: Select as many unique organs as possible, up to N
        available_organs = [organ for organ, reports in organ_to_report.items() if reports]
        random.shuffle(available_organs)
        n_organs = min(self.cfg.data.text.organ_number, len(available_organs))
        selected_organs = available_organs[:n_organs]

        # Step 2: Extract less than m sentences from each selected organ
        selected_sents = []
        for organ in selected_organs:
            reports = organ_to_report[organ]
            # 随机聚合n个句子
            n_sentences = min(self.cfg.data.text.merge_num, len(reports))  # 确保不超过句子总数
            s_sents = random.sample(reports, n_sentences)
            s_sents = " ".join(s_sents)
            selected_sents.append(s_sents)

        #  Step 3: If the number of organ less than organ_number, create (n-avalibale) "outline" and set sents to "."
        if n_organs < self.cfg.data.text.organ_number:
            for _ in range(self.cfg.data.text.organ_number - n_organs):
                selected_organs.append("outline")
                selected_sents.append("Nothing")  # 或者 "[PAD]"

        # Step 5: Tokenize each selected sentence individually
        individual_tokens = []
        for s in selected_sents:
            idv_tokens = self.tokenizer(
                s,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            individual_tokens.append(idv_tokens)

        return tokens, x_len, individual_tokens, selected_organs


    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path[0]))
        img = nii_img.get_fdata()

        mask = nib.load(str(img_path[1]))
        mask = mask.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # transform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
            mask = resize(mask, self.cfg.data.image.imsize, mode='reflect', order=0, anti_aliasing=True)


        if transform is not None:
            img = {"image": img, "mask": mask}
            transformed_data = transform(img)
        return transformed_data["image"], transformed_data["mask"]

    def __getitem__(self, index):
        # Select a report (key)
        report = self.reports[index]
        # Randomly select an image from the list of images corresponding to this report
        # Try to get the imageq
        img_path = random.choice(self.report_to_images[report]['images'])
        imgs, mask = self.get_imgs(img_path, self.transform)
        # Randomly select a sentence (caption) corresponding to this report
        caps, cap_len, shot_caps, organ_list  = self.get_caption(self.report_to_images[report]['organ_to_report'], self.report_to_images[report]['processed_report'])
        organ_token_map = self.organ_mapping(mask.squeeze(0), organ_list)
        return imgs, caps, cap_len, img_path, shot_caps, organ_list, organ_token_map

    def __len__(self):
        return len(self.reports)

    def organ_mapping(self, mask, organ_list):
        organ_mask_label = {
            "spleen": [1],
            "kidney": [2, 3, 23, 24],
            "gallbladder": [4],
            "liver": [5],
            "stomach": [6],
            "pancreas": [7],
            "adrenal_gland": [8, 9],
            "lung": [10, 11, 12, 13, 14],
            "esophagus": [15],
            "trachea": [16],
            "thyroid_gland": [17],
            "small_bowel": [18, 19],
            "colon": [20],
            "urinary_bladder": [21],
            "prostate": [22],
            "vertebrae": [
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
                45, 46, 47, 48, 49, 50
            ],
            "heart": [51, 61],
            "blood_vessels": [
                52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 
                63, 64, 65, 66, 67, 68
            ],
            "humerus": [69, 70],
            "scapula": [71, 72],
            "clavicula": [73, 74],
            "femur": [75, 76],
            "hip": [77, 78],
            "spinal_cord": [79],
            "gluteus": [
                80, 81, 82, 83, 84, 85
            ],
            "autochthon": [86, 87],
            "iliopsoas": [88, 89],
            "brain": [90],
            "skull": [91],
            "ribs_left": [
                92, 93, 94, 95, 96, 97, 98, 99, 100, 
                101, 102, 103
            ],
            "ribs_right": [
                104, 105, 106, 107, 108, 109, 110, 111, 
                112, 113, 114, 115
            ],
            "sternum": [116],
            "costal_cartilages": [117],
            "outline": [118]
        }
        
        result = []
        patch_shape = (16, 16, 8)
        mask_shape = mask.shape
        
        # Calculate the number of patches along each dimension
        num_patches = (
            mask_shape[0] // patch_shape[0],
            mask_shape[1] // patch_shape[1],
            mask_shape[2] // patch_shape[2]
        )
        
        organ_indices = {organ: [] for organ in organ_list}
        sequence_index = 0
        
        for z in range(num_patches[2]):
            for y in range(num_patches[1]):
                for x in range(num_patches[0]):
                    # Extract the patch
                    patch = mask[
                        x * patch_shape[0]:(x + 1) * patch_shape[0],
                        y * patch_shape[1]:(y + 1) * patch_shape[1],
                        z * patch_shape[2]:(z + 1) * patch_shape[2]
                    ]
                    
                    unique_patch_values = np.unique(patch)
                    
                    for organ in organ_list:
                        labels = organ_mask_label.get(organ, [])
                        # Check if any of the labels for the organ are present in the patch
                        if any(label in unique_patch_values for label in labels):
                            organ_indices[organ].append(sequence_index)
                    
                    sequence_index += 1
        
        result = [organ_indices[organ] for organ in organ_list]
        return result



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove


class Multimodal3DCT_RATE_Unique_Organ_Three_PretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):
        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        json_path = os.path.join(CHEXPERT_DATA_DIR, CT_RATE3D_ORGAN_WLABEL_ALL_MASTER_JSON)
        # Load the JSON file containing report-image mapping
        with open(json_path, 'r', encoding='utf-8') as f:
            self.report_to_images = json.load(f)[split]

        # Create list of all reports (keys in the JSON file)
        self.reports = list(self.report_to_images.keys())
        # ipdb.set_trace()

        # Create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)


    def get_caption(self, organ_to_report, split_reports):
        
        #  extract for total image
        series_sents = split_reports
        if len(series_sents) == 0:
            print(split_reports)
            raise Exception("no sentence for report")
        # ipdb.set_trace()
        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        elif self.cfg.data.text.merge_num == 1:
            sent_ix = random.randint(0, len(series_sents) - 1)
            sent = series_sents[sent_ix]
        else:
            # 随机聚合n个句子
            n_sentences = min(self.cfg.data.text.merge_num, len(series_sents))  # 确保不超过句子总数
            selected_sents = random.sample(series_sents, n_sentences)
            sent = " ".join(selected_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        # Step 1: Select the specific organs we want, ensuring they exist in organ_to_report
        target_organs = ["lung", "heart", "blood_vessels"]
        selected_organs = []
        selected_sents = []

        # Step 2: Check each target organ and extract sentences if available, otherwise add "Nothing"
        for organ in target_organs:
            reports = organ_to_report.get(organ, [])
            if reports:  # If there are reports for this organ
                n_sentences = min(self.cfg.data.text.merge_num, len(reports))  # Limit to m sentences per organ
                s_sents = random.sample(reports, n_sentences)  # Randomly select n sentences
                s_sents = " ".join(s_sents)  # Combine selected sentences
                selected_sents.append(s_sents)
                selected_organs.append(organ)
            else:  # If no report is available, add outline and "Nothing"
                selected_sents.append("Nothing")
                selected_organs.append("outline")

        # Step 3: Tokenize each selected sentence individually
        individual_tokens = []
        for s in selected_sents:
            idv_tokens = self.tokenizer(
                s,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            individual_tokens.append(idv_tokens)

        return tokens, x_len, individual_tokens, selected_organs


    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path[0]))
        img = nii_img.get_fdata()

        mask = nib.load(str(img_path[1]))
        mask = mask.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # transform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
            mask = resize(mask, self.cfg.data.image.imsize, mode='reflect', order=0, anti_aliasing=True)


        if transform is not None:
            img = {"image": img, "mask": mask}
            transformed_data = transform(img)
        return transformed_data["image"], transformed_data["mask"]

    def __getitem__(self, index):
        # Select a report (key)
        report = self.reports[index]
        # Randomly select an image from the list of images corresponding to this report
        # Try to get the imageq
        img_path = random.choice(self.report_to_images[report]['images'])
        imgs, mask = self.get_imgs(img_path, self.transform)
        # Randomly select a sentence (caption) corresponding to this report
        caps, cap_len, shot_caps, organ_list  = self.get_caption(self.report_to_images[report]['organ_to_report'], self.report_to_images[report]['processed_report'])
        organ_token_map = self.organ_mapping(mask.squeeze(0), organ_list)
        return imgs, caps, cap_len, img_path, shot_caps, organ_list, organ_token_map

    def __len__(self):
        return len(self.reports)

    def organ_mapping(self, mask, organ_list):
        organ_mask_label = {
            "spleen": [1],
            "kidney": [2, 3, 23, 24],
            "gallbladder": [4],
            "liver": [5],
            "stomach": [6],
            "pancreas": [7],
            "adrenal_gland": [8, 9],
            "lung": [10, 11, 12, 13, 14],
            "esophagus": [15],
            "trachea": [16],
            "thyroid_gland": [17],
            "small_bowel": [18, 19],
            "colon": [20],
            "urinary_bladder": [21],
            "prostate": [22],
            "vertebrae": [
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
                45, 46, 47, 48, 49, 50
            ],
            "heart": [51, 61],
            "blood_vessels": [
                52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 
                63, 64, 65, 66, 67, 68
            ],
            "humerus": [69, 70],
            "scapula": [71, 72],
            "clavicula": [73, 74],
            "femur": [75, 76],
            "hip": [77, 78],
            "spinal_cord": [79],
            "gluteus": [
                80, 81, 82, 83, 84, 85
            ],
            "autochthon": [86, 87],
            "iliopsoas": [88, 89],
            "brain": [90],
            "skull": [91],
            "ribs_left": [
                92, 93, 94, 95, 96, 97, 98, 99, 100, 
                101, 102, 103
            ],
            "ribs_right": [
                104, 105, 106, 107, 108, 109, 110, 111, 
                112, 113, 114, 115
            ],
            "sternum": [116],
            "costal_cartilages": [117],
            "outline": [118]
        }
        
        result = []
        patch_shape = (16, 16, 8)
        mask_shape = mask.shape
        
        # Calculate the number of patches along each dimension
        num_patches = (
            mask_shape[0] // patch_shape[0],
            mask_shape[1] // patch_shape[1],
            mask_shape[2] // patch_shape[2]
        )
        
        organ_indices = {organ: [] for organ in organ_list}
        sequence_index = 0
        
        for z in range(num_patches[2]):
            for y in range(num_patches[1]):
                for x in range(num_patches[0]):
                    # Extract the patch
                    patch = mask[
                        x * patch_shape[0]:(x + 1) * patch_shape[0],
                        y * patch_shape[1]:(y + 1) * patch_shape[1],
                        z * patch_shape[2]:(z + 1) * patch_shape[2]
                    ]
                    
                    unique_patch_values = np.unique(patch)
                    
                    for organ in organ_list:
                        labels = organ_mask_label.get(organ, [])
                        # Check if any of the labels for the organ are present in the patch
                        if any(label in unique_patch_values for label in labels):
                            organ_indices[organ].append(sequence_index)
                    
                    sequence_index += 1
        
        result = [organ_indices[organ] for organ in organ_list]
        return result



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove



class Multimodal3DCT_RATE_Unique_Organ_Three_Label_PretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):
        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        json_path = os.path.join(CHEXPERT_DATA_DIR, CT_RATE3D_ORGAN_WLABEL_ALL_MASTER_JSON)
        # Load the JSON file containing report-image mapping
        with open(json_path, 'r', encoding='utf-8') as f:
            self.report_to_images = json.load(f)[split]

        # Create list of all reports (keys in the JSON file)
        self.reports = list(self.report_to_images.keys())
        # ipdb.set_trace()

        # Create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def get_caption(self, organ_to_report, split_reports):
        
        #  extract for total image
        series_sents = split_reports
        if len(series_sents) == 0:
            print(split_reports)
            raise Exception("no sentence for report")
        # ipdb.set_trace()
        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        elif self.cfg.data.text.merge_num == 1:
            sent_ix = random.randint(0, len(series_sents) - 1)
            sent = series_sents[sent_ix]
        else:
            # 随机聚合n个句子
            n_sentences = min(self.cfg.data.text.merge_num, len(series_sents))  # 确保不超过句子总数
            selected_sents = random.sample(series_sents, n_sentences)
            sent = " ".join(selected_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        # Step 1: Select the specific organs we want, ensuring they exist in organ_to_report
        target_organs = ["lung", "heart", "blood_vessels"]
        selected_organs = []
        selected_sents = []

        # Step 2: Check each target organ and extract sentences if available, otherwise add "Nothing"
        for organ in target_organs:
            reports = organ_to_report.get(organ, [])
            if reports:  # If there are reports for this organ
                n_sentences = min(self.cfg.data.text.merge_num, len(reports))  # Limit to m sentences per organ
                s_sents = random.sample(reports, n_sentences)  # Randomly select n sentences
                s_sents = " ".join(s_sents)  # Combine selected sentences
                selected_sents.append(s_sents)
                selected_organs.append(organ)
            else:  # If no report is available, add outline and "Nothing"
                selected_sents.append("Nothing")
                selected_organs.append("outline")

        # Step 3: Tokenize each selected sentence individually
        individual_tokens = []
        for s in selected_sents:
            idv_tokens = self.tokenizer(
                s,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            individual_tokens.append(idv_tokens)

        return tokens, x_len, individual_tokens, selected_organs


    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path[0]))
        img = nii_img.get_fdata()

        mask = nib.load(str(img_path[1]))
        mask = mask.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # transform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
            mask = resize(mask, self.cfg.data.image.imsize, mode='reflect', order=0, anti_aliasing=True)


        if transform is not None:
            img = {"image": img, "mask": mask}
            transformed_data = transform(img)
        return transformed_data["image"], transformed_data["mask"]

    def __getitem__(self, index):
        # Select a report (key)
        report = self.reports[index]
        # Randomly select an image from the list of images corresponding to this report
        # Try to get the imageq
        img_path = random.choice(self.report_to_images[report]['images'])
        imgs, mask = self.get_imgs(img_path, self.transform)
        # Randomly select a sentence (caption) corresponding to this report
        caps, cap_len, shot_caps, organ_list  = self.get_caption(self.report_to_images[report]['organ_to_report'], self.report_to_images[report]['processed_report'])
        organ_token_map = self.organ_mapping(mask.squeeze(0), organ_list)
        disease_organ = self.report_to_images[report]['disease_organ']
        return imgs, caps, cap_len, img_path, shot_caps, organ_list, organ_token_map, disease_organ

    def __len__(self):
        return len(self.reports)

    def organ_mapping(self, mask, organ_list):
        organ_mask_label = {
            "spleen": [1],
            "kidney": [2, 3, 23, 24],
            "gallbladder": [4],
            "liver": [5],
            "stomach": [6],
            "pancreas": [7],
            "adrenal_gland": [8, 9],
            "lung": [10, 11, 12, 13, 14],
            "esophagus": [15],
            "trachea": [16],
            "thyroid_gland": [17],
            "small_bowel": [18, 19],
            "colon": [20],
            "urinary_bladder": [21],
            "prostate": [22],
            "vertebrae": [
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
                45, 46, 47, 48, 49, 50
            ],
            "heart": [51, 61],
            "blood_vessels": [
                52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 
                63, 64, 65, 66, 67, 68
            ],
            "humerus": [69, 70],
            "scapula": [71, 72],
            "clavicula": [73, 74],
            "femur": [75, 76],
            "hip": [77, 78],
            "spinal_cord": [79],
            "gluteus": [
                80, 81, 82, 83, 84, 85
            ],
            "autochthon": [86, 87],
            "iliopsoas": [88, 89],
            "brain": [90],
            "skull": [91],
            "ribs_left": [
                92, 93, 94, 95, 96, 97, 98, 99, 100, 
                101, 102, 103
            ],
            "ribs_right": [
                104, 105, 106, 107, 108, 109, 110, 111, 
                112, 113, 114, 115
            ],
            "sternum": [116],
            "costal_cartilages": [117],
            "outline": [118]
        }
        
        result = []
        patch_shape = (16, 16, 8)
        mask_shape = mask.shape
        
        # Calculate the number of patches along each dimension
        num_patches = (
            mask_shape[0] // patch_shape[0],
            mask_shape[1] // patch_shape[1],
            mask_shape[2] // patch_shape[2]
        )
        
        organ_indices = {organ: [] for organ in organ_list}
        sequence_index = 0
        
        for z in range(num_patches[2]):
            for y in range(num_patches[1]):
                for x in range(num_patches[0]):
                    # Extract the patch
                    patch = mask[
                        x * patch_shape[0]:(x + 1) * patch_shape[0],
                        y * patch_shape[1]:(y + 1) * patch_shape[1],
                        z * patch_shape[2]:(z + 1) * patch_shape[2]
                    ]
                    
                    unique_patch_values = np.unique(patch)
                    
                    for organ in organ_list:
                        labels = organ_mask_label.get(organ, [])
                        # Check if any of the labels for the organ are present in the patch
                        if any(label in unique_patch_values for label in labels):
                            organ_indices[organ].append(sequence_index)
                    
                    sequence_index += 1
        
        result = [organ_indices[organ] for organ in organ_list]
        return result

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove



class Multimodal3DCT_RATE_Unique_Organ_Three_Large_PretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):
        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image
        json_path = os.path.join(CHEXPERT_DATA_DIR, CT_RATE3D_ORGAN_WLABEL_ALL_MASTER_JSON)
        # Load the JSON file containing report-image mapping
        with open(json_path, 'r', encoding='utf-8') as f:
            self.report_to_images = json.load(f)[split]

        # Create list of all reports (keys in the JSON file)
        self.reports = list(self.report_to_images.keys())
        # ipdb.set_trace()

        # Create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def get_caption(self, organ_to_report, split_reports):
        
        #  extract for total image
        series_sents = split_reports
        if len(series_sents) == 0:
            print(split_reports)
            raise Exception("no sentence for report")
        # ipdb.set_trace()
        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        elif self.cfg.data.text.merge_num == 1:
            sent_ix = random.randint(0, len(series_sents) - 1)
            sent = series_sents[sent_ix]
        else:
            # 随机聚合n个句子
            n_sentences = min(self.cfg.data.text.merge_num, len(series_sents))  # 确保不超过句子总数
            selected_sents = random.sample(series_sents, n_sentences)
            sent = " ".join(selected_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        # Step 1: Select the specific organs we want, ensuring they exist in organ_to_report
        target_organs = ["lung", "heart", "blood_vessels"]
        selected_organs = []
        selected_sents = []

        # Step 2: Check each target organ and extract sentences if available, otherwise add "Nothing"
        for organ in target_organs:
            reports = organ_to_report.get(organ, [])
            if reports:  # If there are reports for this organ
                n_sentences = min(self.cfg.data.text.merge_num, len(reports))  # Limit to m sentences per organ
                s_sents = random.sample(reports, n_sentences)  # Randomly select n sentences
                s_sents = " ".join(s_sents)  # Combine selected sentences
                selected_sents.append(s_sents)
                selected_organs.append(organ)
            else:  # If no report is available, add outline and "Nothing"
                selected_sents.append("Nothing")
                selected_organs.append("outline")

        # Step 3: Tokenize each selected sentence individually
        individual_tokens = []
        for s in selected_sents:
            idv_tokens = self.tokenizer(
                s,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            individual_tokens.append(idv_tokens)

        return tokens, x_len, individual_tokens, selected_organs


    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path[0]))
        img = nii_img.get_fdata()

        mask = nib.load(str(img_path[1]))
        mask = mask.get_fdata()

        if self.cfg.data.image.imsize is not None:
            # transform images
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
            mask = resize(mask, self.cfg.data.image.imsize, mode='reflect', order=0, anti_aliasing=True)


        if transform is not None:
            img = {"image": img, "mask": mask}
            transformed_data = transform(img)
        return transformed_data["image"], transformed_data["mask"]

    def __getitem__(self, index):
        # Select a report (key)
        report = self.reports[index]
        # Randomly select an image from the list of images corresponding to this report
        # Try to get the imageq
        img_path = random.choice(self.report_to_images[report]['images'])
        imgs, mask = self.get_imgs(img_path, self.transform)
        # Randomly select a sentence (caption) corresponding to this report
        caps, cap_len, shot_caps, organ_list  = self.get_caption(self.report_to_images[report]['organ_to_report'], self.report_to_images[report]['processed_report'])
        organ_token_map = self.organ_mapping(mask.squeeze(0), organ_list)
        return imgs, caps, cap_len, img_path, shot_caps, organ_list, organ_token_map

    def __len__(self):
        return len(self.reports)

    def organ_mapping(self, mask, organ_list):
        organ_mask_label = {
            "spleen": [1],
            "kidney": [2, 3, 23, 24],
            "gallbladder": [4],
            "liver": [5],
            "stomach": [6],
            "pancreas": [7],
            "adrenal_gland": [8, 9],
            "lung": [10, 11, 12, 13, 14],
            "esophagus": [15],
            "trachea": [16],
            "thyroid_gland": [17],
            "small_bowel": [18, 19],
            "colon": [20],
            "urinary_bladder": [21],
            "prostate": [22],
            "vertebrae": [
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
                45, 46, 47, 48, 49, 50
            ],
            "heart": [51, 61],
            "blood_vessels": [
                52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 
                63, 64, 65, 66, 67, 68
            ],
            "humerus": [69, 70],
            "scapula": [71, 72],
            "clavicula": [73, 74],
            "femur": [75, 76],
            "hip": [77, 78],
            "spinal_cord": [79],
            "gluteus": [
                80, 81, 82, 83, 84, 85
            ],
            "autochthon": [86, 87],
            "iliopsoas": [88, 89],
            "brain": [90],
            "skull": [91],
            "ribs_left": [
                92, 93, 94, 95, 96, 97, 98, 99, 100, 
                101, 102, 103
            ],
            "ribs_right": [
                104, 105, 106, 107, 108, 109, 110, 111, 
                112, 113, 114, 115
            ],
            "sternum": [116],
            "costal_cartilages": [117],
            "outline": [118]
        }
        
        result = []
        patch_shape = (16, 16, 8)
        mask_shape = mask.shape
        
        # Calculate the number of patches along each dimension
        num_patches = (
            mask_shape[0] // patch_shape[0],
            mask_shape[1] // patch_shape[1],
            mask_shape[2] // patch_shape[2]
        )
        
        organ_indices = {organ: [] for organ in organ_list}
        sequence_index = 0
        
        for z in range(num_patches[2]):
            for y in range(num_patches[1]):
                for x in range(num_patches[0]):
                    # Extract the patch
                    patch = mask[
                        x * patch_shape[0]:(x + 1) * patch_shape[0],
                        y * patch_shape[1]:(y + 1) * patch_shape[1],
                        z * patch_shape[2]:(z + 1) * patch_shape[2]
                    ]
                    
                    unique_patch_values = np.unique(patch)
                    
                    for organ in organ_list:
                        labels = organ_mask_label.get(organ, [])
                        # Check if any of the labels for the organ are present in the patch
                        if any(label in unique_patch_values for label in labels):
                            organ_indices[organ].append(sequence_index)
                    
                    sequence_index += 1
        
        result = [organ_indices[organ] for organ in organ_list]
        return result



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove



class Multimodal3DCT_RATECachPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):
        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # Read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CT_RATE3D_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        # Load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # Load all images into memory
        self.images = {filename: self.get_imgs(filename) for filename in self.filenames}

        # Create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # Get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_ct_rate_3D.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(self.df, self.max_word_num)
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # Filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][CHEXPERT_PATH_COL].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        return filenames, path2sent

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents) - 1)
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path):
        nii_img = nib.load(str(img_path))
        img = nii_img.get_fdata()

        if self.cfg.data.image.imsize is not None:
            img = resize(img, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)

        return img

    def __getitem__(self, index):
        key = self.filenames[index]
        imgs = self.images[key]  # Retrieve preloaded image
        if self.transform is not None:
            imgs = self.transform(imgs)

        caps, cap_len = self.get_caption(key)
        return imgs, caps, cap_len, key

    
    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove




class Multimodal3DPretrainingCacheDataset(CacheDataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEST3D_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

        data = [{'img': f, 'label': self.path2sent[f]} for f in self.filenames if f in self.path2sent]

        super().__init__(data=data, transform=self.prepare_transforms(), cache_rate=1.0, num_workers=4)


    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_3D.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        nii_img = nib.load(str(img_path))
        x = nii_img.get_fdata()
     
        # tranform images
        img = resize(x, self.cfg.data.image.imsize, mode='reflect', anti_aliasing=True)
        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove



class MultimodalComPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")
        if self.cfg.data.text.n is not None and len(series_sents) >= self.cfg.data.text.n:
            selected_indices = np.random.choice(len(series_sents), self.cfg.data.text.n, replace=False)
            selected_sents = [series_sents[i] for i in selected_indices]
            sent = " ".join(selected_sents)
        elif self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingDQNDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):
        
        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingLLMDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_llm.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    

class MultimodalPretrainingXHComDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_XH)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_XH.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.n is not None and len(series_sents) >= self.cfg.data.text.n:
            selected_indices = np.random.choice(len(series_sents), self.cfg.data.text.n, replace=False)
            selected_sents = [series_sents[i] for i in selected_indices]
            sent = " ".join(selected_sents)
        elif self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_XH_REPORT_COL]) == str:
                captions += row[CHEXPERT_XH_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    



class MultimodalPretrainingXHDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_XH)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_XH.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]
        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_XH_REPORT_COL]) == str:
                captions += row[CHEXPERT_XH_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingXHOnlyDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_XH)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_XHonly.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            # if type(row[CHEXPERT_REPORT_COL]) == str:
            #     captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_XH_REPORT_COL]) == str:
                captions += row[CHEXPERT_XH_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    
    
class MultimodalPretrainingXHOnlyAugDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_XH_keyword)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_XHonlyAug.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            
            total_path2sent = []
            total_toremove = []   
            
            total_path2sent.append(path2sent)
            total_toremove.extend(to_remove) 
                  
            template = ["It is [no] observed that the patient has [disease]",
            "The examination shows [no] evidence of [disease]",
            "Patient's condition indicates [no] presence of [disease]",
            "[No] signs of [disease] are detected",
            "There are [no] indications of [disease] in the diagnosis",
            "Clinical findings [no] suggest [disease]",
            "The scan [no] reveals [disease]"]
            
       
            for i, temp in enumerate(template):
                self.df['aug_report' + str(i)] = self.df['keywords_with_no_info'].apply(
                lambda x: self.generate_report_from_template(json.loads(x), temp))
                path2sent, to_remove = self.create_path_2_sent_aug_mapping(
                self.df, self.max_word_num, 'aug_report' + str(i))
                total_path2sent.append(path2sent)
                total_toremove.extend(to_remove)
            
      
            total_toremove.extend(to_remove)
            total_toremove = list(set(total_toremove))
            
            
            with open(filepath, "wb") as f:
                pickle.dump([total_path2sent, total_toremove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        filenames = [ f for f in filenames if len(path2sent[0][f]) ==  len(path2sent[1][f])]
        
        print("len of filenames for training: ", len(filenames))
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def generate_report_from_template(self, keywords_info, template):
        """根据模板和关键词信息生成报告，如果所有关键词列表为空，则生成空字符串"""
        report_parts = []
        for sentence_keywords in keywords_info:
            # 如果关键词列表为空，跳过当前列表
            if not sentence_keywords:
                continue

            sentence_phrases = []
            for keyword, has_no in sentence_keywords:
                # 根据是否有 'no' 来填充模板
                phrase = template.replace('[disease]', keyword).replace('[no]', 'no' if has_no else '')
                sentence_phrases.append(phrase)
            
            # 使用 'and' 连接同一句子中的多个关键词
            combined_sentence = ' and '.join(sentence_phrases)

            # 确保每个句子以句号结束
            if not combined_sentence.endswith('.'):
                combined_sentence += '.'

            report_parts.append(combined_sentence)
        
        # 如果所有关键词列表都为空，返回空字符串
        if not report_parts:
            return ''

        return ' '.join(report_parts)

    def get_caption(self, path):
        
        # if len(self.path2sent[0][path]) != len(self.path2sent[1][path]):
        #     ipdb.set_trace()
            
        # ipdb.set_trace()
        series_sents = self.path2sent[0][path]
        all_series = []
        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")
        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]
            all_series.append(sent)

        for i in range(1, 8):
            all_series.append(self.path2sent[i][path][sent_ix]) 
        
  
        
        tokens = self.tokenizer(
            all_series,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            # if idx  == 1153:
            #     ipdb.set_trace()
            # pick impression, findings, last_paragraph
            captions = ""
            # if type(row[CHEXPERT_REPORT_COL]) == str:
            #     captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_XH_REPORT_COL]) == str:
                captions += row[CHEXPERT_XH_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                # if cnt == max_word_num:
                #     break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove


    def create_path_2_sent_aug_mapping(self, df, max_word_num, key_name):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            # if type(row[CHEXPERT_REPORT_COL]) == str:
            #     captions += row[CHEXPERT_REPORT_COL]
            if type(row[key_name]) == str:
                captions += row[key_name]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                # if cnt == max_word_num:
                #     break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove
    
    
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
    


class MultimodalPretrainingXHADODataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_XH_ADO)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_XH_ADO.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_XH_REPORT_COL]) == str:
                captions += row[CHEXPERT_XH_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    

class MultimodalPretrainingXHGPT4TemplateDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_XH_GPT4TEMPLATE)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_XH_GPT4_Template.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_XH_REPORT_COL]) == str:
                captions += row[CHEXPERT_XH_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingLLMV1Dataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V7)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_llm_v1.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_V1_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_V1_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingLLMDQNDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_llm.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingLLMDQNGLDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_llm.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingLLMDQNLDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V5)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_llm_large.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingLLMDQNFDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions_llm_fast.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open( os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
            noise_sample = pickle.load(f)
        
        tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
        filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]
        
        # filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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
    


class MultimodalPretrainingUniversalRAMDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        # self.ram_transform = ram_transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # self.df = self.df[self.df[CHEXPERT_DataFlag_COL] == 0]

        # load studies and study to text mapping
        self.filenames, self.path2sent, self.path2flag, self.path2index = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove, path2flag, path2index = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove, path2flag, path2index], f, protocol=2)
                print("Save to: ", filepath)

        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove, path2flag, path2index = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open('/data/haoranlai/Project/gloria/Dataset/cxr_report_noise_sample.pickle', 'rb') as f:
            noise_sample = pickle.load(f)
        filenames = [f for f in filenames if f not in noise_sample]


        return filenames, path2sent, path2flag, path2index
    

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):
        x = cv2.imread(str(img_path), 0)
        # x = Image.open(str(img_path))
        # x = np.array(x)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        # img2arr = img.permute(1, 2, 0).numpy
        # ram_img = self.ram_transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(caps)
        # print(cap_len)

        flag = self.path2flag[key]

        index_ = self.path2index[key]

        return imgs, caps, cap_len, key, flag, index_

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        path2flag = {}
        path2index = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])
            
            path2flag[row[CHEXPERT_PATH_COL]] = row[CHEXPERT_DataFlag_COL]
             
            vector_str = row[CHEXPERT_RAMINDEX_COL]
            # 去除双引号和括号，并将字符串拆分为数字列表
            vector_list = vector_str.strip('[]').split()

            # 将字符串列表转换为整数列表
            vector = [int(num) for num in vector_list]

            # 或者将字符串列表转换为 numpy 数组
            vector = np.array(vector_list, dtype=int)

            path2index[row[CHEXPERT_PATH_COL]] = vector


        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove, path2flag, path2index

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


# class MultimodalPretrainingUniversalCRDDataset(data.Dataset):
#     def __init__(self, cfg, split="train", transform=None):

#         if CHEXPERT_DATA_DIR is None:
#             raise RuntimeError(
#                 "CheXpert data path empty\n"
#                 + "Make sure to download data from:\n"
#                 + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
#                 + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
#             )

#         self.cfg = cfg
#         self.transform = transform
#         self.max_word_num = self.cfg.data.text.captions_per_image

#         # read CheXpert csv file
#         csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
#         self.df = pd.read_csv(csv_path)

#         with open('/data/haoranlai/Project/gloria/Dataset/position.json', 'r') as f:
#             self.position =  json.load(f)


#         csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
#         self.df_label = pd.read_csv(csv_path)



#         # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
#         #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
#         # )
#         # only extract from Frontal

#         # self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

#         # load studies and study to text mapping
#         self.filenames, self.path2sent = self.load_text_data(split)
#         self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)

#         # create BERT tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



#     def load_text_data(self, split):
#         # get study to captions mapping
#         filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
#         if not os.path.isfile(filepath):
#             print(f"Caption file {filepath} does not exit. Creating captions...")
#             path2sent, to_remove = self.create_path_2_sent_mapping(
#                 self.df, self.max_word_num
#             )
#             with open(filepath, "wb") as f:
#                 pickle.dump([path2sent, to_remove], f, protocol=2)
#                 print("Save to: ", filepath)
#         else:
#             with open(filepath, "rb") as f:
#                 print(f"Loading captions from {filepath}")
#                 path2sent, to_remove = pickle.load(f)

#         # filter studies to use for current split
#         filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
#             CHEXPERT_PATH_COL
#         ].tolist()
#         filenames = [f for f in filenames if f not in to_remove]

#         return filenames, path2sent
    
#     def load_CXR_diseases_data(self, split):
#         # get study to captions mapping
#         filepath = os.path.join(CHEXPERT_DATA_DIR, "cxr_label.pickle")
#         # if not os.path.isfile(filepath):
#         #     print(f"Caption file {filepath} does not exit. Creating captions...")
#         #     path2sent, to_remove = self.create_path_2_sent_mapping(
#         #         self.df, self.max_word_num
#         #     )
#         #     with open(filepath, "wb") as f:
#         #         pickle.dump([path2sent, to_remove], f, protocol=2)
#         #         print("Save to: ", filepath)
#         # else:
#         with open(filepath, "rb") as f:
#             print(f"Loading cxr_labels from {filepath}")
#             labels = pickle.load(f)

#         # filter studies to use for current split
#         filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
#             CHEXPERT_PATH_COL
#         ].tolist()
#         # filenames = list(labels.keys()) 

#         # filenames = [f for f in filenames if f not in to_remove]

#         return filenames, labels

#     def get_caption(self, path):

#         series_sents = self.path2sent[path]

#         if len(series_sents) == 0:
#             print(path)
#             raise Exception("no sentence for path")

#         if self.cfg.data.text.full_report is True:
#             sent = " ".join(series_sents)
#         else:
#             sent_ix = random.randint(0, len(series_sents))
#             sent = series_sents[sent_ix]

#         tokens = self.tokenizer(
#             sent,
#             return_tensors="pt",
#             truncation=True,
#             padding="max_length",
#             max_length=self.cfg.data.text.word_num,
#         )
#         x_len = len([t for t in tokens["input_ids"][0] if t != 0])

#         return tokens, x_len

#     def get_imgs(self, img_path, transform=None):

#         x = cv2.imread(str(img_path), 0)

#         # tranform images
#         x = self._resize_img(x, self.cfg.data.image.imsize)
#         img = Image.fromarray(x).convert("RGB")

#         if transform is not None:
#             img = transform(img)

#         return img

#     def __getitem__(self, index):

#         key = self.filenames[index]

#         imgs = self.get_imgs(key, self.transform)

#         # randomly select a sentence
#         caps, cap_len = self.get_caption(key)

#         # randomly select a files in diseases files
#         indx = random.randint(0, len(self.disease_filenames))

#         d_key = self.disease_filenames[indx]
   
#         d_imgs = self.get_imgs(d_key, self.transform)

#         d_class = torch.from_numpy(self.diseases_class[d_key]) 


#         return imgs, caps, cap_len, key, d_imgs, d_class, d_key

#     def __len__(self):
#         return len(self.filenames)

#     def create_path_2_sent_mapping(self, df, max_word_num):

#         sent_lens, num_sents, to_remove = [], [], []
#         path2sent = {}
#         for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

#             # pick impression, findings, last_paragraph
#             captions = ""
#             if type(row[CHEXPERT_REPORT_COL]) == str:
#                 pos = row[CHEXPERT_Original_VIEW_COL]
#                 random_pos = random.choice(self.position[pos])
#                 if random.random() < 0.5:
#                     captions += random_pos + " "
#                     captions += row[CHEXPERT_REPORT_COL]
#                 else:
#                     captions += row[CHEXPERT_REPORT_COL]
#                     captions += " " + random_pos   

#             # remove empty reports
#             if len(captions) == 0:
#                 to_remove.append(row[CHEXPERT_PATH_COL])

#             # use space instead of newline
#             captions = captions.replace("\n", " ")

#             # split sentences
#             splitter = re.compile("[0-9]+\.")
#             captions = splitter.split(captions)
#             captions = [point.split(".") for point in captions]
#             captions = [sent for point in captions for sent in point]

#             cnt = 0
#             study_sent = []
#             # create tokens from captions
#             for cap in captions:

#                 if len(cap) == 0:
#                     continue

#                 cap = cap.replace("\ufffd\ufffd", " ")
#                 # picks out sequences of alphanumeric characters as tokens
#                 # and drops everything else
#                 tokenizer = RegexpTokenizer(r"\w+")
#                 tokens = tokenizer.tokenize(cap.lower())

#                 # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
#                 if len(tokens) <= 1:
#                     # if len(tokens) < 3:
#                     continue

#                 # filter tokens for current sentence
#                 included_tokens = []
#                 for t in tokens:
#                     t = t.encode("ascii", "ignore").decode("ascii")
#                     if len(t) > 0:
#                         included_tokens.append(t)
#                 study_sent.append(" ".join(included_tokens))

#                 # check if reached maximum number of words in the sentences
#                 cnt += len(included_tokens)
#                 if cnt == max_word_num:
#                     break

#                 sent_lens.append(len(included_tokens))
#             num_sents.append(len(study_sent))

#             # remove paths without setnences
#             if len(study_sent) > 0:
#                 path2sent[row[CHEXPERT_PATH_COL]] = study_sent
#             else:
#                 to_remove.append(row[CHEXPERT_PATH_COL])

#         # get report word/setence statistics
#         sent_lens = np.array(sent_lens)
#         num_sents = np.array(num_sents)
#         print(
#             f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
#         )
#         print(
#             f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
#         )

#         return path2sent, to_remove

#     def _resize_img(self, img, scale):
#         """
#         Args:
#             img - image as numpy array (cv2)
#             scale - desired output image-size as scale x scale
#         Return:
#             image resized to scale x scale with shortest dimension 0-padded
#         """
#         size = img.shape
#         max_dim = max(size)
#         max_ind = size.index(max_dim)

#         # Resizing
#         if max_ind == 0:
#             # image is heigher
#             wpercent = scale / float(size[0])
#             hsize = int((float(size[1]) * float(wpercent)))
#             desireable_size = (scale, hsize)
#         else:
#             # image is wider
#             hpercent = scale / float(size[1])
#             wsize = int((float(size[0]) * float(hpercent)))
#             desireable_size = (wsize, scale)
#         resized_img = cv2.resize(
#             img, desireable_size[::-1], interpolation=cv2.INTER_AREA
#         )  # this flips the desireable_size vector

#         # Padding
#         if max_ind == 0:
#             # height fixed at scale, pad the width
#             pad_size = scale - resized_img.shape[1]
#             left = int(np.floor(pad_size / 2))
#             right = int(np.ceil(pad_size / 2))
#             top = int(0)
#             bottom = int(0)
#         else:
#             # width fixed at scale, pad the height
#             pad_size = scale - resized_img.shape[0]
#             top = int(np.floor(pad_size / 2))
#             bottom = int(np.ceil(pad_size / 2))
#             left = int(0)
#             right = int(0)
#         resized_img = np.pad(
#             resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
#         )

#         return resized_img

class MultimodalPretrainingUniversalCRDDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)



        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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



class MultimodalPretrainingUniversalCRDLLMDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)



        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_llm_captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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




class MultimodalPretrainingUniversalCRDV2Dataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V2)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV_V2)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)


        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_captions_V2.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        # if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
        #     with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
        #         noise_sample = pickle.load(f)
        #     filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label_V2.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        # if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
        #     with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
        #         noise_sample = pickle.load(f)
        #     filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]

        if not os.path.exists(d_key):
            print(d_key)
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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




class MultimodalPretrainingUniversalCRDV3Dataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V3)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV_V3)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)


        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_captions_V3.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        # if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
        #     with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
        #         noise_sample = pickle.load(f)
        #     filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label_V3.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        # if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
        #     with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
        #         noise_sample = pickle.load(f)
        #     filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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




class MultimodalPretrainingUniversalCRDLLMV2Dataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V2)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV_V2)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)


        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_llm_captions_V2.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        # if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
        #     with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
        #         noise_sample = pickle.load(f)
        #     filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label_V2.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        # if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
        #     with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
        #         noise_sample = pickle.load(f)
        #     filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]

           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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




class MultimodalPretrainingUniversalCRDCPLLMDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent, self.path2disease = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)


        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_llm_CP_captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove, path2disease = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove, path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove, path2disease = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent, path2disease
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        r_class = torch.from_numpy(self.path2disease[key].astype(np.float32))
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, r_class, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[5:17].values
            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]

           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove, path2disease

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



class MultimodalPretrainingUniversalCRDCPLLMV2Dataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V2)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV_V2)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent, self.path2disease = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)


        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_llm_CP_captions_V2.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove, path2disease = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove, path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove, path2disease = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        # if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
        #     with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
        #         noise_sample = pickle.load(f)
        #     filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent, path2disease
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label_V2.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        # if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
        #     with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
        #         noise_sample = pickle.load(f)
        #     filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        r_class = torch.from_numpy(self.path2disease[key].astype(np.float32))
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, r_class, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[5:17].values
            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]

           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove, path2disease

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



class MultimodalPretrainingUniversalCRDPDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)



        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_P_captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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


class MultimodalPretrainingUniversalCRDPLLMDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)



        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_P_llm_captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            # filenames = [f for f in filenames if f not in noise_sample]
            tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
            filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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



class MultimodalPretrainingUniversalCRDPLLMV5Dataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V5)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV_V5)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)



        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_P_llm_captions_v5.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            # filenames = [f for f in filenames if f not in noise_sample]
            tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
            filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label_v5.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
            filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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



class MultimodalPretrainingUniversalCRDPLLMV6Dataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V6)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV_V6)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)



        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_P_llm_captions_v6.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            # filenames = [f for f in filenames if f not in noise_sample]
            tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
            filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label_v6.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            tempnoise_sample = [ os.path.basename(f) for f in noise_sample]
            filenames = [f for f in filenames if os.path.basename(f) not in tempnoise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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



class MultimodalPretrainingUniversalCRDPLLMV4Dataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV_V4)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV_V4)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)



        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_P_llm_captions_v4.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label_v4.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files

        indx = random.randint(0, len(self.disease_filenames))
        

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
            if type(row[CHEXPERT_LLM_REPORT_COL]) == str:
                captions += row[CHEXPERT_LLM_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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




class MultimodalPretrainingUniversalCRDCPDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent, self.path2disease = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)


        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_CP_captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove, path2disease = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove, path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove, path2disease = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent, path2disease
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        r_class = torch.from_numpy(self.path2disease[key].astype(np.float32))
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, r_class, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[5:17].values
            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]

           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove, path2disease

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




class MultimodalPretrainingUniversalCRDCPDALIDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent, self.path2disease = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)

        self.mimic_image = np.memmap('/haoranlai/Project/gloria/NumpyDataset/preprocessed_mimic_images.npy', dtype=np.uint8, mode='r+', shape=(230546, 256, 256))
        with open('/haoranlai/Project/gloria/NumpyDataset/preprocessed_mimic_images.pickle', 'rb') as f:
            self.mimic_image_index = pickle.load(f)
         
        self.all_cxr_image = np.memmap('/haoranlai/Project/gloria/NumpyDataset/preprocessed_all_cxr_images.npy', dtype=np.uint8, mode='r+', shape=(175449, 256, 256))
        with open('/haoranlai/Project/gloria/NumpyDataset/preprocessed_all_cxr_images.pickle', 'rb') as f:
            self.all_cxr_image_index = pickle.load(f)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove, path2disease = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove, path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove, path2disease = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open('/haoranlai/Project/gloria/Dataset/cxr_report_noise_sample.pickle', 'rb') as f:
            noise_sample = pickle.load(f)
        filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent,path2disease
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "cxr_label.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        with open('/haoranlai/Project/gloria/Dataset/cxr_disease_noise_sample.pickle', 'rb') as f:
            noise_sample = pickle.load(f)
        filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        # imgs = self.get_imgs(key, self.transform)
        imgs = self.mimic_image[self.mimic_image_index[key]]
        imgs = Image.fromarray(imgs).convert("RGB")
        if self.transform is not None:
            imgs = self.transform (imgs)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        r_class = torch.from_numpy(self.path2disease[key].astype(np.float32))
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        # d_imgs = self.get_imgs(d_key, self.transform)
        d_imgs = self.all_cxr_image[self.all_cxr_image_index[d_key]]
        d_imgs = Image.fromarray(d_imgs).convert("RGB")
        if self.transform is not None:
            d_imgs = self.transform (d_imgs)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, r_class, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[5:].values
            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove, path2disease

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





class MultimodalPretrainingUniversalCRDCATDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)

        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_LABRL_CSV)
        self.df_label = pd.read_csv(csv_path)

        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        # only extract from Frontal

        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        self.disease_filenames, self.diseases_class = self.load_CXR_diseases_data(split)



        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)



    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_report_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        return filenames, path2sent
    
    def load_CXR_diseases_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "univercdd_cxr_label.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2disease = self.create_path_2_disease_mapping(
                self.df_label
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2disease], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading cxr_labels from {filepath}")
                path2disease = pickle.load(f)
                path2disease = path2disease[0]
        # ipdb.set_trace()
        # filter studies to use for current split
        filenames = self.df_label[self.df_label[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        # filenames = self.df_label[
        #     CHEXPERT_PATH_COL
        # ].tolist()
        if os.path.exists(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle')):
            with open(os.path.join(CHEXPERT_DATA_DIR, 'cxr_disease_noise_sample.pickle'), 'rb') as f:
                noise_sample = pickle.load(f)
            filenames = [f for f in filenames if f not in noise_sample]

        # filenames = [f for f in filenames if f not in to_remove]
        return filenames, path2disease

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)
        # print(len(self.disease_filenames))
        # ipdb.set_trace()
        # randomly select a files in diseases files
        indx = random.randint(0, len(self.disease_filenames))

        d_key = self.disease_filenames[indx]
   
        d_imgs = self.get_imgs(d_key, self.transform)

        d_class = torch.from_numpy(self.diseases_class[d_key].astype(np.float32)) 

        return imgs, caps, cap_len, key, d_imgs, d_class, d_key

    def __len__(self):
        return len(self.filenames)


    def create_path_2_disease_mapping(self, df):
        path2disease = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            path2disease[row[CHEXPERT_PATH_COL]] = row.iloc[2:].values
        return path2disease



    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:     
                captions += row[CHEXPERT_REPORT_COL]
           

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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

class MultimodalPretrainingTrippleDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        # self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
        #     lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        # )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent, self.path2path = self.load_text_data(split)


        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        with open(os.path.join(CHEXPERT_DATA_DIR, "path2path.json"), 'r') as f:
            path2path = json.load(f)

        return filenames, path2sent, path2path

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_embedding(self, path):
        with open(self.path2path[path] , 'rb') as f:
            data = pickle.load(f)
            embedding = torch.from_numpy(self.padding_shape(np.stack(data[0], axis=0))).float()     
        return embedding, len(data[0])
    
    def padding_shape(self, M):

        bing = np.zeros((59, 1536))

        # 将原始矩阵放置到目标矩阵中

        bing[0:M.shape[0], 0:M.shape[1]] = M

        return bing
    
    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)
        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        embedding, emb_len = self.get_embedding(key)


        return imgs, caps, cap_len, key, embedding, emb_len

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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



def multimodal_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path = [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
    }
    return return_dict



def multimodal_collate_organ_mask_fn(batch):
    """Group main captions and shot_caps without sorting shot_caps; keep each cap and shot_caps in original order"""
    imgs, cap_len, ids_list, tokens_list, attention_list, shot_caps_grouped, path = [], [], [], [], [], [], []
   
    # Flatten and separate main captions and shot_caps, treating each cap and its shot_caps as a group
    for b_idx, b in enumerate(batch):
        img, cap, cap_l, p, shot_caps, organ_list, organ_token_map = b
        imgs.append(img)
        cap_len.append(cap_l)
        
        # Append primary caption
        ids_list.append(cap["input_ids"])
        tokens_list.append(cap["token_type_ids"])
        attention_list.append(cap["attention_mask"])

        # Group shot_caps data, retaining organ information for each group
        shot_caps_grouped.append({
            "shot_caps": shot_caps,
            "organ_list": organ_list,
            "organ_token_map": organ_token_map,
            "original_index": b_idx  # Retain the original index for future reference
        })

        path.append(p)

    # Stack images
    imgs = torch.stack(imgs)

    # Sort main captions based on caption lengths
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)

    # Sort main captions and images according to sorted indices
    sorted_ids_list = [ids_list[idx] for idx in sorted_cap_indices]
    sorted_tokens_list = [tokens_list[idx] for idx in sorted_cap_indices]
    sorted_attention_list = [attention_list[idx] for idx in sorted_cap_indices]
    sorted_imgs = imgs[sorted_cap_indices]
    sorted_paths = [path[idx] for idx in sorted_cap_indices]

    # Stack sorted main captions
    ids = torch.stack(sorted_ids_list).squeeze()
    tokens = torch.stack(sorted_tokens_list).squeeze()
    attention = torch.stack(sorted_attention_list).squeeze()

    # Maintain the order of shot_caps within each group, matching the sorted main captions
    sorted_shot_caps_grouped = [shot_caps_grouped[idx] for idx in sorted_cap_indices]

    # Prepare shot_caps after sorting main captions, preserving original order within each group
    sorted_shot_ids, sorted_shot_tokens, sorted_shot_attention = [], [], []
    sorted_shot_organ_lists, sorted_shot_organ_token_maps, sorted_shot_indices = [], [], []

    for group in sorted_shot_caps_grouped:
        shot_caps = group["shot_caps"]
        organ_list = group["organ_list"]
        organ_token_map = group["organ_token_map"]
        original_index = group["original_index"]

        # Append all shot_caps related to this main cap in their original order
        for shot_cap in shot_caps:
            sorted_shot_ids.append(shot_cap["input_ids"])
            sorted_shot_tokens.append(shot_cap["token_type_ids"])
            sorted_shot_attention.append(shot_cap["attention_mask"])
            sorted_shot_indices.append(original_index)
        sorted_shot_organ_lists.append(organ_list)
        sorted_shot_organ_token_maps.append(organ_token_map)

    # Stack shot_caps without additional sorting within each group
    sorted_shot_ids = torch.stack(sorted_shot_ids).squeeze()
    sorted_shot_tokens = torch.stack(sorted_shot_tokens).squeeze()
    sorted_shot_attention = torch.stack(sorted_shot_attention).squeeze()

    # Create return dictionary
    return_dict = {
        "caption_ids": ids,
        "token_type_ids": tokens,
        "attention_mask": attention,
        "imgs": sorted_imgs,
        "cap_lens": sorted_cap_lens,
        "path": sorted_paths,
        "shot_caption_ids": sorted_shot_ids,
        "shot_token_type_ids": sorted_shot_tokens,
        "shot_attention_mask": sorted_shot_attention,
        "shot_organ_lists": sorted_shot_organ_lists,
        "shot_organ_token_maps": sorted_shot_organ_token_maps,
        "shot_original_indices": sorted_shot_indices,  # Retain original indices for mapping back
    }

    return return_dict



def multimodal_collate_organ_mask_disease_fn(batch):
    """Group main captions and shot_caps without sorting shot_caps; keep each cap and shot_caps in original order"""
    imgs, cap_len, ids_list, tokens_list, attention_list, shot_caps_grouped, path, disease_organ_list = [], [], [], [], [], [], [], []
   
    # Flatten and separate main captions and shot_caps, treating each cap and its shot_caps as a group
    for b_idx, b in enumerate(batch):
        img, cap, cap_l, p, shot_caps, organ_list, organ_token_map, disease_organ = b
        imgs.append(img)
        cap_len.append(cap_l)
        
        # Append primary caption
        ids_list.append(cap["input_ids"])
        tokens_list.append(cap["token_type_ids"])
        attention_list.append(cap["attention_mask"])

        # Group shot_caps data, retaining organ information for each group
        shot_caps_grouped.append({
            "shot_caps": shot_caps,
            "organ_list": organ_list,
            "organ_token_map": organ_token_map,
            "original_index": b_idx  # Retain the original index for future reference
        })

        path.append(p)
        disease_organ_list.append(disease_organ) 

    # Stack images
    imgs = torch.stack(imgs)

    # Sort main captions based on caption lengths
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)

    # Sort main captions and images according to sorted indices
    sorted_ids_list = [ids_list[idx] for idx in sorted_cap_indices]
    sorted_tokens_list = [tokens_list[idx] for idx in sorted_cap_indices]
    sorted_attention_list = [attention_list[idx] for idx in sorted_cap_indices]
    sorted_imgs = imgs[sorted_cap_indices]
    sorted_paths = [path[idx] for idx in sorted_cap_indices]
    sorted_disease_organ_list = [disease_organ_list[idx] for idx in sorted_cap_indices]

    # Stack sorted main captions
    ids = torch.stack(sorted_ids_list).squeeze()
    tokens = torch.stack(sorted_tokens_list).squeeze()
    attention = torch.stack(sorted_attention_list).squeeze()

    # Maintain the order of shot_caps within each group, matching the sorted main captions
    sorted_shot_caps_grouped = [shot_caps_grouped[idx] for idx in sorted_cap_indices]

    # Prepare shot_caps after sorting main captions, preserving original order within each group
    sorted_shot_ids, sorted_shot_tokens, sorted_shot_attention = [], [], []
    sorted_shot_organ_lists, sorted_shot_organ_token_maps, sorted_shot_indices = [], [], []

    for group in sorted_shot_caps_grouped:
        shot_caps = group["shot_caps"]
        organ_list = group["organ_list"]
        organ_token_map = group["organ_token_map"]
        original_index = group["original_index"]

        # Append all shot_caps related to this main cap in their original order
        for shot_cap in shot_caps:
            sorted_shot_ids.append(shot_cap["input_ids"])
            sorted_shot_tokens.append(shot_cap["token_type_ids"])
            sorted_shot_attention.append(shot_cap["attention_mask"])
            sorted_shot_indices.append(original_index)
        sorted_shot_organ_lists.append(organ_list)
        sorted_shot_organ_token_maps.append(organ_token_map)

    # Stack shot_caps without additional sorting within each group
    sorted_shot_ids = torch.stack(sorted_shot_ids).squeeze()
    sorted_shot_tokens = torch.stack(sorted_shot_tokens).squeeze()
    sorted_shot_attention = torch.stack(sorted_shot_attention).squeeze()

    # Create return dictionary
    return_dict = {
        "caption_ids": ids,
        "token_type_ids": tokens,
        "attention_mask": attention,
        "imgs": sorted_imgs,
        "cap_lens": sorted_cap_lens,
        "path": sorted_paths,
        "shot_caption_ids": sorted_shot_ids,
        "shot_token_type_ids": sorted_shot_tokens,
        "shot_attention_mask": sorted_shot_attention,
        "shot_organ_lists": sorted_shot_organ_lists,
        "shot_organ_token_maps": sorted_shot_organ_token_maps,
        "shot_original_indices": sorted_shot_indices,  # Retain original indices for mapping back
        "disease_organ": sorted_disease_organ_list
    }

    return return_dict





def multimodal_aug_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path = [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.cat(ids, dim=0)
    tokens = torch.cat(tokens,dim=0)
    attention = torch.cat(attention,dim=0)

    # # sort and add to dictionary
    # sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    # return_dict = {
    #     "caption_ids": ids[sorted_cap_indices],
    #     "token_type_ids": tokens[sorted_cap_indices],
    #     "attention_mask": attention[sorted_cap_indices],
    #     "imgs": imgs[sorted_cap_indices],
    #     "cap_lens": sorted_cap_lens,
    #     "path": path,
    # }
    
      # sort and add to dictionary
    return_dict = {
        "caption_ids": ids,
        "token_type_ids": tokens,
        "attention_mask": attention,
        "imgs": imgs,
        "cap_lens": cap_len,
        "path": path,
    }

    return return_dict

def multimodal_collate_UniversalRAM_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, flag, index = [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, f, ind = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        flag.append(f)
        index.append(ind)

    # stack
    imgs = torch.stack(imgs)
    # ram_imgs = torch.stack(ram_imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    flag = np.squeeze(np.stack(flag)) 
    index = np.squeeze(np.stack(index)) 


    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "flag": flag[sorted_cap_indices],
        "index": index[sorted_cap_indices]
    }

    return return_dict

def multimodal_UniversalCRD_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict



def multimodal_UniversalCRDLLM_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict



def multimodal_UniversalCRDV2_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict




def multimodal_UniversalCRDV3_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict



def multimodal_UniversalCRDLLMV2_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict


def multimodal_UniversalCRDP_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict


def multimodal_UniversalCRDCP_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, r_label, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, r_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        r_label.append(r_l)
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)
    r_label = torch.stack(r_label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "r_label": r_label[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict


def multimodal_UniversalCRDCPLLMV2_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, r_label, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, r_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        r_label.append(r_l)
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)
    r_label = torch.stack(r_label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "r_label": r_label[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict


def multimodal_UniversalCRDCPLLM_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, r_label, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, r_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        r_label.append(r_l)
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)
    r_label = torch.stack(r_label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "r_label": r_label[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict



def multimodal_UniversalCRDCPDALI_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, r_label, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, r_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        r_label.append(r_l)
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)
    r_label = torch.stack(r_label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "r_label": r_label[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict



def multimodal_UniversalCRDCAT_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, label_imgs, label, label_image_path = [], [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, limgs, l, limgsp = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        label_imgs.append(limgs)
        label.append(l)
        label_image_path.append(limgsp)


    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    label_imgs = torch.stack(label_imgs)
    label =  torch.stack(label)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "label_imgs": label_imgs,
        "label": label,
        "label_image_path":label_image_path

    }

    return return_dict



def multimodal_tripple_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path, embedding, emb_len = [], [], [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p, emb, emb_l = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)
        embedding.append(emb)
        emb_len.append(emb_l)

    # stack
    imgs = torch.stack(imgs)
    embedding = torch.stack(embedding)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()


    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
        "embeddings": embedding[sorted_cap_indices],
        "emb_lens": torch.tensor(emb_len)[sorted_cap_indices]
    }

    return return_dict