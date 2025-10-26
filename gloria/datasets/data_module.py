import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from . import image_dataset
from . import pretraining_dataset
from .. import builder
import ipdb
import numpy as np
import random
import torch
from monai.transforms import RandSpatialCrop


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    


# 在DataLoader中明确设置生成器
# g = torch.Generator()
# g.manual_seed(1)  # 使用与主进程一致的随机种子


class PretrainingDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class Pretraining3DDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DPretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=T,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class Pretraining3DCT_RATEDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DCT_RATEPretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class Pretraining3DCT_RATE_Unique_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DCT_RATE_Unique_PretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers
        )



class Pretraining3DCT_RATE_Unique_M3D_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DCT_RATE_Unique_PretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D_M3D(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D_M3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D_M3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class Pretraining3DCT_RATE_Unique_M3D_Organ_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DCT_RATE_Unique_Organ_PretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_organ_mask_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class Pretraining3DCT_RATE_Unique_M3D_Organ_All_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DCT_RATE_Unique_Organ_All_PretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_organ_mask_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class Pretraining3DCT_RATE_Unique_M3D_Organ_Three_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DCT_RATE_Unique_Organ_Three_PretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_organ_mask_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class Pretraining3DCT_RATE_Unique_M3D_Organ_Three_Label_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DCT_RATE_Unique_Organ_Three_Label_PretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_organ_mask_disease_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class Pretraining3DCT_RATE_Unique_M3D_Organ_Three_Split_Valid_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.train_dataset = pretraining_dataset.Multimodal3DCT_RATE_Unique_Organ_Three_PretrainingDataset
        self.train_collate_fn = pretraining_dataset.multimodal_collate_organ_mask_fn
        self.dataset = valid_dataset.ImageDataset
        self.collate_fn = valid_dataset.collate_fn


    def train_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "train")
        dataset = self.train_dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.train_collate_fn,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D_M3D_with_mask(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )




class Pretraining3DCT_RATECachDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.Multimodal3DCT_RATECachPretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation_3D(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingComDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalComPretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class PretrainingXHComDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingXHComDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    

class PretrainingDQNDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingDQNDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    


class PretrainingLLMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingLLMDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingXHDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingXHDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)

        # return DataLoader(
        #     dataset,
        #     pin_memory=True,
        #     drop_last=True,
        #     shuffle=True,
        #     batch_size=self.cfg.train.batch_size,
        #     num_workers=self.cfg.train.num_workers,
        #     collate_fn=self.collate_fn,
        #     worker_init_fn=worker_init_fn
        # )

        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    


class PretrainingXHOnlyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingXHOnlyDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    





class PretrainingXHOnlyAugDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingXHOnlyAugDataset
        self.collate_fn = pretraining_dataset.multimodal_aug_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )
    
    # def train_dataloader(self):
    #     transform = builder.build_transformation(self.cfg, "train")
    #     dataset = self.dataset(self.cfg, split="train", transform=transform)

    #     return DataLoader(
    #         dataset,
    #         pin_memory=True,
    #         drop_last=True,
    #         shuffle=True,
    #         batch_size=self.cfg.train.batch_size,
    #         num_workers=self.cfg.train.num_workers,
    #         collate_fn=self.collate_fn,
    #     )
        
        

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    



class PretrainingXHADODataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingXHADODataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
   


class PretrainingXHGPT4TemplateDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingXHGPT4TemplateDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    



class PretrainingXHBatchDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingXHDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.valid_batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.valid_batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    



class PretrainingLLMV1DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingLLMV1Dataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    


class PretrainingLLMDQNDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingLLMDQNDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    


class PretrainingLLMDQNGLDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingLLMDQNGLDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    


class PretrainingLLMDQNLDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingLLMDQNLDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size= batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
  


class PretrainingLLMDQNFDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingLLMDQNFDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        # if self.trainer.current_epoch >= 6:
        #     batch_size = self.cfg.train.batch_size
        # elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
        #     batch_size = self.cfg.train.batch_size // 2
        # elif self.trainer.current_epoch < 2:
        #     batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=256,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
  



class PretrainingUniversalRAMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalRAMDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_UniversalRAM_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        # ram_transform = builder.build_ram_transformation(self.cfg)
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        # ram_transform = builder.build_ram_transformation(self.cfg)
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        # ram_transform = builder.build_ram_transformation(self.cfg)
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDDataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRD_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDLLMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDLLMDataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDLLM_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDV2DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDV2Dataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDV2_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDV3DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDV3Dataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDV3_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDLLMV2DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDLLMV2Dataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDLLMV2_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDCPLLMV2DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDCPLLMV2Dataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDCPLLMV2_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDCPLLMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDCPLLMDataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDCPLLM_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size ,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )




class PretrainingUniversalCRDPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDPDataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDP_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDPLLMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDPLLMDataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDP_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDPLLMV5DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDPLLMV5Dataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDP_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDPLLMV6DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDPLLMV6Dataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDP_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )




class PretrainingUniversalCRDPLLMV4DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDPLLMV4Dataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDP_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDCPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDCPDataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDCP_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



# from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
# import nvidia.dali as dali
# from nvidia.dali import pipeline_def
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types


# @pipeline_def
# def GetDatasetPipeline(data_path, device, shard_id=0, num_shards=1):
#     jpegs, labels = fn.readers.caffe2(path=data_path, shard_id=shard_id, num_shards=num_shards, random_shuffle=True, name="Reader")
#     images = fn.decoders.image(jpegs,
#                                device='mixed' if device == 'gpu' else 'cpu',
#                         output_type=types.GRAY)
#     images = fn.crop_mirror_normalize(images,
#                                       dtype=types.FLOAT,
#                                       std=[0.3081 * 255],
#                                       mean=[0.1307 * 255],
#                                       output_layout="CHW")
#     if device == "gpu":
#         labels = labels.gpu()
#     # PyTorch expects labels as INT64
#     labels = fn.cast(labels, dtype=types.INT64)
#     return images, labels


class PretrainingUniversalCRDCPDALIDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDCPDALIDataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDCPDALI_collate_fn
    
    # def setup(self, stage: str) -> None:
    #     device_id = self.local_rank
    #     shard_id = self.global_rank
    #     num_shards = self.trainer.world_size
    #     self.train_pipeline = GetDatasetPipeline(batch_size=self.cfg.train.batch_size, device='gpu', device_id=device_id, shard_id=shard_id,num_shards=num_shards, num_threads=8)
       
        
    def train_dataloader(self):
        # train_loader = DALIClassificationIterator(self.train_pipeline, reader_name="Reader",
        #                                                last_batch_policy=LastBatchPolicy.PARTIAL)
        # return train_loader
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingUniversalCRDCatDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingUniversalCRDCATDataset
        self.collate_fn = pretraining_dataset.multimodal_UniversalCRDCAT_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        # change batch size based on epoch
        if self.trainer.current_epoch >= 6:
            batch_size = self.cfg.train.batch_size
        elif self.trainer.current_epoch >= 2 and  self.trainer.current_epoch < 6:
            batch_size = self.cfg.train.batch_size // 2
        elif self.trainer.current_epoch < 2:
            batch_size = self.cfg.train.batch_size  // 4
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )



class PretrainingTrippleDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingTrippleDataset
        self.collate_fn = pretraining_dataset.multimodal_tripple_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = image_dataset.CheXpertImageDataset

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class CheXpertOriDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = image_dataset.CheXpertOriImageDataset

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )




class PneumothoraxDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = image_dataset.PneumothoraxImageDataset

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class PneumoniaDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = image_dataset.PneumoniaImageDataset

        if cfg.phase == "detection":
            self.collate_fn = image_dataset.detection_collate_fn
        else:
            self.collate_fn = None

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
