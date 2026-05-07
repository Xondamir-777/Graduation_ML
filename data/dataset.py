"""
data/dataset.py — PyTorch Dataset + DataModule
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

from config import (
    DATA_DIR, OUTPUTS_DIR, BATCH_SIZE, NUM_WORKERS,
    TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED, CLIMATE_FEATURES
)


# ─────────────────────────────────────────────────────────────
# 🧪 Augmentations
# ─────────────────────────────────────────────────────────────

def get_transforms(split: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05
            ),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class PlantDiseaseDataset(Dataset):
    """
    Returns:
      image   : Tensor [3, 224, 224]
      climate : Tensor [n_features]
      label   : long (class index)
      prob    : float (disease probability)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Path,
        transform,
        scaler: StandardScaler = None,
        fit_scaler: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.root = root_dir
        self.transform = transform

        climate_cols = [c for c in CLIMATE_FEATURES if c in df.columns]
        X_clim = df[climate_cols].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler().fit(X_clim)
            joblib.dump(self.scaler, OUTPUTS_DIR / "climate_scaler.pkl")
        else:
            self.scaler = scaler

        self.X_climate = self.scaler.transform(X_clim).astype(np.float32)
        self.climate_dim = self.X_climate.shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.root / row["image_path"]

        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        climate = torch.tensor(self.X_climate[idx], dtype=torch.float32)
        label   = torch.tensor(int(row["disease_class_idx"]), dtype=torch.long)
        prob    = torch.tensor(float(row["disease_probability"]), dtype=torch.float32)

        return {
            "image": img,
            "climate": climate,
            "label": label,
            "prob": prob
        }


class PlantDiseaseDataModule:
    def __init__(
        self,
        csv_path: Path = DATA_DIR / "dataset_ml.csv",
        root_dir: Path = DATA_DIR
    ):
        self.csv_path = csv_path
        self.root_dir = root_dir
        self._loaders = {}
        self.scaler = None
        self.climate_dim = None

    def setup(self):
        df = pd.read_csv(self.csv_path)

        df_train, df_temp = train_test_split(
            df,
            test_size=(1 - TRAIN_SPLIT),
            stratify=df["disease_class_idx"],
            random_state=RANDOM_SEED
        )

        df_val, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp["disease_class_idx"],
            random_state=RANDOM_SEED
        )
        train_ds = PlantDiseaseDataset(
            df_train,
            self.root_dir,
            get_transforms("train"),
            fit_scaler=True
        )

        self.scaler = train_ds.scaler
        self.climate_dim = train_ds.climate_dim

        val_ds = PlantDiseaseDataset(
            df_val,
            self.root_dir,
            get_transforms("val"),
            scaler=self.scaler
        )

        test_ds = PlantDiseaseDataset(
            df_test,
            self.root_dir,
            get_transforms("test"),
            scaler=self.scaler
        )

        self._loaders = {
            "train": DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True
            ),
            "val": DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True
            ),
            "test": DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True
            ),
        }

        print("\n DATA SPLIT")
        print(f"  Train: {len(train_ds)}")
        print(f"  Validation: {len(val_ds)}")
        print(f"  Test: {len(test_ds)}")

        return self

    def train_loader(self):
        return self._loaders["train"]

    def val_loader(self):
        return self._loaders["val"]

    def test_loader(self):
        return self._loaders["test"]