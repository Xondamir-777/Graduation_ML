"""
train.py — Main training script

Run:
  # Step 1: generate dataset (one time only)
  python -m data.generate_dataset

  # Step 2: train model
  python train.py

  # With GPU:
  python train.py --device cuda

  # Quick test (5 epochs):
  python train.py --epochs 5 --device cpu
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, OUTPUTS_DIR, NUM_EPOCHS, RANDOM_SEED,
    TRAIN_SPLIT, IDX_TO_LABEL, NUM_DISEASE_CLASSES
)
from data.dataset import PlantDiseaseDataModule
from models.multimodal_model import MultimodalDiseaseModel
from models.rf_climate import RFClimateModel
from utils.trainer import Trainer
from utils.visualize import (
    plot_training_history, plot_confusion_matrix,
    plot_rf_feature_importance, plot_risk_heatmap
)
import config as cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto",
                   help="cpu / cuda / mps / auto")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--skip_rf", action="store_true",
                   help="Load already trained RF model if available")
    return p.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():          return torch.device("cuda")
        if torch.backends.mps.is_available():   return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"\n[Device] {device}")
 
    print("\n[1] Loading data...")
    dm = PlantDiseaseDataModule(
        csv_path=DATA_DIR / "dataset_ml.csv",
        root_dir=DATA_DIR,
    ).setup()
 
    rf_path = OUTPUTS_DIR / "rf_climate.pkl"
    if args.skip_rf and rf_path.exists():
        print("\n[2] Loading saved Random Forest model...")
        rf = RFClimateModel.load(rf_path)
    else:
        print("\n[2] Training Random Forest...")

        df_full = pd.read_csv(DATA_DIR / "dataset_ml.csv")

        df_train_rf, df_val_rf = train_test_split(
            df_full,
            test_size=0.2,
            stratify=df_full["disease_class_idx"],
            random_state=RANDOM_SEED
        )

        rf = RFClimateModel(n_estimators=300)
        rf.fit(df_train_rf, df_val_rf)
 
        plot_rf_feature_importance(rf)
 
    print("\n[3] Creating multimodal model...")
    model = MultimodalDiseaseModel(
        cnn_embedding_dim=256,
        rf_output_dim=NUM_DISEASE_CLASSES,
        num_classes=NUM_DISEASE_CLASSES,
        pretrained=True,
    )

    param_info = model.count_params()
    print(f"  Total parameters:     {param_info['total']:,}")
    print(f"  Trainable parameters: {param_info['trainable']:,}")
    print(f"  Frozen parameters:    {param_info['frozen']:,}")

    print("\n[4] Training multimodal model...")
    trainer = Trainer(model=model, rf=rf, device=device)
 
    
    cfg.NUM_EPOCHS = args.epochs

    trainer.fit(dm.train_loader(), dm.val_loader())
 
    print("\n[5] Testing model performance...")

    class_names = [
        IDX_TO_LABEL[i].replace("___", " | ").replace("_", " ")
        for i in range(NUM_DISEASE_CLASSES)
    ]

    trainer.evaluate(dm.test_loader(), class_names=class_names)
 
    print("\n[6] Generating plots...")

    plot_training_history()
    plot_confusion_matrix()
    plot_risk_heatmap(DATA_DIR / "dataset.csv")

    print(f"\n{'='*60}")
    print("  Training completed successfully!")
    print(f"  Checkpoints: {OUTPUTS_DIR / 'checkpoints'}")
    print(f"  Figures:     {OUTPUTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()