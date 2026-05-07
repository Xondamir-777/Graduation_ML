"""
utils/inference.py — Предсказание для одного изображения + климатических данных
"""

import torch
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
from torchvision import transforms

from config import OUTPUTS_DIR, CHECKPOINTS_DIR, IDX_TO_LABEL, NUM_DISEASE_CLASSES
from data.references import SOIL_TO_IDX, SEASON_TO_IDX, REGION_TO_IDX
from models.multimodal_model import MultimodalDiseaseModel
from models.rf_climate import RFClimateModel


SCALER_PATH = OUTPUTS_DIR / "climate_scaler.pkl"
RF_PATH     = OUTPUTS_DIR / "rf_climate.pkl"
MODEL_PATH  = CHECKPOINTS_DIR / "best_model.pth"

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class DiseasePredictor:
    """
    Готовый предиктор для production/demo.
    Принимает путь к изображению + климатические параметры,
    возвращает топ-3 болезни с вероятностями.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # Загрузить модель
        self.model = MultimodalDiseaseModel(pretrained=False)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device)
        )
        self.model.eval().to(self.device)

        # RF + scaler
        self.rf     = RFClimateModel.load(RF_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        print(f"[Predictor] Модель загружена ({self.device})")

    def predict(self,
                image_path: str,
                temperature: float,
                humidity: float,
                rainfall: float,
                soil_ph: float,
                soil_type: str,
                season: str,
                region: str) -> dict:
        """
        Параметры
        ---------
        image_path  : путь к JPG/PNG файлу
        temperature : средняя температура °C
        humidity    : влажность %
        rainfall    : осадки мм
        soil_ph     : pH почвы
        soil_type   : одно из loam_sierozem / alluvial_loam / loess_sierozem /
                               meadow_sierozem / takyr_solonchak / alluvial_meadow
        season      : Весна / Лето / Осень / Зима
        region      : Ташкент / Фергана / Самарканд / Сурхандарья /
                      Каракалпакстан / Хорезм

        Возвращает
        ----------
        dict с ключами:
          top3       : [(label, probability), ...]
          risk_score : float [0,1]
          raw_logits : list
        """

        # ── Изображение ──────────────────────────────────────────────────
        img = Image.open(image_path).convert("RGB")
        img_t = IMG_TRANSFORM(img).unsqueeze(0).to(self.device)

        # ── Климат → numpy ───────────────────────────────────────────────
        X_raw = np.array([[
            temperature,
            humidity,
            rainfall,
            soil_ph,
            SOIL_TO_IDX.get(soil_type, 0),
            SEASON_TO_IDX.get(season, 0),
            REGION_TO_IDX.get(region, 0),
        ]], dtype=np.float32)

        X_scaled = self.scaler.transform(X_raw)

        # ── RF proba ─────────────────────────────────────────────────────
        rf_proba = self.rf.predict_proba_from_array(X_scaled)
        rf_t     = torch.tensor(rf_proba, dtype=torch.float32).to(self.device)

        # ── Inference ────────────────────────────────────────────────────
        with torch.no_grad():
            out     = self.model(img_t, rf_t)
            logits  = out["logits"].squeeze(0)
            prob    = out["prob"].item()
            softmax = torch.softmax(logits, dim=0).cpu().numpy()

        # ── Топ-3 болезни ────────────────────────────────────────────────
        top3_idx = np.argsort(softmax)[::-1][:3]
        top3 = [(IDX_TO_LABEL[i], round(float(softmax[i]), 4))
                for i in top3_idx]

        return {
            "top3":        top3,
            "risk_score":  round(prob, 4),
            "raw_logits":  logits.cpu().numpy().tolist(),
        }


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plant Disease Predictor")
    parser.add_argument("--image",       required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity",    type=float, required=True)
    parser.add_argument("--rainfall",    type=float, default=10.0)
    parser.add_argument("--soil_ph",     type=float, default=7.5)
    parser.add_argument("--soil_type",   default="loam_sierozem")
    parser.add_argument("--season",      default="Весна")
    parser.add_argument("--region",      default="Ташкент")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    predictor = DiseasePredictor(device=args.device)
    result = predictor.predict(
        image_path  = args.image,
        temperature = args.temperature,
        humidity    = args.humidity,
        rainfall    = args.rainfall,
        soil_ph     = args.soil_ph,
        soil_type   = args.soil_type,
        season      = args.season,
        region      = args.region,
    )

    print("\n── Результат предсказания ──────────────────────────")
    print(f"  Risk score: {result['risk_score']:.4f}")
    print("  Топ-3 болезни:")
    for rank, (label, prob) in enumerate(result["top3"], 1):
        print(f"    {rank}. {label:<35} {prob:.4f}")
