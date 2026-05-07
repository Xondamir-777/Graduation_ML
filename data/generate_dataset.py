"""
data/generate_dataset.py
Run: python -m data.generate_dataset

Creates:
  data/images/        — resized 224×224 images from PlantVillage
  data/dataset.csv    — main dataset
  data/dataset_ml.csv — ML-ready dataset (with encoded features)
"""

import sys, os
sys.path.insert(0, str(__file__[: __file__.rfind("/data")]))

import random
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[WARN] Pillow not installed — run: pip install pillow")

from config import (
    PLANTVILLAGE_ROOT, DATA_DIR, IMG_SIZE, SAMPLES_PER_COMBO,
    LABEL_THRESHOLD, NOISE_SIGMA, RANDOM_SEED, PV_FOLDER_MAP, LABEL_TO_IDX
)
from data.references import (
    REGIONS, DISEASES, SOIL_TO_IDX, SEASON_TO_IDX, REGION_TO_IDX
)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

IMAGES_DIR = DATA_DIR / "images"


def build_image_index() -> dict:
    index = {}
    print(f"\n[1] Indexing images from: {PLANTVILLAGE_ROOT}")

    for label, folder_name in PV_FOLDER_MAP.items():
        folder = PLANTVILLAGE_ROOT / folder_name

        if not folder.exists():
            print(f"  [WARN] Folder not found: {folder}")
            index[label] = []
            continue

        imgs = (
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.JPG")) +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png"))
        )

        random.shuffle(imgs)
        index[label] = imgs

        print(f"  {label}: {len(imgs)} images")

    total = sum(len(v) for v in index.values())
    print(f"  Total: {total} images\n")

    return index


def process_image(src: Path, dst: Path):
    if dst.exists():
        return

    if PIL_OK:
        img = Image.open(src).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.LANCZOS
        )
        img.save(dst, "JPEG", quality=92)
    else:
        shutil.copy2(src, dst)


def generate() -> pd.DataFrame:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index()

    image_pools = {k: list(v) for k, v in image_index.items()}
    for k in image_pools:
        random.shuffle(image_pools[k])

    rows = []

    print("[2] Generating dataset rows...")

    total_combos = len(REGIONS) * 4 * len(DISEASES) * SAMPLES_PER_COMBO

    with tqdm(total=total_combos, unit="img") as bar:

        for region_name, seasons in REGIONS.items():
            for season, c in seasons.items():

                for disease_label, d in DISEASES.items():

                    base = d["base_risk"][region_name][season]
                    pool = image_pools.get(disease_label, [])

                    if len(pool) == 0:
                        print(f"[STOP] No images for {disease_label}")
                        continue

                    for i in range(SAMPLES_PER_COMBO):

                        if len(pool) == 0:
                            print(f"[INFO] Ran out of images for {disease_label}")
                            break

                        # ── climate generation ─────────────────────
                        temp = round(float(np.random.normal(c["t"], c["ts"])), 1)
                        hum  = round(float(np.clip(np.random.normal(c["h"], c["hs"]), 10, 99)), 1)
                        rain = round(float(np.clip(np.random.normal(c["r"], max(c["r"] * 0.3, 1)), 0, 250)), 1)
                        ph   = round(float(np.clip(np.random.normal(c["ph"], 0.2), 5.0, 9.5)), 2)

                        prob = round(float(np.clip(base + np.random.normal(0, NOISE_SIGMA), 0.01, 0.97)), 4)
                        label_bin = 1 if prob >= LABEL_THRESHOLD else 0

                        src_path = pool.pop()

                        safe_region = region_name.replace(" ", "_")
                        fname = f"{disease_label}__{safe_region}__{season}__{i+1:03d}.jpg"
                        dst_path = IMAGES_DIR / fname

                        process_image(src_path, dst_path)

                        rows.append({
                            "image_path":         f"images/{fname}",
                            "plantvillage_label": disease_label,
                            "disease_class_idx":  LABEL_TO_IDX[disease_label],
                            "crop":               d["crop"],

                            "region":             region_name,
                            "season":             season,

                            "avg_temperature_c":  temp,
                            "humidity_pct":       hum,
                            "rainfall_mm":        rain,
                            "soil_type":          c["soil"],
                            "soil_ph":            ph,

                            "disease_probability": prob,
                            "disease_label":       label_bin,
                        })

                        bar.update(1)

    return pd.DataFrame(rows)


def add_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["soil_type_encoded"] = df["soil_type"].map(SOIL_TO_IDX)
    df["season_encoded"] = df["season"].map(SEASON_TO_IDX)
    df["region_encoded"] = df["region"].map(REGION_TO_IDX)

    for col, prefix in [("season", "season"), ("region", "region"), ("soil_type", "soil")]:
        dummies = pd.get_dummies(df[col], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)

    return df


def save_metadata(df: pd.DataFrame):
    meta = {
        "dataset_name": "PlantVillage-Uzbekistan Synthetic Disease Risk Dataset",
        "version": "2.0.0 (no image reuse)",
        "total_rows": len(df),
        "image_size": f"{IMG_SIZE}×{IMG_SIZE}",
        "num_classes": df["plantvillage_label"].nunique(),
        "class_balance": df["disease_label"].value_counts().to_dict(),
        "diseases": list(df["plantvillage_label"].unique()),
        "regions": list(df["region"].unique()),
    }

    with open(DATA_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    print("=" * 60)
    print("  PlantVillage × Uzbekistan — Dataset Generator")
    print("=" * 60)

    df = generate()
    df_ml = add_encodings(df)

    print("\n[3] Saving datasets...")

    df.to_csv(DATA_DIR / "dataset.csv", index=False, encoding="utf-8-sig")
    df_ml.to_csv(DATA_DIR / "dataset_ml.csv", index=False, encoding="utf-8-sig")

    save_metadata(df)

    n_imgs = len(list(IMAGES_DIR.glob("*.jpg")))

    print("\n✓ Done!")
    print(f"  Rows:         {len(df)}")
    print(f"  Images:       {n_imgs}")

    print("\n  Class balance:")
    print(df["disease_label"].value_counts().to_string())