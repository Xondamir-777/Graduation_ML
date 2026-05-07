"""
config.py — Central project configuration
Change PLANTVILLAGE_ROOT to your local PlantVillage dataset path
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS — change PLANTVILLAGE_ROOT for your system
# ─────────────────────────────────────────────────────────────
PLANTVILLAGE_ROOT = Path("archive/plantvillage dataset/color")   # folder with disease subfolders
PROJECT_ROOT      = Path(__file__).parent
DATA_DIR          = PROJECT_ROOT / "data"
OUTPUTS_DIR       = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR   = OUTPUTS_DIR / "checkpoints"
LOGS_DIR          = OUTPUTS_DIR / "logs"

for d in [DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
IMG_SIZE          = 224
SAMPLES_PER_COMBO = 20       # samples per (region × season × disease)
LABEL_THRESHOLD   = 0.45     # binary label threshold
NOISE_SIGMA       = 0.07     # noise for disease probability
RANDOM_SEED       = 42

# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
BATCH_SIZE        = 32
NUM_EPOCHS        = 5
LEARNING_RATE     = 1e-4
WEIGHT_DECAY      = 1e-4
TRAIN_SPLIT       = 0.70
VAL_SPLIT         = 0.15
TEST_SPLIT        = 0.15
NUM_WORKERS       = 4

# CNN backbone: "resnet18" | "resnet50" | "efficientnet_b0"
CNN_BACKBONE      = "resnet18"
CNN_FREEZE_LAYERS = True     # freeze all layers except last 2 blocks

# Climate features (input to Random Forest and MLP)
CLIMATE_FEATURES  = [
    "avg_temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "soil_ph",
    "soil_type_encoded",
    "season_encoded",
    "region_encoded",
]

NUM_CLIMATE_FEATURES = len(CLIMATE_FEATURES)

PV_FOLDER_MAP = {
    "Tomato___Early_blight":   "Tomato___Early_blight",
    "Tomato___Late_blight":    "Tomato___Late_blight",
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold":      "Tomato___Leaf_Mold",
    "Potato___Late_blight":    "Potato___Late_blight",
    "Potato___Early_blight":   "Potato___Early_blight",
    "Grape___Black_rot":       "Grape___Black_rot",
    "Apple___Apple_scab":      "Apple___Apple_scab",
}

NUM_DISEASE_CLASSES = len(PV_FOLDER_MAP)

LABEL_TO_IDX = {label: i for i, label in enumerate(PV_FOLDER_MAP.keys())}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

# ─────────────────────────────────────────────────────────────
# FEATURE DIMENSIONS
# ─────────────────────────────────────────────────────────────

CNN_EMBED_DIM  = 256                           # CNN backbone output
RF_PROBA_DIM   = NUM_DISEASE_CLASSES           # RF predicted probabilities (8 classes)
RAW_CLIMATE_DIM = NUM_CLIMATE_FEATURES         # raw climate features (optional)

# Final fusion vector size:
# CNN (256) + RF (8) + climate (7) = 271
FUSION_INPUT_DIM = CNN_EMBED_DIM + RF_PROBA_DIM + RAW_CLIMATE_DIM  # 271