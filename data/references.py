"""
data/references.py — Reference data: regions, climate, diseases, and risk priors
"""

# ─────────────────────────────────────────────────────────────
# REGIONAL CLIMATE STATISTICS
# ─────────────────────────────────────────────────────────────

REGIONS = {
    "Tashkent": {
        "Spring": {"t": 17, "ts": 3, "h": 65, "hs": 8, "r": 60, "soil": "loam_sierozem", "ph": 7.7},
        "Summer": {"t": 32, "ts": 3, "h": 30, "hs": 7, "r": 5,  "soil": "loam_sierozem", "ph": 7.7},
        "Autumn": {"t": 16, "ts": 4, "h": 55, "hs": 8, "r": 30, "soil": "loam_sierozem", "ph": 7.7},
        "Winter": {"t": 1,  "ts": 4, "h": 72, "hs": 7, "r": 50, "soil": "loam_sierozem", "ph": 7.7},
    },
    "Fergana": {
        "Spring": {"t": 18, "ts": 3, "h": 70, "hs": 9, "r": 45, "soil": "alluvial_loam", "ph": 7.4},
        "Summer": {"t": 29, "ts": 3, "h": 40, "hs": 8, "r": 8,  "soil": "alluvial_loam", "ph": 7.4},
        "Autumn": {"t": 17, "ts": 3, "h": 60, "hs": 8, "r": 20, "soil": "alluvial_loam", "ph": 7.4},
        "Winter": {"t": 2,  "ts": 4, "h": 78, "hs": 7, "r": 35, "soil": "alluvial_loam", "ph": 7.4},
    },
    "Samarkand": {
        "Spring": {"t": 16, "ts": 3, "h": 62, "hs": 8, "r": 55, "soil": "loess_sierozem", "ph": 7.9},
        "Summer": {"t": 30, "ts": 3, "h": 28, "hs": 6, "r": 2,  "soil": "loess_sierozem", "ph": 7.9},
        "Autumn": {"t": 15, "ts": 4, "h": 52, "hs": 8, "r": 25, "soil": "loess_sierozem", "ph": 7.9},
        "Winter": {"t": 0,  "ts": 4, "h": 70, "hs": 7, "r": 45, "soil": "loess_sierozem", "ph": 7.9},
    },
    "Surkhandarya": {
        "Spring": {"t": 22, "ts": 3, "h": 55, "hs": 7, "r": 40, "soil": "meadow_sierozem", "ph": 7.5},
        "Summer": {"t": 36, "ts": 2, "h": 22, "hs": 5, "r": 1,  "soil": "meadow_sierozem", "ph": 7.5},
        "Autumn": {"t": 22, "ts": 3, "h": 45, "hs": 7, "r": 20, "soil": "meadow_sierozem", "ph": 7.5},
        "Winter": {"t": 5,  "ts": 4, "h": 65, "hs": 7, "r": 30, "soil": "meadow_sierozem", "ph": 7.5},
    },
    "Karakalpakstan": {
        "Spring": {"t": 15, "ts": 4, "h": 50, "hs": 8, "r": 20, "soil": "takyr_solonchak", "ph": 8.5},
        "Summer": {"t": 30, "ts": 3, "h": 20, "hs": 5, "r": 2,  "soil": "takyr_solonchak", "ph": 8.5},
        "Autumn": {"t": 14, "ts": 4, "h": 45, "hs": 7, "r": 10, "soil": "takyr_solonchak", "ph": 8.5},
        "Winter": {"t": -3, "ts": 5, "h": 65, "hs": 7, "r": 12, "soil": "takyr_solonchak", "ph": 8.5},
    },
    "Khorezm": {
        "Spring": {"t": 16, "ts": 3, "h": 58, "hs": 8, "r": 18, "soil": "alluvial_meadow", "ph": 8.0},
        "Summer": {"t": 31, "ts": 3, "h": 30, "hs": 6, "r": 3,  "soil": "alluvial_meadow", "ph": 8.0},
        "Autumn": {"t": 15, "ts": 4, "h": 52, "hs": 7, "r": 12, "soil": "alluvial_meadow", "ph": 8.0},
        "Winter": {"t": -2, "ts": 4, "h": 68, "hs": 7, "r": 10, "soil": "alluvial_meadow", "ph": 8.0},
    },
}

# ─────────────────────────────────────────────────────────────
# ENCODINGS
# ─────────────────────────────────────────────────────────────

SOIL_TYPES = [
    "loam_sierozem",
    "alluvial_loam",
    "loess_sierozem",
    "meadow_sierozem",
    "takyr_solonchak",
    "alluvial_meadow",
]

SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
REGION_NAMES = list(REGIONS.keys())

SOIL_TO_IDX   = {s: i for i, s in enumerate(SOIL_TYPES)}
SEASON_TO_IDX = {s: i for i, s in enumerate(SEASONS)}
REGION_TO_IDX = {r: i for i, r in enumerate(REGION_NAMES)}

# ─────────────────────────────────────────────────────────────
# DISEASE PRIOR RISK TABLES
# ─────────────────────────────────────────────────────────────

DISEASES = {
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "base_risk": {
            "Tashkent":       {"Spring":0.62,"Summer":0.58,"Autumn":0.55,"Winter":0.05},
            "Fergana":        {"Spring":0.70,"Summer":0.60,"Autumn":0.58,"Winter":0.05},
            "Samarkand":      {"Spring":0.58,"Summer":0.50,"Autumn":0.48,"Winter":0.04},
            "Surkhandarya":   {"Spring":0.55,"Summer":0.25,"Autumn":0.45,"Winter":0.05},
            "Karakalpakstan": {"Spring":0.35,"Summer":0.20,"Autumn":0.30,"Winter":0.03},
            "Khorezm":        {"Spring":0.40,"Summer":0.22,"Autumn":0.32,"Winter":0.03},
        },
    },

    "Tomato___Late_blight": {
        "crop": "Tomato",
        "base_risk": {
            "Tashkent":       {"Spring":0.65,"Summer":0.08,"Autumn":0.60,"Winter":0.20},
            "Fergana":        {"Spring":0.72,"Summer":0.10,"Autumn":0.65,"Winter":0.22},
            "Samarkand":      {"Spring":0.60,"Summer":0.07,"Autumn":0.55,"Winter":0.18},
            "Surkhandarya":   {"Spring":0.42,"Summer":0.05,"Autumn":0.38,"Winter":0.12},
            "Karakalpakstan": {"Spring":0.30,"Summer":0.04,"Autumn":0.28,"Winter":0.10},
            "Khorezm":        {"Spring":0.35,"Summer":0.05,"Autumn":0.30,"Winter":0.12},
        },
    },

    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "base_risk": {
            "Tashkent":       {"Spring":0.45,"Summer":0.68,"Autumn":0.30,"Winter":0.05},
            "Fergana":        {"Spring":0.50,"Summer":0.72,"Autumn":0.32,"Winter":0.05},
            "Samarkand":      {"Spring":0.40,"Summer":0.62,"Autumn":0.28,"Winter":0.04},
            "Surkhandarya":   {"Spring":0.55,"Summer":0.40,"Autumn":0.35,"Winter":0.05},
            "Karakalpakstan": {"Spring":0.25,"Summer":0.20,"Autumn":0.18,"Winter":0.02},
            "Khorezm":        {"Spring":0.28,"Summer":0.25,"Autumn":0.20,"Winter":0.03},
        },
    },

    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "base_risk": {
            "Tashkent":       {"Spring":0.55,"Summer":0.10,"Autumn":0.50,"Winter":0.30},
            "Fergana":        {"Spring":0.62,"Summer":0.12,"Autumn":0.55,"Winter":0.32},
            "Samarkand":      {"Spring":0.48,"Summer":0.08,"Autumn":0.44,"Winter":0.25},
            "Surkhandarya":   {"Spring":0.35,"Summer":0.05,"Autumn":0.30,"Winter":0.15},
            "Karakalpakstan": {"Spring":0.22,"Summer":0.04,"Autumn":0.20,"Winter":0.12},
            "Khorezm":        {"Spring":0.28,"Summer":0.05,"Autumn":0.25,"Winter":0.15},
        },
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "base_risk": {
            "Tashkent":       {"Spring":0.68,"Summer":0.09,"Autumn":0.62,"Winter":0.22},
            "Fergana":        {"Spring":0.75,"Summer":0.10,"Autumn":0.68,"Winter":0.24},
            "Samarkand":      {"Spring":0.62,"Summer":0.07,"Autumn":0.56,"Winter":0.20},
            "Surkhandarya":   {"Spring":0.44,"Summer":0.04,"Autumn":0.38,"Winter":0.14},
            "Karakalpakstan": {"Spring":0.30,"Summer":0.04,"Autumn":0.28,"Winter":0.10},
            "Khorezm":        {"Spring":0.33,"Summer":0.05,"Autumn":0.30,"Winter":0.12},
        },
    },

    "Potato___Early_blight": {
        "crop": "Potato",
        "base_risk": {
            "Tashkent":       {"Spring":0.60,"Summer":0.55,"Autumn":0.52,"Winter":0.04},
            "Fergana":        {"Spring":0.66,"Summer":0.58,"Autumn":0.55,"Winter":0.04},
            "Samarkand":      {"Spring":0.54,"Summer":0.48,"Autumn":0.46,"Winter":0.03},
            "Surkhandarya":   {"Spring":0.50,"Summer":0.22,"Autumn":0.42,"Winter":0.04},
            "Karakalpakstan": {"Spring":0.32,"Summer":0.18,"Autumn":0.28,"Winter":0.03},
            "Khorezm":        {"Spring":0.36,"Summer":0.20,"Autumn":0.30,"Winter":0.03},
        },
    },

    "Grape___Black_rot": {
        "crop": "Grape",
        "base_risk": {
            "Tashkent":       {"Spring":0.55,"Summer":0.45,"Autumn":0.40,"Winter":0.04},
            "Fergana":        {"Spring":0.62,"Summer":0.50,"Autumn":0.44,"Winter":0.04},
            "Samarkand":      {"Spring":0.50,"Summer":0.40,"Autumn":0.36,"Winter":0.03},
            "Surkhandarya":   {"Spring":0.48,"Summer":0.20,"Autumn":0.32,"Winter":0.04},
            "Karakalpakstan": {"Spring":0.28,"Summer":0.15,"Autumn":0.22,"Winter":0.02},
            "Khorezm":        {"Spring":0.32,"Summer":0.18,"Autumn":0.25,"Winter":0.03},
        },
    },

    "Apple___Apple_scab": {
        "crop": "Apple",
        "base_risk": {
            "Tashkent":       {"Spring":0.62,"Summer":0.08,"Autumn":0.55,"Winter":0.15},
            "Fergana":        {"Spring":0.68,"Summer":0.10,"Autumn":0.60,"Winter":0.18},
            "Samarkand":      {"Spring":0.58,"Summer":0.07,"Autumn":0.52,"Winter":0.14},
            "Surkhandarya":   {"Spring":0.40,"Summer":0.04,"Autumn":0.36,"Winter":0.10},
            "Karakalpakstan": {"Spring":0.28,"Summer":0.03,"Autumn":0.24,"Winter":0.08},
            "Khorezm":        {"Spring":0.32,"Summer":0.04,"Autumn":0.28,"Winter":0.10},
        },
    },
}