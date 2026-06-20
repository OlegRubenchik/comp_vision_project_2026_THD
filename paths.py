from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(r"C:\projects\comp_vision_project_2026_THD")
DATA_DIR     = ROOT_DIR / "data"
TRAIN_CSV    = DATA_DIR / "train.csv"
TRAIN_IMAGES = DATA_DIR / "train_images"
MODELS_DIR   = ROOT_DIR / "models"
RESULTS_DIR  = ROOT_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_model.pth"

# Class labels used throughout the project
CLASS_NAMES = ["No Defect", "Defect 1", "Defect 2", "Defect 3", "Defect 4"]
NUM_CLASSES = len(CLASS_NAMES)   # 5
