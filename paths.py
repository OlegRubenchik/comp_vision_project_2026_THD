from pathlib import Path
# ── CONFIG ──
ROOT_DIR     = Path(r"C:\projects\comp_vision_project_2026_THD")
DATA_DIR     = ROOT_DIR / "data"   # folder containing train_images/, train.csv
TRAIN_CSV    = DATA_DIR / "train.csv"
TRAIN_IMAGES = DATA_DIR / "train_images"
MODELS_DIR   = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH   = MODELS_DIR / "best_model.pth"
RESULTS_DIR  = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
 