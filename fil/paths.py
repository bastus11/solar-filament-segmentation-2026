"""Central place for dataset / artefact locations (override with env vars)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FIL_ROOT", "/workspace"))
DATA = Path(os.environ.get("FIL_DATA", ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026"))
TRAIN_IMG_DIR = DATA / "train/train_images"
TEST_IMG_DIR = DATA / "test/test_images"
TRAIN_JSON = DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CACHE = Path(os.environ.get("FIL_CACHE", ROOT / "cache"))
RUNS = Path(os.environ.get("FIL_RUNS", ROOT / "runs"))
SUBS = Path(os.environ.get("FIL_SUBS", ROOT / "subs"))
H = W = 2048
