from pathlib import Path

# Project root (../work)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos_160"

TRAIN_SPLIT = DATA_DIR / "train_split_1.pkl"
VAL_SPLIT = DATA_DIR / "val_split_1.pkl"

CHECKPOINT_DIR = ROOT / "checkpoints" # stores the best model weights
MODEL_DIR = ROOT / "models"

IMG_SIZE = 160

SEQUENCE_LENGTH = 64

BATCH_SIZE = 16

NUM_WORKERS = 1

EPOCHS = 20

FREEZE_EPOCHS = 5

LEARNING_RATE = 1e-3

DEVICE = "cuda"

NUM_CLASSES = 9