# config.py
from pathlib import Path
import torch
import numpy as np

# =====================================================================
# Device configuration
# =====================================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# =====================================================================
# Paths
# =====================================================================
ORIGINAL_DATA_PATH = Path("../space-images-category")
OUTPUT_PATH = Path("../space_images_split")

# =====================================================================
# Split ratios
# =====================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =====================================================================
# Hyperparameters
# =====================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 15
EPOCHS_STAGE3 = EPOCHS_STAGE2
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5
LEARNING_RATE_STAGE3 = LEARNING_RATE_STAGE1 * 0.1

# =====================================================================
# Seed
# =====================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == "mps":
    torch.mps.manual_seed(SEED)