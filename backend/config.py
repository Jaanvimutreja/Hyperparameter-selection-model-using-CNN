"""
config.py
---------
Centralized configuration for the entire HPSM pipeline.
Change parameters here to adjust experiments without editing code.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "meta_cnn.pth")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
TRAINING_META_PATH = os.path.join(MODEL_DIR, "training_meta.json")

DATASET_CACHE_DIR = os.path.join(PROJECT_ROOT, "datasets", "cache")
KNOWLEDGE_BASE_PATH = os.path.join(PROJECT_ROOT, "models", "meta_knowledge_base.json")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "plots")

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
MAX_DATASET_SAMPLES = 2000  # subsample large datasets for fast SVM

# 40 training datasets (small + medium + large)
TRAIN_DATASETS = [
    # --- Small (<1000 rows) ---
    "iris", "wine", "breast_cancer", "glass", "seeds",
    "ionosphere", "vehicle", "sonar", "zoo", "ecoli",
    "vertebral", "dermatology", "haberman", "balance_scale",
    "blood_transfusion", "liver", "hayes_roth", "teaching",
    "user_knowledge", "planning_relax", "diabetes", 
    "splice", "sick", "colic",
    # --- Medium (1k–10k rows) ---
    "banknote", "car", "segment", "satimage", "optdigits",
    "pendigits", "waveform", "page_blocks", "mfeat_factors",
    "steel_plates", "yeast", "abalone", "credit_german",
    "vowel", "wall_robot", "kr_vs_kp", "mfeat_pixel", 
    "phishing", "eeg",
    # --- Large (>10k rows) ---
    "letter", "magic", "shuttle", "nursery", "mushroom",
    "electricity", "nomao",
]

# 10 test datasets
TEST_DATASETS = [
    "heart", "titanic", "adult", "spambase",
    "credit_australian", "mfeat_morphological",
    "tic_tac_toe", "cylinder_bands",
    "climate_model", "monks1",
]

ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS
DATASET_COUNT = len(ALL_DATASETS)

# ---------------------------------------------------------------------------
# Meta-feature extraction
# ---------------------------------------------------------------------------
META_FEATURE_GROUPS = ["general", "statistical", "info-theory", "landmarking", "model-based"]
MATRIX_SIZE = 20
META_FEATURE_LENGTH = MATRIX_SIZE * MATRIX_SIZE  # 400

# ---------------------------------------------------------------------------
# Algorithm + Hyperparameter search space
# ---------------------------------------------------------------------------
# See algorithm_space.py for the full config registry
NUM_CONFIGS = 36  # total configs across all algorithms

HP_CV_FOLDS = 5
HP_SCORING = "accuracy"
SVM_MAX_ITER = 5000

# ---------------------------------------------------------------------------
# CNN meta-learner
# ---------------------------------------------------------------------------
CNN_EPOCHS = 200
CNN_BATCH_SIZE = 16
CNN_LEARNING_RATE = 1e-3
CNN_DROPOUT = 0.4
CNN_WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
NUM_AUGMENTED = 30          # noisy copies per real sample
NOISE_STD = 0.05            # Gaussian noise std
SUBSAMPLE_AUGMENT = 10      # subsampled copies per real sample
FEATURE_PERTURB_AUGMENT = 5 # feature-shuffled copies per real sample

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_CV_FOLDS = 5
TOP_K = 3
BASELINE_TRIALS = 10

# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
SIMILARITY_K = 5  # nearest neighbors for dataset similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
for _dir in [MODEL_DIR, DATASET_CACHE_DIR, LOG_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
