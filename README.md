# Lightweight CNN-Based Meta-Learning Framework for Fast Hyperparameter Recommendation

A research-grade prototype that predicts optimal SVM hyperparameters for new tabular datasets instantly, by learning dataset-property → best-config mappings through a lightweight CNN meta-learner.

---

## Architecture

```
Dataset Input
    ↓
Dataset Cleaning & Preprocessing
    ↓
Meta-Feature Extraction (pymfe)
    ↓
12×12 Matrix Transformation
    ↓
CNN Meta-Learning Model (PyTorch)
    ↓
Hyperparameter Recommendation
    ↓
SVM Training with Recommended HP
    ↓
Evaluation & Results Display (Streamlit)
```

## Project Structure

```
HPSM/
├── backend/
│   ├── config.py                 # Centralized configuration
│   ├── dataset_loader.py         # Load ~20 tabular datasets
│   ├── preprocessing.py          # Data cleaning pipeline
│   ├── feature_extraction.py     # Meta-feature extraction (pymfe → 12×12)
│   ├── hyperparameter_search.py  # SVM grid search (9 configs)
│   ├── cnn_model.py              # PyTorch CNN architecture
│   ├── train_meta_model.py       # Training pipeline
│   ├── recommend.py              # Inference for new datasets
│   ├── baseline.py               # Random baseline for comparison
│   ├── pipeline.py               # Full automated pipeline
│   └── logger.py                 # Structured logging
├── experiments/
│   ├── metrics.py                # MRR, Hit Rate, accuracy metrics
│   ├── evaluation.py             # Evaluate on test datasets
│   ├── visualization.py          # Publication-quality plots
│   ├── results/                  # CSV + JSON experiment results
│   └── plots/                    # Generated charts
├── frontend/
│   └── app.py                    # Streamlit web interface
├── models/                       # Saved CNN weights + metadata
├── datasets/cache/               # Cached dataset downloads
├── logs/                         # Pipeline + training logs
├── tests/
│   ├── test_dataset_loader.py
│   ├── test_feature_extraction.py
│   └── test_model.py
├── pipeline.py                   # One-command entry point
├── verify_pipeline.py            # End-to-end verification
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Run Full Pipeline (One Command)

```bash
python pipeline.py
```

This automatically:
1. Loads 15 training datasets (sklearn / OpenML)
2. Preprocesses and standardizes data
3. Extracts meta-features for each dataset
4. Runs SVM hyperparameter search (9 configs × 5-fold CV)
5. Trains CNN meta-learner with data augmentation (~200 epochs)
6. Saves model to `models/meta_cnn.pth`
7. Evaluates on 5 held-out test datasets
8. Compares CNN vs random baseline
9. Generates publication-quality plots
10. Saves all results (CSV + JSON)

**Estimated time:** ~10-15 minutes on CPU.

### 3. Launch the Web Interface

```bash
streamlit run frontend/app.py
```

**Workflow:**
1. **Upload** a CSV dataset (last column = target)
2. **Extract** meta-features → view the 12×12 heatmap
3. **Recommend** hyperparameters with CNN confidence scores
4. **Train & Evaluate** an SVM with the recommended parameters

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Verify Everything

```bash
python verify_pipeline.py
```

Checks all components: imports, CNN, datasets, meta-features, preprocessing, model files, result CSV/JSON, plots, logs, and Streamlit readiness.

## Configuration

All tuneable parameters are centralized in `backend/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CNN_EPOCHS` | 200 | Training epochs |
| `CNN_BATCH_SIZE` | 16 | Batch size |
| `CNN_LEARNING_RATE` | 0.001 | Adam learning rate |
| `NUM_AUGMENTED` | 30 | Noise copies per sample |
| `MATRIX_SIZE` | 12 | Meta-feature matrix dim |
| `C_VALUES` | [0.1, 1, 10] | SVM C grid |
| `GAMMA_VALUES` | [0.01, 0.1, 1] | SVM gamma grid |

## Generated Output Files

After running `python pipeline.py`:

| File | Content |
|------|---------|
| `models/meta_cnn.pth` | Trained CNN weights |
| `models/model_info.json` | Model metadata |
| `experiments/results/hyperparameter_results.csv` | Best HP per training dataset |
| `experiments/results/meta_features.csv` | Extracted meta-features |
| `experiments/results/cnn_training_history.csv` | Loss + accuracy per epoch |
| `experiments/results/evaluation_metrics.csv` | Test dataset results |
| `experiments/results/evaluation_results.json` | Full evaluation detail |
| `experiments/plots/training_curves.png` | Loss & accuracy curves |
| `experiments/plots/accuracy_comparison.png` | CNN vs Random vs True Best |
| `experiments/plots/metric_summary.png` | Aggregate metrics bar chart |
| `experiments/plots/confidence_chart.png` | Per-dataset CNN confidence |
| `experiments/plots/ablation_cnn_vs_random.png` | Ablation comparison |
| `logs/pipeline.log` | Pipeline execution log |

## CNN Architecture

```
Input: (1, 12, 12)
Conv2d(1→16, 3×3, pad=1) → ReLU → MaxPool(2)    → (16, 6, 6)
Conv2d(16→32, 3×3, pad=1) → ReLU → MaxPool(2)   → (32, 3, 3)
Flatten → Linear(288, 64) → ReLU → Dropout(0.3)
Linear(64, 9)  →  9 HP configurations
```

~23,881 trainable parameters — CPU-only, lightweight.

## Hardware Requirements

- **CPU only** — no GPU required
- 8+ GB RAM recommended
- Python 3.9+
- Total pipeline time: < 20 minutes

## License

MIT License — for research and educational purposes.
