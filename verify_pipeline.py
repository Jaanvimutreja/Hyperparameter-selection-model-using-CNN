"""
verify_pipeline.py
-------------------
End-to-end verification script that checks every component
of the HPSM system is working correctly.

Run with:  python verify_pipeline.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ANSI colors for terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
CHECK = f"{GREEN}[OK]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"


def _status(label, passed, msg=""):
    tag = CHECK if passed else FAIL
    extra = f"  ({msg})" if msg else ""
    print(f"  {label:40s} {tag}{extra}")
    return passed


def verify():
    print("=" * 60)
    print("  HPSM — PIPELINE VERIFICATION")
    print("=" * 60)

    results = {}
    all_pass = True

    # ------------------------------------------------------------------
    # 1. Python imports
    # ------------------------------------------------------------------
    print("\n[1] Module Imports")

    modules = [
        ("backend.config", "Configuration"),
        ("backend.dataset_loader", "Dataset Loader"),
        ("backend.preprocessing", "Preprocessing"),
        ("backend.feature_extraction", "Feature Extraction"),
        ("backend.hyperparameter_search", "HP Search"),
        ("backend.cnn_model", "CNN Model"),
        ("backend.train_meta_model", "Training Pipeline"),
        ("backend.recommend", "Recommendation"),
        ("backend.baseline", "Baseline"),
        ("backend.logger", "Logger"),
        ("experiments.metrics", "Metrics"),
        ("experiments.evaluation", "Evaluation"),
        ("experiments.visualization", "Visualization"),
    ]

    for mod_name, label in modules:
        try:
            __import__(mod_name)
            ok = _status(label, True)
        except Exception as e:
            ok = _status(label, False, str(e))
        results[label] = ok
        all_pass = all_pass and ok

    # ------------------------------------------------------------------
    # 2. CNN model forward pass
    # ------------------------------------------------------------------
    print("\n[2] CNN Model")
    try:
        import torch
        from backend.cnn_model import MetaLearnerCNN, count_parameters

        model = MetaLearnerCNN()
        x = torch.randn(1, 1, 20, 20)
        out = model(x)
        ok = out.shape == (1, 36)
        _status("Forward pass (1,1,20,20) -> (1,36)", ok, f"shape={out.shape}")
        _status(f"Parameters: {count_parameters(model):,}", True)
    except Exception as e:
        _status("CNN Forward Pass", False, str(e))
        all_pass = False

    # ------------------------------------------------------------------
    # 3. Dataset loading (sklearn only — fast)
    # ------------------------------------------------------------------
    print("\n[3] Dataset Loading (sklearn)")
    try:
        from backend.dataset_loader import load_dataset
        import numpy as np

        for name in ["iris", "wine", "breast_cancer"]:
            X, y = load_dataset(name)
            ok = X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
            _status(f"  {name}", ok, f"{X.shape[0]}×{X.shape[1]}")
            all_pass = all_pass and ok
    except Exception as e:
        _status("Dataset Loading", False, str(e))
        all_pass = False

    # ------------------------------------------------------------------
    # 4. Meta-feature extraction
    # ------------------------------------------------------------------
    print("\n[4] Meta-Feature Extraction")
    try:
        from backend.feature_extraction import extract_and_reshape
        from sklearn.datasets import load_iris
        import numpy as np

        X, y = load_iris(return_X_y=True)
        matrix, names = extract_and_reshape(X, y)
        ok = matrix.shape == (20, 20) and not np.isnan(matrix).any()
        _status("Extract + reshape (iris)", ok, f"shape={matrix.shape}")
        all_pass = all_pass and ok
    except Exception as e:
        _status("Meta-Feature Extraction", False, str(e))
        all_pass = False

    # ------------------------------------------------------------------
    # 5. Preprocessing
    # ------------------------------------------------------------------
    print("\n[5] Preprocessing")
    try:
        from backend.preprocessing import clean_dataset
        from sklearn.datasets import load_iris
        import numpy as np

        X, y = load_iris(return_X_y=True)
        Xc, yc = clean_dataset(X, y)
        ok = Xc.shape == X.shape and not np.isnan(Xc).any()
        _status("clean_dataset (iris)", ok)
        all_pass = all_pass and ok
    except Exception as e:
        _status("Preprocessing", False, str(e))
        all_pass = False

    # ------------------------------------------------------------------
    # 6. Trained model
    # ------------------------------------------------------------------
    print("\n[6] Trained Model")
    model_path = os.path.join(PROJECT_ROOT, "models", "meta_cnn.pth")
    model_exists = os.path.isfile(model_path)
    _status("models/meta_cnn.pth exists", model_exists)

    model_info_path = os.path.join(PROJECT_ROOT, "models", "model_info.json")
    info_exists = os.path.isfile(model_info_path)
    if info_exists:
        with open(model_info_path) as f:
            info = json.load(f)
        _status("model_info.json", True,
                f"acc={info.get('final_accuracy', '?'):.4f}")
    else:
        _status("model_info.json", False, "not found")

    if not model_exists:
        print(f"\n  {YELLOW}→ Run 'python -m backend.pipeline' to train{RESET}")
        all_pass = False

    # ------------------------------------------------------------------
    # 7. Experiment results
    # ------------------------------------------------------------------
    print("\n[7] Experiment Results")
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    expected_files = [
        "evaluation_results.json",
        "hyperparameter_results.csv",
        "meta_features.csv",
        "cnn_training_history.csv",
        "evaluation_metrics.csv",
    ]
    for fname in expected_files:
        fpath = os.path.join(results_dir, fname)
        exists = os.path.isfile(fpath)
        _status(f"results/{fname}", exists)
        if not exists:
            all_pass = False

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    print("\n[8] Plots")
    plots_dir = os.path.join(PROJECT_ROOT, "experiments", "plots")
    expected_plots = [
        "training_curves.png",
        "accuracy_comparison.png",
        "metric_summary.png",
        "confidence_chart.png",
        "ablation_cnn_vs_random.png",
    ]
    for fname in expected_plots:
        fpath = os.path.join(plots_dir, fname)
        exists = os.path.isfile(fpath)
        _status(f"plots/{fname}", exists)
        if not exists:
            all_pass = False

    # ------------------------------------------------------------------
    # 9. Logs directory
    # ------------------------------------------------------------------
    print("\n[9] Logging")
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    _status("logs/ directory exists", os.path.isdir(log_dir))

    # ------------------------------------------------------------------
    # 10. Streamlit ready
    # ------------------------------------------------------------------
    print("\n[10] Streamlit Frontend")
    app_path = os.path.join(PROJECT_ROOT, "frontend", "app.py")
    app_exists = os.path.isfile(app_path)
    _status("frontend/app.py exists", app_exists)

    try:
        import streamlit
        _status("streamlit importable", True, f"v{streamlit.__version__}")
    except ImportError:
        _status("streamlit importable", False)
        all_pass = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    if all_pass:
        print(f"  {GREEN}ALL CHECKS PASSED ✓{RESET}")
    else:
        print(f"  {YELLOW}SOME CHECKS FAILED — see above{RESET}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
