"""
app.py
------
Full-control Streamlit frontend for the HPSM Meta-Learning Framework.

Tabs:
  1. Upload Dataset — upload CSV, preview, recommend HP, train & evaluate
  2. System Controls — run pipeline, train, evaluate, test, verify
  3. Experiments — view plots and result tables
  4. Logs — live log viewer
  5. About — project info

Run with:  streamlit run frontend/app.py
"""

import os
import sys
import time
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.feature_extraction import extract_and_reshape, MATRIX_SIZE
from backend.hyperparameter_search import PARAM_GRID, get_config_by_index, NUM_CONFIGS
from backend.recommend import recommend_hyperparameters, recommend_top_k, DEFAULT_MODEL_PATH
from frontend.actions import (
    run_full_pipeline, run_training, run_evaluation,
    run_tests, run_verification, get_job, is_job_running,
)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Paths
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "plots")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_INFO_PATH = os.path.join(PROJECT_ROOT, "models", "model_info.json")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HPSM — Hyperparameter Meta-Learner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea18 0%, #764ba218 100%);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid #667eea30;
        transition: transform 0.15s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #999;
        margin-top: 0.2rem;
    }
    .status-ok { color: #2ecc71; font-weight: 600; }
    .status-fail { color: #e74c3c; font-weight: 600; }
    .status-run { color: #f39c12; font-weight: 600; }
    .control-btn button {
        width: 100%;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 22px;
        font-weight: 500;
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #eee;
    }
    .log-box {
        background: #1e1e1e;
        color: #d4d4d4;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.78rem;
        padding: 1rem;
        border-radius: 10px;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ HPSM — Hyperparameter Meta-Learner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Lightweight CNN-Based Meta-Learning for Fast Hyperparameter Recommendation</div>', unsafe_allow_html=True)

model_exists = os.path.exists(DEFAULT_MODEL_PATH)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_controls, tab_experiments, tab_logs, tab_knowledge, tab_about = st.tabs([
    "📂 Upload & Predict",
    "🎛️ System Controls",
    "📊 Experiments",
    "📋 Logs",
    "🧠 Knowledge Base",
    "ℹ️ About",
])

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key in ["X", "y", "df", "recommendation", "meta_matrix"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =====================================================================
# TAB 1 — Upload & Predict
# =====================================================================
with tab_upload:

    upload_col, summary_col = st.columns([1.2, 0.8])

    with upload_col:
        st.header("Upload Your Dataset")
        st.caption("Upload a CSV file. The **last column** is the target variable.")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.df = df

        with summary_col:
            st.header("Summary")
            n_rows, n_cols = df.shape
            target_col = df.columns[-1]
            n_classes = df[target_col].nunique()

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Rows", f"{n_rows:,}")
            mc2.metric("Cols", n_cols)
            mc3.metric("Features", n_cols - 1)
            mc4.metric("Classes", n_classes)

        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True, height=280)

        # Prepare X, y
        target = df.iloc[:, -1]
        features = df.iloc[:, :-1].copy()
        for col in features.columns:
            if features[col].dtype == object or str(features[col].dtype) == "category":
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
        if target.dtype == object or str(target.dtype) == "category":
            le = LabelEncoder()
            target = le.fit_transform(target.astype(str))
        X = np.nan_to_num(features.values.astype(np.float64))
        y = np.array(target, dtype=np.int64)
        X = StandardScaler().fit_transform(X)
        st.session_state.X = X
        st.session_state.y = y

        st.success(f"✅ Dataset loaded: {n_rows:,} samples, {X.shape[1]} features, {n_classes} classes")

        # ----- Action buttons row -----
        st.divider()
        act1, act2, act3 = st.columns(3)

        # --- Meta-features ---
        with act1:
            if st.button("🔬 Extract Meta-Features", use_container_width=True):
                with st.spinner("Extracting meta-features…"):
                    matrix, names = extract_and_reshape(X, y)
                    st.session_state.meta_matrix = matrix
                st.success(f"✅ {MATRIX_SIZE}×{MATRIX_SIZE} matrix ready")

        # --- Recommend ---
        with act2:
            rec_disabled = not model_exists
            if st.button("⚡ Recommend HP", disabled=rec_disabled, use_container_width=True):
                with st.spinner("Running CNN inference…"):
                    result = recommend_hyperparameters(X, y)
                    st.session_state.recommendation = result
                st.success("✅ Recommendation ready")
            if rec_disabled:
                st.caption("⚠ Train the model first")

        # --- Train SVM ---
        with act3:
            svm_disabled = st.session_state.recommendation is None
            if st.button("🎯 Train SVM", disabled=svm_disabled, use_container_width=True):
                cfg = st.session_state.recommendation["predicted_config"]
                with st.spinner(f"Training SVM (C={cfg['C']}, γ={cfg['gamma']})…"):
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    clf = SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf", max_iter=5000)
                    clf.fit(X_tr, y_tr)
                    train_acc = clf.score(X_tr, y_tr)
                    test_acc = clf.score(X_te, y_te)
                    y_pred = clf.predict(X_te)
                st.session_state["svm_results"] = {
                    "train_acc": train_acc, "test_acc": test_acc,
                    "y_test": y_te, "y_pred": y_pred, "cfg": cfg
                }
                st.success("✅ Training done")
            if svm_disabled:
                st.caption("⚠ Get recommendation first")

        # ----- Results display -----
        # Meta-feature heatmap
        if st.session_state.meta_matrix is not None:
            with st.expander("🔬 Meta-Feature Heatmap", expanded=False):
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(st.session_state.meta_matrix, cmap="viridis", annot=False, ax=ax,
                            xticklabels=False, yticklabels=False)
                ax.set_title("12 × 12 Meta-Feature Matrix")
                st.pyplot(fig)
                plt.close(fig)

        # Recommendation
        if st.session_state.recommendation is not None:
            result = st.session_state.recommendation
            cfg = result["predicted_config"]

            with st.expander("⚡ Hyperparameter Recommendation", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{cfg["C"]}</div>'
                                f'<div class="metric-label">C (Regularization)</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{cfg["gamma"]}</div>'
                                f'<div class="metric-label">gamma (Kernel coeff.)</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{result["confidence"]:.1%}</div>'
                                f'<div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)

                # Confidence bar chart
                probs = result["all_probabilities"]
                config_labels = [f"C={p['C']}, γ={p['gamma']}" for p in PARAM_GRID]
                fig, ax = plt.subplots(figsize=(9, 3.5))
                colors = ["#667eea" if i == result["predicted_index"] else "#ddd" for i in range(NUM_CONFIGS)]
                ax.bar(config_labels, probs, color=colors, edgecolor="#666", linewidth=0.4)
                ax.set_ylabel("Probability")
                ax.set_title("CNN Confidence per Configuration")
                plt.xticks(rotation=40, ha="right", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Top-3 table
                top3, _ = recommend_top_k(X, y, k=3)
                st.markdown("**Top 3 Recommendations**")
                st.table(pd.DataFrame([
                    {"Rank": r["rank"], "C": r["config"]["C"], "gamma": r["config"]["gamma"],
                     "Probability": f'{r["probability"]:.4f}'}
                    for r in top3
                ]))

                if "nearest_datasets" in result and result["nearest_datasets"]:
                    st.markdown("**Nearest Known Datasets**")
                    st.table(pd.DataFrame([
                        {"Dataset": d["name"], "Similarity": f'{d["similarity"]:.4f}', 
                         "Best Algo": d["best_algo"], "Accuracy": f'{d["best_accuracy"]:.4f}'}
                        for d in result["nearest_datasets"]
                    ]))

        # SVM results
        if st.session_state.get("svm_results"):
            res = st.session_state["svm_results"]
            with st.expander("🎯 SVM Training Results", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{res["train_acc"]:.1%}</div>'
                                f'<div class="metric-label">Train Accuracy</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{res["test_acc"]:.1%}</div>'
                                f'<div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(res["y_test"], res["y_pred"])
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                    st.pyplot(fig); plt.close(fig)
                with col2:
                    st.markdown("**Classification Report**")
                    report = classification_report(res["y_test"], res["y_pred"], output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)

                # Cross-validation
                st.markdown("**5-Fold Cross-Validation**")
                cv_scores = cross_val_score(
                    SVC(C=res["cfg"]["C"], gamma=res["cfg"]["gamma"], kernel="rbf", max_iter=5000),
                    X, y, cv=5, scoring="accuracy"
                )
                st.bar_chart(pd.DataFrame({"Accuracy": cv_scores}, index=[f"Fold {i+1}" for i in range(5)]))
                st.write(f"**Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}**")

    else:
        st.info("👆 Upload a CSV file to get started.")


# =====================================================================
# TAB 2 — System Controls
# =====================================================================
with tab_controls:
    st.header("System Controls")
    st.caption("Run pipeline operations directly from the UI. Long tasks run in background threads.")

    # Model status cards
    st.subheader("System Status")
    s1, s2, s3 = st.columns(3)
    with s1:
        if model_exists:
            st.markdown('<div class="metric-card"><div class="status-ok">● Model Trained</div>'
                        '<div class="metric-label">meta_cnn.pth exists</div></div>', unsafe_allow_html=True)
            if os.path.exists(MODEL_INFO_PATH):
                with open(MODEL_INFO_PATH) as f:
                    info = json.load(f)
                st.caption(f"Accuracy: {info.get('final_accuracy', 0):.1%} · Params: {info.get('parameters', 0):,}")
        else:
            st.markdown('<div class="metric-card"><div class="status-fail">● No Model</div>'
                        '<div class="metric-label">Run pipeline first</div></div>', unsafe_allow_html=True)

    with s2:
        results_exist = os.path.exists(os.path.join(RESULTS_DIR, "evaluation_results.json"))
        if results_exist:
            st.markdown('<div class="metric-card"><div class="status-ok">● Results Ready</div>'
                        '<div class="metric-label">Evaluation completed</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="status-fail">● No Results</div>'
                        '<div class="metric-label">Run evaluation</div></div>', unsafe_allow_html=True)

    with s3:
        plots_exist = len(glob.glob(os.path.join(PLOTS_DIR, "*.png"))) >= 5
        if plots_exist:
            st.markdown('<div class="metric-card"><div class="status-ok">● Plots Generated</div>'
                        '<div class="metric-label">5 charts ready</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="status-fail">● No Plots</div>'
                        '<div class="metric-label">Run pipeline</div></div>', unsafe_allow_html=True)

    st.divider()

    # ----- Control buttons -----
    st.subheader("Actions")

    # Helper to render a job button + status
    def _render_job_button(label, icon, job_name, action_fn, description, col):
        with col:
            running = is_job_running(job_name)
            btn_label = f"⏳ {label} running…" if running else f"{icon} {label}"
            if st.button(btn_label, disabled=running, use_container_width=True, key=f"btn_{job_name}"):
                action_fn()
                st.rerun()

            st.caption(description)

            job = get_job(job_name)
            if job:
                if job.status == "running":
                    st.markdown(f'<span class="status-run">● Running ({job.elapsed_str})</span>', unsafe_allow_html=True)
                    st.code(job.output[-800:] if job.output else "Starting…", language="text")
                    time.sleep(2)
                    st.rerun()
                elif job.status == "done":
                    st.markdown(f'<span class="status-ok">✓ Completed ({job.elapsed_str})</span>', unsafe_allow_html=True)
                    with st.expander("View output", expanded=False):
                        st.code(job.output[-2000:] if job.output else "No output", language="text")
                elif job.status == "error":
                    st.markdown(f'<span class="status-fail">✗ Failed (code {job.returncode})</span>', unsafe_allow_html=True)
                    with st.expander("View error output", expanded=True):
                        st.code(job.output[-2000:] if job.output else "No output", language="text")

    c1, c2 = st.columns(2)
    _render_job_button("Run Full Pipeline", "🚀", "pipeline", run_full_pipeline,
                       "Load → Train → Evaluate → Plot (all 10 steps)", c1)
    _render_job_button("Train Model", "🧠", "training", run_training,
                       "Train CNN meta-learner on training datasets", c2)

    c3, c4 = st.columns(2)
    _render_job_button("Run Evaluation", "📊", "evaluation", run_evaluation,
                       "Evaluate on test datasets + generate plots", c3)
    _render_job_button("Run Tests", "🧪", "tests", run_tests,
                       "Run pytest test suite (25 tests)", c4)

    c5, _ = st.columns(2)
    _render_job_button("Verify System", "✅", "verification", run_verification,
                       "Check all components are working", c5)


# =====================================================================
# TAB 3 — Experiments
# =====================================================================
with tab_experiments:
    st.header("Experiment Results")

    # Check for results
    eval_json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")

    if not os.path.exists(eval_json_path):
        st.warning("⚠ No experiment results found. Run the pipeline first via **System Controls** tab.")
    else:
        with open(eval_json_path) as f:
            eval_data = json.load(f)

        # Aggregate metrics
        agg = eval_data.get("aggregate", {})
        st.subheader("Aggregate Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{agg.get("recommendation_accuracy", 0):.0%}</div>'
                        f'<div class="metric-label">Rec. Accuracy</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{agg.get("mrr", 0):.2f}</div>'
                        f'<div class="metric-label">MRR</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{agg.get("hit_rate_at_1", 0):.0%}</div>'
                        f'<div class="metric-label">Hit@1</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{agg.get("hit_rate_at_3", 0):.0%}</div>'
                        f'<div class="metric-label">Hit@3</div></div>', unsafe_allow_html=True)

        st.divider()

        # Per-dataset results table
        st.subheader("Per-Dataset Results")
        per = eval_data.get("per_dataset", {})
        if per:
            rows = []
            for name, info in per.items():
                rows.append({
                    "Dataset": name,
                    "True Best Acc": f"{info['true_best_accuracy']:.4f}",
                    "CNN Pred Acc": f"{info['pred_accuracy']:.4f}",
                    "Random Avg": f"{info['random_mean_accuracy']:.4f}",
                    "Confidence": f"{info['pred_confidence']:.1%}",
                    "Match": "✅" if info["match"] else "❌",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # Plots
        st.subheader("Generated Plots")

        plot_files = {
            "Training Curves": "training_curves.png",
            "Accuracy Comparison": "accuracy_comparison.png",
            "Metric Summary": "metric_summary.png",
            "CNN Confidence": "confidence_chart.png",
            "Ablation: CNN vs Random": "ablation_cnn_vs_random.png",
        }

        plot_tabs = st.tabs(list(plot_files.keys()))
        for tab, (label, fname) in zip(plot_tabs, plot_files.items()):
            with tab:
                fpath = os.path.join(PLOTS_DIR, fname)
                if os.path.exists(fpath):
                    st.image(fpath, use_container_width=True)
                else:
                    st.info(f"Plot not found: {fname}")

        st.divider()

        # CSV downloads
        st.subheader("Download Results")
        csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
        if csv_files:
            for fpath in sorted(csv_files):
                fname = os.path.basename(fpath)
                with open(fpath, "rb") as f:
                    st.download_button(
                        f"📥 {fname}",
                        f.read(),
                        file_name=fname,
                        mime="text/csv",
                        key=f"dl_{fname}",
                    )


# =====================================================================
# TAB 4 — Logs
# =====================================================================
with tab_logs:
    st.header("Pipeline Logs")

    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))

    if not log_files:
        st.info("No log files found. Run the pipeline to generate logs.")
    else:
        log_tabs = st.tabs([os.path.basename(f) for f in sorted(log_files)])
        for tab, fpath in zip(log_tabs, sorted(log_files)):
            with tab:
                if st.button(f"🔄 Refresh", key=f"refresh_{os.path.basename(fpath)}"):
                    st.rerun()
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    # Show last 200 lines
                    lines = content.strip().split("\n")
                    if len(lines) > 200:
                        st.caption(f"Showing last 200 of {len(lines)} lines")
                        content = "\n".join(lines[-200:])
                    st.markdown(f'<div class="log-box">{content}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not read log: {e}")


# =====================================================================
# TAB 5 — Knowledge Base
# =====================================================================
with tab_knowledge:
    st.header("Meta-Knowledge Base")
    st.caption("Historical dataset performance used for similarity-based recommendations.")

    from backend.knowledge_base import load_knowledge_base, get_summary
    kb = load_knowledge_base()

    if not kb:
        st.info("Knowledge base is empty. Run the pipeline to populate it.")
    else:
        summary = get_summary(kb)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{summary["total_datasets"]}</div>'
                        f'<div class="metric-label">Datasets Indexed</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(summary["algo_distribution"])}</div>'
                        f'<div class="metric-label">Algorithms Selected</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{summary["mean_accuracy"]:.1%}</div>'
                        f'<div class="metric-label">Mean Accuracy</div></div>', unsafe_allow_html=True)

        st.subheader("Algorithm Distribution")
        dist = summary["algo_distribution"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(list(dist.keys()), list(dist.values()), color="#667eea")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Dataset Registry")
        rows = []
        for name, entry in kb.items():
            rows.append({
                "Dataset": name,
                "Best Algorithm": entry["best_algo"],
                "Best Params": str(entry["best_params"]),
                "Best Accuracy": f'{entry["best_accuracy"]:.4f}'
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =====================================================================
# TAB 6 — About
# =====================================================================
with tab_about:
    st.header("About This Project")

    st.markdown("""
    ### Lightweight CNN-Based Meta-Learning Framework

    **Hyperparameter tuning** is one of the most expensive steps in machine learning.
    This project implements a **meta-learning approach** — learning relationships between
    dataset properties and optimal hyperparameters across many datasets, then predicting
    good hyperparameters for new datasets **instantly**.

    ---

    ### System Pipeline

    ```
    Dataset Input
        ↓
    Preprocessing & Standardization
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
    Evaluation & Visualization
    ```

    ---

    ### Architecture

    | Component | Technology |
    |-----------|-----------|
    | Meta-features | pymfe |
    | Meta-learner | PyTorch CNN (23,881 params) |
    | Base classifier | scikit-learn SVM |
    | Frontend | Streamlit |
    | Data sources | OpenML, sklearn |
    | Visualization | matplotlib, seaborn |

    ---

    ### Hardware

    - **CPU only** — no GPU required
    - 8+ GB RAM recommended
    - Pipeline runs in ~3-4 minutes
    """)

    st.divider()

    # Model info if available
    if os.path.exists(MODEL_INFO_PATH):
        st.subheader("Trained Model Info")
        with open(MODEL_INFO_PATH) as f:
            info = json.load(f)
        st.json(info)

    st.info("Built as a research prototype for fast hyperparameter recommendation.")
