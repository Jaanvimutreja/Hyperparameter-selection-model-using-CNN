"""
app.py
------
Full-control Streamlit frontend for the HPSM Meta-Learning Framework.
Includes a highly professional, interactive, and explanatory Canva-like UI.
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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (Retaining existing color scheme, upgrading Canva-like layout)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main Typography */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        font-weight: 400;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Layout Containers */
    .step-container {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border-left: 6px solid #667eea;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .step-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
    .step-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Info and Help Boxes */
    .info-box {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #764ba2;
        color: #374151;
        font-size: 0.95rem;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .highlight-text {
        font-weight: 600;
        color: #667eea;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        border: 1px solid #667eea30;
        transition: transform 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 6px 15px rgba(102, 126, 234, 0.15); }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #667eea;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #666;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Status indicators */
    .status-ok { color: #10b981; font-weight: 700; font-size: 1.1rem; }
    .status-fail { color: #ef4444; font-weight: 700; font-size: 1.1rem; }
    .status-run { color: #f59e0b; font-weight: 700; font-size: 1.1rem; }
    
    /* Tabs Redesign */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        background-color: #f8fafc;
        padding: 10px 10px 0 10px;
        border-radius: 16px 16px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #64748b;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        color: #667eea;
    }
    
    /* Expanders */
    div[data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        background-color: #fff;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 1.1rem;
        color: #334155;
    }
    
    /* Log Terminal */
    .log-box {
        background: #0f172a;
        color: #e2e8f0;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
        padding: 1.5rem;
        border-radius: 12px;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar specific
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103130.png", width=60) # placeholder logo
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">HPSM AI</div>', unsafe_allow_html=True)
    st.caption("AutoML Hyperparameter Recommender")
    st.divider()
    
    st.markdown("### 🧬 How it works")
    st.info("""
    **Zero-Shot Learning for Data**
    Instead of training hundreds of models to find the best settings, HPSM looks at your dataset's **mathematical heartbeat** and instantly recommends the best algorithm and settings based on what it learned from 150+ diverse datasets!
    """)
    
    model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    st.divider()
    if model_exists:
        st.success("✅ Brain is Active (Model Loaded)")
    else:
        st.error("❌ Brain is Offline (Train model first)")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ HPSM Meta-Learning Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Instantly predict the most optimal Machine Learning Algorithm and Hyperparameters for any tabular dataset using deep Convolutional Neural Networks (CNNs). Stop guessing, start predicting.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_about, tab_interactive_pipeline, tab_experiments, tab_knowledge, tab_logs = st.tabs([
    "🎓 Introduction & About",
    "✨ Try It: Interactive Pipeline",
    "📊 Experiment Results",
    "🧠 The Knowledge Base",
    "📋 Developer Logs"
])

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key in ["X", "y", "df", "recommendation", "meta_matrix"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =====================================================================
# TAB 1 — About & Introduction
# =====================================================================
with tab_about:
    st.markdown("""
    <div class="step-container">
    <div class="step-title">👋 Welcome to HPSM: The End of Trial-and-Error ML</div>
    
    Normally, when you get a new dataset, you spend hours or days running **Grid Search** or **Random Search** to find out which algorithm (SVM, Random Forest, etc.) and what parameters work best. 
    
    HPSM completely bypasses this using **Meta-Learning** (learning to learn). 
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 🔍 How The AI Brain Works
        1. **Feature Extraction**: We scan your dataset and extract 400 mathematical rules (called Meta-Features). These include statistical variance, class imbalance, skewness, entropy, and more.
        2. **Image Transformation**: We convert these 400 features into a **20x20 pixel heat-map image**. Your dataset literally becomes a 2D picture!
        3. **CNN Prediction**: A powerful Convolutional Neural Network (CNN) reads this "dataset picture". Because it has studied pictures of 150+ other datasets previously, it instantly predicts which of the **36 Algorithm Configurations** will yield the highest accuracy.
        """)
        
    with c2:
        st.markdown("""
        ### 🤖 Supported Algorithms
        Our AI can confidently recommend between 5 major algorithm families, completely tuned:
        - **SVM** (Support Vector Machines)
        - **RandomForest** (Ensemble trees)
        - **GradientBoosting / XGBoost**
        - **LogisticRegression**
        - **KNN** (K-Nearest Neighbors)
        
        *Are you ready? Head over to the **✨ Try It** tab to test it yourself!*
        """)

# =====================================================================
# TAB 2 — Upload & Predict (Interactive Pipeline)
# =====================================================================
def build_classifier(cfg):
    """Refactored builder for the frontend pipeline"""
    algo = cfg["algo"]
    params = cfg["params"]
    if algo == "SVM":
        return SVC(kernel="rbf", max_iter=5000, **params)
    elif algo == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif algo == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42, **params)
    elif algo == "LogisticRegression":
        return LogisticRegression(random_state=42, **params)
    elif algo == "KNN":
        return KNeighborsClassifier(**params)
    return None

with tab_interactive_pipeline:

    # --- STEP 1: UPLOAD ---
    st.markdown('<div class="step-container"><div class="step-title">📌 Step 1: Upload Your Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload any clean CSV dataset. <b>Rule:</b> The absolute last column in your file must be the target/label you want to predict.</div>', unsafe_allow_html=True)
    
    col_up1, col_up2 = st.columns([1, 1])
    with col_up1:
        uploaded = st.file_uploader("📥 Drag and drop your CSV here", type=["csv"], label_visibility="collapsed")
    
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        
        with col_up2:
            n_rows, n_cols = df.shape
            n_classes = df.iloc[:, -1].nunique()
            st.success(f"**Loaded Successfully!** \\nWe found **{n_rows:,} rows**, **{n_cols-1} features**, and **{n_classes} categories** to predict.")
            
        with st.expander("👀 Peek at your raw data"):
            st.dataframe(df.head(10), use_container_width=True)

        target = df.iloc[:, -1]
        features = df.iloc[:, :-1].copy()
        
        # Super simple auto-encoder for the frontend
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
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.X is not None:
        # --- STEP 2: META FEATURES ---
        st.markdown('<div class="step-container"><div class="step-title">🔬 Step 2: Convert to Meta-Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Now we will analyze the statistical DNA of your dataset and form a 20x20 visual matrix. This is what the AI actually "sees".</div>', unsafe_allow_html=True)
        
        if st.button("🧬 Generate Dataset DNA Matrix", use_container_width=True, type="primary"):
            with st.spinner("Crunching mathematics... Extracting Skewness, Kurtosis, Landmarking features..."):
                matrix, names = extract_and_reshape(st.session_state.X, st.session_state.y)
                st.session_state.meta_matrix = matrix
        
        if st.session_state.meta_matrix is not None:
            col_mf1, col_mf2 = st.columns([1, 2])
            with col_mf1:
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.heatmap(st.session_state.meta_matrix, cmap="mako", annot=False, ax=ax, cbar=False,
                            xticklabels=False, yticklabels=False, robust=True)
                st.pyplot(fig)
                plt.close(fig)
            with col_mf2:
                st.success("✅ **Matrix Generated!**")
                st.write("This 20x20 colorful grid represents 400 deep statistical patterns found within your rows and columns. This unique pattern is what gets fed into the neural network brain.")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- STEP 3: AI RECOMMENDATION ---
        if st.session_state.meta_matrix is not None:
            st.markdown('<div class="step-container"><div class="step-title">🧠 Step 3: Ask the AI for the Best Model</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">The CNN will evaluate the heat-map above against 36 different highly-tuned algorithm configurations to instantly predict the winner.</div>', unsafe_allow_html=True)
            
            rec_disabled = not model_exists
            if st.button("⚡ Get hyperparameter recommendation from CNN", disabled=rec_disabled, use_container_width=True, type="primary"):
                with st.spinner("CNN is evaluating 36 configurations..."):
                    result = recommend_hyperparameters(st.session_state.X, st.session_state.y)
                    st.session_state.recommendation = result
                st.success("✅ Decision Made!")
            
            if st.session_state.recommendation is not None:
                result = st.session_state.recommendation
                predicted_algo = result["predicted_algo"]
                predicted_params = result["predicted_config"]
                
                st.markdown(f"### 🏆 AI Selected: **<span style='color:#667eea;'>{predicted_algo}</span>**", unsafe_allow_html=True)
                
                # Show params in metric cards
                param_cols = st.columns(len(predicted_params) + 1)
                for idx, (k, v) in enumerate(predicted_params.items()):
                    with param_cols[idx]:
                         st.markdown(f'<div class="metric-card"><div class="metric-value">{v}</div><div class="metric-label">{k}</div></div>', unsafe_allow_html=True)
                with param_cols[-1]:
                     st.markdown(f'<div class="metric-card"><div class="metric-value">{result["confidence"]:.1%}</div><div class="metric-label">AI Confidence</div></div>', unsafe_allow_html=True)

                with st.expander("📊 View Probability Distribution across all 36 models"):
                    probs = result["all_probabilities"]
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    colors = ["#764ba2" if i == result["predicted_index"] else "#d1d5db" for i in range(NUM_CONFIGS)]
                    ax.bar(range(NUM_CONFIGS), probs, color=colors)
                    ax.set_ylabel("Probability")
                    ax.set_xlabel("Algorithm Configurations (Index 0 to 35)")
                    st.pyplot(fig)
                    plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- STEP 4: TRAIN AND VERIFY ---
            if st.session_state.recommendation is not None:
                st.markdown('<div class="step-container"><div class="step-title">🎯 Step 4: Prove it (Train & Evaluate)</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">Let\'s actually train the AI\'s recommended model heavily on your data to see if the hypothesis was right, and evaluate the final real-world accuracy.</div>', unsafe_allow_html=True)
                
                if st.button("🚀 Train Recommended Model Live", use_container_width=True, type="primary"):
                    res_rec = st.session_state.recommendation
                    full_cfg = {"algo": res_rec["predicted_algo"], "params": res_rec["predicted_config"]}
                    with st.spinner(f"Training {full_cfg['algo']}... Please wait (this is actually doing the hard work on your data now)."):
                        X_tr, X_te, y_tr, y_te = train_test_split(st.session_state.X, st.session_state.y, test_size=0.2, random_state=42, stratify=st.session_state.y)
                        clf = build_classifier(full_cfg)
                        clf.fit(X_tr, y_tr)
                        train_acc = clf.score(X_tr, y_tr)
                        test_acc = clf.score(X_te, y_te)
                        y_pred = clf.predict(X_te)
                        st.session_state["live_results"] = {
                            "train_acc": train_acc, "test_acc": test_acc,
                            "y_test": y_te, "y_pred": y_pred, "cfg": full_cfg
                        }
                
                if "live_results" in st.session_state:
                    res = st.session_state["live_results"]
                    c_res1, c_res2 = st.columns(2)
                    with c_res1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{res["train_acc"]:.2%}</div><div class="metric-label">Training Accuracy</div></div>', unsafe_allow_html=True)
                    with c_res2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{res["test_acc"]:.2%}</div><div class="metric-label">Testing (Unseen) Accuracy</div></div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("Show detailed Classification Report"):
                        report = classification_report(res["y_test"], res["y_pred"], output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# TAB 4 — Experiments
# =====================================================================
with tab_experiments:
    eval_json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if not os.path.exists(eval_json_path):
        st.warning("⚠️ No experiments have been finalized. Please generate data first from the Controls tab.")
    else:
        with open(eval_json_path) as f:
            eval_data = json.load(f)

        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("📊 Global AI Performance Benchmarks")
        st.markdown('<div class="info-box">These metrics demonstrate how perfectly the CNN was able to predict the EXACT best model settings compared to an exhaustive mathematical brute-force search.</div>', unsafe_allow_html=True)
        
        agg = eval_data.get("aggregate", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{agg.get("recommendation_accuracy", 0):.0%}</div><div class="metric-label">Algorithm Hit Rate</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{agg.get("mrr", 0):.2f}</div><div class="metric-label">Mean Reciprocal Rank</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{agg.get("hit_rate_at_3", 0):.0%}</div><div class="metric-label">Top-3 Hit Rate</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-value">{agg.get("mean_cnn_regret", 0):.4f}</div><div class="metric-label">Regret (Error Margin)</div></div>', unsafe_allow_html=True)
        
        st.subheader("High-Resolution Charts")
        plot_files = {"Loss Curve": "training_curves.png", "Prediction Impact": "accuracy_comparison.png", "AI Confidence Grid": "confidence_chart.png", "Ablation Engine": "ablation_cnn_vs_random.png"}
        ptabs = st.tabs(list(plot_files.keys()))
        for t, (name, fname) in zip(ptabs, plot_files.items()):
            path = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(path):
                t.image(path, use_container_width=True)
            else:
                t.info(f"Visual {fname} is tracking offline.")
                
        st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# TAB 5 — Knowledge Base
# =====================================================================
with tab_knowledge:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🧠 Exploring The Knowledge Base")
    st.markdown('<div class="info-box">The Knowledge Base acts as the "Long Term Memory" for HPSM. Every dataset it studies is documented here along with its absolute optimal algorithm so that the Nearest-Neighbor algorithms can help with prediction.</div>', unsafe_allow_html=True)

    from backend.knowledge_base import load_knowledge_base, get_summary
    kb = load_knowledge_base()
    if not kb:
        st.info("Knowledge engine is currently empty.")
    else:
        summary = get_summary(kb)
        st.success(f"Successfully connected to long-term memory. Scanning **{summary['total_datasets']} Datasets**.")
        rows = [{"Dataset Registry ID": name, "Supreme Algorithm": entry["best_algo"], "Optimized Configurations": str(entry["best_params"]), "Max Accuracy": f'{entry["best_accuracy"]:.4f}'} for name, entry in kb.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# TAB 6 — Logs
# =====================================================================
with tab_logs:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Terminal Core Telemetry")
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    if not log_files:
        st.info("System quiet. Awaiting processing logs.")
    else:
        ltabs = st.tabs([os.path.basename(f) for f in sorted(log_files)])
        for t, fpath in zip(ltabs, sorted(log_files)):
            with t:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    text = "".join(lines[-300:])
                    st.markdown(f'<div class="log-box">{text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Read error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

