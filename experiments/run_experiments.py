"""
run_experiments.py
------------------
Full benchmark experiment runner.
Compares CNN meta-learner against multiple baselines and generates LaTeX tables.
Results are saved with timestamps.

Run with:  python -m experiments.run_experiments
"""

import os
import sys
import json
import time
from datetime import datetime
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.evaluation import run_full_benchmark
from backend.config import RESULTS_DIR

def generate_latex_table(results_df):
    """Generate a LaTeX table from benchmark results."""
    latex_str = results_df.to_latex(
        index=False,
        float_format="%.4f",
        columns=["dataset", "best_config", "cnn_config", "cnn_accuracy", "random_accuracy"],
        caption="Benchmark Results: CNN vs Random Baseline",
        escape=False
    )
    return latex_str

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run the benchmark from evaluation.py
    run_full_benchmark()
    
    # Load the results
    result_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(result_path, "r") as f:
        data = json.load(f)
    
    results = data["results"]
    summary = data["summary"]
    
    # Save timestamped JSON
    timestamped_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    with open(timestamped_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Timestamped results saved to: {os.path.basename(timestamped_path)}")
    
    # Generate LaTeX table
    df = pd.DataFrame(results)
    
    try:
        latex_table = generate_latex_table(df)
        latex_path = os.path.join(RESULTS_DIR, f"benchmark_table_{timestamp}.tex")
        with open(latex_path, "w") as f:
            f.write(latex_table)
        print(f"  LaTeX table saved to: {os.path.basename(latex_path)}")
    except Exception as e:
        print(f"  Could not generate LaTeX table: {e}")

if __name__ == "__main__":
    main()
