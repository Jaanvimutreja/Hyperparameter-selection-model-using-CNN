"""
actions.py
----------
Backend action wrapper for Streamlit UI.
Runs pipeline operations in background threads so the UI doesn't freeze.
Captures stdout/stderr for live log display.
"""

import os
import sys
import subprocess
import threading
import time
from io import StringIO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Shared state for background jobs
# ---------------------------------------------------------------------------
_job_lock = threading.Lock()
_jobs = {}


class JobResult:
    """Holds the state of a background job."""
    __slots__ = ("name", "status", "output", "returncode", "start_time", "end_time")

    def __init__(self, name):
        self.name = name
        self.status = "pending"      # pending | running | done | error
        self.output = ""
        self.returncode = None
        self.start_time = None
        self.end_time = None

    @property
    def elapsed(self):
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def elapsed_str(self):
        s = self.elapsed
        if s < 60:
            return f"{s:.0f}s"
        return f"{s/60:.1f} min"


def _run_subprocess(job: JobResult, cmd: list[str]):
    """Run a subprocess and capture output into the job object."""
    job.status = "running"
    job.start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        for line in proc.stdout:
            lines.append(line)
            # Keep last 500 lines to avoid memory issues
            if len(lines) > 500:
                lines = lines[-400:]
            job.output = "".join(lines)

        proc.wait()
        job.returncode = proc.returncode
        job.status = "done" if proc.returncode == 0 else "error"
    except Exception as e:
        job.output += f"\n\nERROR: {e}"
        job.status = "error"
        job.returncode = -1
    finally:
        job.end_time = time.time()


def start_job(name: str, cmd: list[str]) -> JobResult:
    """Start a background job. Returns the JobResult immediately."""
    with _job_lock:
        job = JobResult(name)
        _jobs[name] = job

    t = threading.Thread(target=_run_subprocess, args=(job, cmd), daemon=True)
    t.start()
    return job


def get_job(name: str) -> JobResult | None:
    """Get a running or finished job by name."""
    return _jobs.get(name)


def is_job_running(name: str) -> bool:
    job = _jobs.get(name)
    return job is not None and job.status == "running"


# ---------------------------------------------------------------------------
# High-level actions
# ---------------------------------------------------------------------------

def run_full_pipeline():
    """Run the full training + evaluation pipeline."""
    return start_job("pipeline", [sys.executable, "pipeline.py"])


def run_training():
    """Train the CNN meta-model only."""
    return start_job("training", [sys.executable, "-m", "backend.train_meta_model"])


def run_evaluation():
    """Run experiment evaluation on test datasets."""
    return start_job("evaluation", [sys.executable, "-m", "experiments.evaluation"])


def run_tests():
    """Run the pytest test suite."""
    return start_job("tests", [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])


def run_verification():
    """Run the system verification script."""
    return start_job("verification", [sys.executable, "verify_pipeline.py"])
