"""
ECG Insight AI - Model Evaluation CLI & Entrypoint
Delegates to unified src.evaluation module.
"""
from src.evaluation import run_evaluation

if __name__ == "__main__":
    run_evaluation()
