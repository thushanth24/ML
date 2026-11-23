"""Command-line entry point for the real dataset (no synthetic rows)."""
# AI-assisted implementation: OpenAI ChatGPT (GPT-6), 2026-11-22. See AI_DISCLOSURE.md.

from data_utils import download_data, load_raw, prepare_data, chronological_split
from modeling import run_experiment


def main():
    source = download_data()
    print("Dataset:", source)
    raw = load_raw(source)
    print(f"Raw rows: {len(raw):,}, districts: {raw['district'].nunique()}")
    data = prepare_data(raw)
    train, validation, test = chronological_split(data)
    print(f"After feature engineering: {len(data):,} rows")
    print(f"Train 2020–2023: {len(train):,}; validate 2024: {len(validation):,}; test 2025: {len(test):,}")
    _, metrics, _ = run_experiment(train, validation, test, save=True)
    print("\nFINAL HELD-OUT 2025 SCORES (computed, not estimated):")
    print(metrics[metrics.split.eq("test_2025")][["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print("\nSaved artifacts/metrics.csv, predictions.csv and trained .joblib files")


if __name__ == "__main__":
    main()
