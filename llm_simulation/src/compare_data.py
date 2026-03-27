import pandas as pd
import re

# -------------------------------
# CONFIG
# -------------------------------
INPUT_PATH = "llm_simulation/outputs/llm_results.csv"
OUTPUT_DIR = "llm_simulation/outputs/"

# -------------------------------
# NORMALIZATION
# -------------------------------
def normalize(text):
    if pd.isna(text):
        return text

    text = str(text).strip().lower()

    # remove prefixos tipo "1.", "2.", "3."
    text = re.sub(r"^\d+\.\s*", "", text)

    # remove múltiplos espaços
    text = re.sub(r"\s+", " ", text)

    return text


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    df = pd.read_csv(INPUT_PATH)

    df["llm_answer"] = df["llm_answer"].apply(normalize)
    df["true_answer"] = df["true_answer"].apply(normalize)

    return df


# -------------------------------
# METRICS
# -------------------------------
def compute_accuracy(df):
    df["correct"] = df["llm_answer"] == df["true_answer"]
    return df["correct"].mean(), df


def compute_accuracy_per_question(df):
    return (
        df.groupby("question")["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )


# -------------------------------
# DISTRIBUTIONS
# -------------------------------
def compute_distribution(df):
    rows = []

    for q in df["question"].unique():
        subset = df[df["question"] == q]

        real_dist = subset["true_answer"].value_counts(normalize=True)
        llm_dist = subset["llm_answer"].value_counts(normalize=True)

        all_labels = set(real_dist.index).union(set(llm_dist.index))

        for label in all_labels:
            rows.append({
                "question": q,
                "answer": label,
                "real_pct": real_dist.get(label, 0),
                "llm_pct": llm_dist.get(label, 0)
            })

    return pd.DataFrame(rows)


# -------------------------------
# EXPORT
# -------------------------------
def export_results(df, accuracy, acc_per_q, dist_df):
    # overall
    pd.DataFrame({
        "metric": ["accuracy"],
        "value": [accuracy]
    }).to_csv(f"{OUTPUT_DIR}/overall_metrics.csv", index=False)

    # per question
    acc_per_q.to_csv(
        f"{OUTPUT_DIR}/accuracy_per_question.csv",
        index=False
    )

    # distributions
    dist_df.to_csv(
        f"{OUTPUT_DIR}/distribution_comparison.csv",
        index=False
    )

    # full comparison
    df.to_csv(
        f"{OUTPUT_DIR}/full_comparison.csv",
        index=False
    )


# -------------------------------
# MAIN
# -------------------------------
def main():
    df = load_data()

    accuracy, df = compute_accuracy(df)
    acc_per_q = compute_accuracy_per_question(df)
    dist_df = compute_distribution(df)

    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print("Accuracy per question:")
    print(acc_per_q)

    export_results(df, accuracy, acc_per_q, dist_df)
    print(f"Results exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()