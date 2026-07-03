import pandas as pd
import re
from scipy.spatial.distance import jensenshannon

# -------------------------------
# CONFIG
# -------------------------------
MODEL = "qwen3"
INPUT_PATH = "llm_simulation/outputs/"
OUTPUT_DIR = f"llm_simulation/outputs/{MODEL}/"

# -------------------------------
# JSD + DISTRIBUTION (UNIFICADO)
# -------------------------------
def compute_jsd_and_distribution(df):
    jsd_rows = []
    dist_rows = []

    for q in df["question"].unique():
        subset = df[df["question"] == q]

        real = subset["true_answer"].value_counts(normalize=True)
        llm = subset["llm_answer"].value_counts(normalize=True)

        all_labels = sorted(set(real.index).union(set(llm.index)))

        real = real.reindex(all_labels, fill_value=0)
        llm = llm.reindex(all_labels, fill_value=0)

        # JSD real (sem raiz)
        jsd_value = jensenshannon(real.values, llm.values) ** 2

        jsd_rows.append({
            "question": q,
            "jsd": jsd_value
        })

        for label in all_labels:
            dist_rows.append({
                "question": q,
                "answer": label,
                "real_prob": real[label],
                "llm_prob": llm[label]
            })

    return pd.DataFrame(jsd_rows), pd.DataFrame(dist_rows)

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
def load_data(prompt_style):
    df = pd.read_csv(f"{INPUT_PATH}/{MODEL}_{prompt_style}.csv")

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
# EXPORT
# -------------------------------
def export_results(df, accuracy, acc_per_q, dist_df, jsd_df, prompt_style):
    # overall
    pd.DataFrame({
        "metric": ["accuracy"],
        "value": [accuracy]
    }).to_csv(f"{OUTPUT_DIR}/{prompt_style}/overall_metrics.csv", index=False)

    # per question
    acc_per_q.to_csv(
        f"{OUTPUT_DIR}/{prompt_style}/accuracy_per_question.csv",
        index=False
    )

    # distributions
    dist_df.to_csv(
        f"{OUTPUT_DIR}/{prompt_style}/distribution_comparison.csv",
        index=False
    )

    # full comparison
    df.to_csv(
        f"{OUTPUT_DIR}/{prompt_style}/full_comparison.csv",
        index=False
    )

    # JSD
    jsd_df.to_csv(
        f"{OUTPUT_DIR}/{prompt_style}/jsd_per_question.csv",
        index=False
    )


# -------------------------------
# MAIN
# -------------------------------
def main():
    print(f"Model - {MODEL}")
    for prompt_style in [
    "natural",
    "key_value",
    "markdown",
    "json_prompt",
]:
        print(f"\n\n\Prompt style - {prompt_style}")
        df = load_data(prompt_style)

        accuracy, df = compute_accuracy(df)
        acc_per_q = compute_accuracy_per_question(df)
        jsd_df, dist_df = compute_jsd_and_distribution(df)
        print("\nJSD per question:")
        print(jsd_df)

        print(f"Overall Accuracy: {accuracy:.4f}\n")
        print("Accuracy per question:")
        print(acc_per_q)

        export_results(df, accuracy, acc_per_q, dist_df, jsd_df, prompt_style)
        print(f"Results exported to {OUTPUT_DIR}/{prompt_style}")


if __name__ == "__main__":
    main()