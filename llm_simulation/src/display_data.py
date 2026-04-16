import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import os

df = pd.read_csv("llm_simulation/outputs/distribution_comparison.csv")
OUTPUT_DIR = "llm_simulation/outputs/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def wrap_labels(labels, width=20):
    return ["\n".join(textwrap.wrap(label, width)) for label in labels]

def plot_question(question, title):
    subset = df[df["question"] == question]
    subset = subset.sort_values("real_prob", ascending=False)

    x = list(range(len(subset)))

    labels = wrap_labels(subset["answer"], width=32)

    plt.figure(figsize=(16, 6))

    width = 0.4
    plt.bar([i - width/2 for i in x], subset["real_prob"], width=width, label="Real")
    plt.bar([i + width/2 for i in x], subset["llm_prob"], width=width, label="LLM")


    for i, label in enumerate(labels):
        labels[i] = label.capitalize()
        
    plt.xticks(x, labels, rotation=90, ha="center", fontsize=8)
    plt.ylabel("Probabilidade")
    plt.title(f"Distribuição de respostas - {question} ({title})")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{question}.png")
    plt.close()

def main():
    plot_question("P2_1", "Qual dessas propostas você acha que deveria ser prioridade de um(a) político(a)?")
    plot_question("P3_1", "Quais dessas opções você acredita que poderiam contribuir no combate à divulgação de fake news?")

if __name__ == "__main__":
    main()