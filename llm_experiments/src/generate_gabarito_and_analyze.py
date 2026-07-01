import os
import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.titlesize': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
SAV_PATH = BASE_DIR / "docs/04832-percepcao-dos-brasileiros-acerca-da-democracia/04832.SAV"
df_test_path = BASE_DIR / "ml_simulation/data/df_test.csv"
llm_results_path = BASE_DIR / "llm_experiments/assets/batches/results/political_priorities_with_selected_features_e.jsonl"
gabarito_path = BASE_DIR / "llm_experiments/assets/gabarito_cesop.jsonl"
plot_output_path = BASE_DIR / "llm_experiments/assets/plots/comparison_selected_features.png"

# Ensure output directories exist
gabarito_path.parent.mkdir(parents=True, exist_ok=True)
plot_output_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Options Mapping & Code Mapping
# ---------------------------------------------------------------------------
P02_OPTIONS = list("abcdefghijkl")  # Valid option letters (a-l)

# SPSS/SAV Value to Letter Mapping
P02_CODE_TO_LETTER = {float(i): chr(ord('a') + i - 1) for i in range(1, 13)}
P02_CODE_TO_LETTER[99.0] = "m"  # Não sabe / Não respondeu

P03_CODE_TO_LETTER = {float(i): chr(ord('a') + i - 1) for i in range(1, 7)}
P03_CODE_TO_LETTER[99.0] = "g"

P04_CODE_TO_LETTER = {1.0: "a", 2.0: "b", 3.0: "c", 99.0: "d"}

# Labels Text to Option mapping (to parse df_test.csv targets)
P02_TEXT_TO_LETTER_RAW = {
    "Reduzir as desigualdades sociais": "a",
    "Combater o preconceito (racismo, homofobia, diferença de classe social, etc.)": "b",
    "Aumentar os impostos de grandes fortunas (ou dos mais ricos)": "c",
    "Incentivar a geração de empregos": "d",
    "Combater as mudanças climáticas/desmatamento": "e",
    "Ampliar o uso de energias renováveis": "f",
    "Preservar os valores ligados à família": "g",
    "Defender a igualdade entre homens e mulheres": "h",
    "Melhorar a qualidade da saúde": "i",
    "Melhorar a qualidade da educação": "j",
    "Reduzir a violência": "k",
    "Ampliar os espaços de participação política da população": "l",
    "Não sabe/ Não respondeu": "m",
    "Não sabe": "m",
    "Não respondeu": "n"
}

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = " ".join(text.split())
    # remove accents
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text

P02_TEXT_TO_LETTER = {normalize_text(k): v for k, v in P02_TEXT_TO_LETTER_RAW.items()}

# ---------------------------------------------------------------------------
# Statistics Helper Functions
# ---------------------------------------------------------------------------
def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Divergence in base 2. 0 = identical, 1 = maximum divergence."""
    eps = 1e-10
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    # jensenshannon in scipy returns square root of JSD, so we square it to get actual JSD
    return float(jensenshannon(p, q, base=2)) ** 2

def compute_cramers_v(p_llm: np.ndarray, p_real: np.ndarray, n: int = 2000):
    """Cramér's V comparing LLM distribution vs real distribution."""
    obs = np.round(p_llm * n).astype(int)
    exp = np.round(p_real * n).astype(int)
    obs = np.maximum(obs, 0)
    exp = np.maximum(exp, 0)

    table = np.array([obs, exp])
    col_sums = table.sum(axis=0)
    table = table[:, col_sums > 0]

    if table.shape[1] < 2:
        return np.nan, np.nan

    try:
        chi2, p_val, dof, _ = chi2_contingency(table)
        n_total = table.sum()
        k = min(table.shape)
        v = np.sqrt(chi2 / (n_total * (k - 1))) if (n_total * (k - 1)) > 0 else 0.0
        return float(v), float(p_val)
    except Exception:
        return np.nan, np.nan

# ---------------------------------------------------------------------------
# Step 1: Generate Gabarito from SAV
# ---------------------------------------------------------------------------
def generate_gabarito():
    print(f"Reading SAV file from: {SAV_PATH}")
    df_sav, meta_sav = pyreadstat.read_sav(str(SAV_PATH), apply_value_formats=False)
    
    print(f"Saving structured gabarito to: {gabarito_path}")
    count = 0
    with open(gabarito_path, "w", encoding="utf-8") as f:
        for idx, row in df_sav.iterrows():
            # P02 (multiple choice, up to 3)
            p02_letters = []
            for col in ["P2_1", "P2_2", "P2_3"]:
                val = row[col]
                if pd.notna(val) and val in P02_CODE_TO_LETTER:
                    p02_letters.append(P02_CODE_TO_LETTER[val])
            p02_letters = list(dict.fromkeys(p02_letters)) # Deduplicate
            
            # P03 (multiple choice, up to 6)
            p03_letters = []
            for col in [f"P3_{i}" for i in range(1, 7)]:
                val = row[col]
                if pd.notna(val) and val in P03_CODE_TO_LETTER:
                    p03_letters.append(P03_CODE_TO_LETTER[val])
            p03_letters = list(dict.fromkeys(p03_letters)) # Deduplicate
            
            # P04 (single choice)
            p04_letter = None
            val = row["P4"]
            if pd.notna(val) and val in P04_CODE_TO_LETTER:
                p04_letter = P04_CODE_TO_LETTER[val]
                
            record = {
                "id": int(row.get("ID_Ipec", idx)),
                "P02": p02_letters,
                "P03": p03_letters,
                "P04": p04_letter
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Successfully generated gabarito with {count} records.")
    return df_sav

# ---------------------------------------------------------------------------
# Step 2: Parse LLM Simulation Results
# ---------------------------------------------------------------------------
def parse_llm_results():
    print(f"Reading LLM simulation results from: {llm_results_path}")
    llm_answers = []
    
    with open(llm_results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            resp = obj.get("response", {})
            if not resp or resp.get("status_code") != 200:
                continue
            
            body = resp.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                continue
                
            content_str = choices[0]["message"]["content"]
            
            # Extract letter
            answer = ""
            try:
                content_obj = json.loads(content_str)
                answer = str(content_obj.get("answer", "")).strip().lower()
            except Exception:
                m = re.search(r'"answer"\s*:\s*"([a-nA-N])"', content_str)
                if m:
                    answer = m.group(1).lower()
            
            if answer in P02_OPTIONS or answer in ["m", "n"]:
                llm_answers.append(answer)
                
    print(f"Loaded {len(llm_answers)} LLM answers.")
    return llm_answers

# ---------------------------------------------------------------------------
# Step 3: Analyze and Plot
# ---------------------------------------------------------------------------
def run_analysis(df_sav, llm_answers):
    # Load df_test
    print(f"Loading test set from: {df_test_path}")
    df_test = pd.read_csv(df_test_path)
    
    # 1. LLM Distribution (P02)
    llm_counts = defaultdict(int)
    for ans in llm_answers:
        llm_counts[ans] += 1
    
    # 2. Real Distribution (Full SAV, Multiple Choice P2_1, P2_2, P2_3)
    real_full_mult_counts = defaultdict(int)
    for col in ["P2_1", "P2_2", "P2_3"]:
        for val in df_sav[col].dropna():
            letter = P02_CODE_TO_LETTER.get(val)
            if letter:
                real_full_mult_counts[letter] += 1
                
    # 3. Real Distribution (Full SAV, First Choice P2_1 Only)
    real_full_single_counts = defaultdict(int)
    for val in df_sav["P2_1"].dropna():
        letter = P02_CODE_TO_LETTER.get(val)
        if letter:
            real_full_single_counts[letter] += 1
            
    # 4. Real Distribution (Test Set, Multiple Choice P2_1, P2_2, P2_3)
    real_test_mult_counts = defaultdict(int)
    for col in ["P2_1", "P2_2", "P2_3"]:
        if col in df_test.columns:
            for val in df_test[col].dropna():
                if isinstance(val, str):
                    letter = P02_TEXT_TO_LETTER.get(normalize_text(val))
                else:
                    letter = P02_CODE_TO_LETTER.get(val)
                if letter:
                    real_test_mult_counts[letter] += 1
                    
    # 5. Real Distribution (Test Set, First Choice P2_1 Only)
    real_test_single_counts = defaultdict(int)
    for val in df_test["P2_1"].dropna():
        if isinstance(val, str):
            letter = P02_TEXT_TO_LETTER.get(normalize_text(val))
        else:
            letter = P02_CODE_TO_LETTER.get(val)
        if letter:
            real_test_single_counts[letter] += 1

    # Build normalized probability vectors for JSD/Cramer's V (over valid options a-l)
    def get_normalized_vector(counts_dict):
        v = np.array([counts_dict.get(o, 0.0) for o in P02_OPTIONS], dtype=float)
        s = v.sum()
        return v / s if s > 0 else v

    p_llm = get_normalized_vector(llm_counts)
    p_real_full_mult = get_normalized_vector(real_full_mult_counts)
    p_real_full_single = get_normalized_vector(real_full_single_counts)
    p_real_test_mult = get_normalized_vector(real_test_mult_counts)
    p_real_test_single = get_normalized_vector(real_test_single_counts)
    
    # Calculate metrics
    comparisons = {
        "Full SAV (Multiple Choice)": (p_real_full_mult, len(df_sav)),
        "Full SAV (First Choice Only)": (p_real_full_single, len(df_sav)),
        "Test Set (Multiple Choice)": (p_real_test_mult, len(df_test)),
        "Test Set (First Choice Only)": (p_real_test_single, len(df_test)),
    }
    
    results = []
    print("\n=== COMPARISON RESULTS (P02 - POLITICAL PRIORITIES) ===")
    for name, (p_real, n) in comparisons.items():
        jsd = compute_jsd(p_llm, p_real)
        cv, p_val = compute_cramers_v(p_llm, p_real, n=len(llm_answers))
        results.append({
            "Comparison": name,
            "JSD": jsd,
            "CramersV": cv,
            "p-value": p_val
        })
        print(f"Against {name}:")
        print(f"  JSD:        {jsd:.4f}")
        print(f"  Cramer's V: {cv:.4f} (p-value: {p_val:.4e})")
        print("-" * 50)

    # Plot distributions
    x_indices = np.arange(len(P02_OPTIONS))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Draw bars
    ax.bar(x_indices - 2*width, p_real_full_mult, width, label="Real Full (Multiple Choice)", color="#1f77b4", alpha=0.85)
    ax.bar(x_indices - width, p_real_test_mult, width, label="Real Test (Multiple Choice)", color="#aec7e8", alpha=0.85)
    ax.bar(x_indices, p_real_full_single, width, label="Real Full (First Choice Only)", color="#ff7f0e", alpha=0.85)
    ax.bar(x_indices + width, p_real_test_single, width, label="Real Test (First Choice Only)", color="#ffbb78", alpha=0.85)
    ax.bar(x_indices + 2*width, p_llm, width, label="LLM Simulation (Selected Features)", color="#2ca02c", alpha=0.9)
    
    # Formatting
    ax.set_title("P02 Option Distribution Comparison (Real vs LLM)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Questionnaire Option Letter", fontsize=13, labelpad=10)
    ax.set_ylabel("Normalized Probability", fontsize=13, labelpad=10)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(P02_OPTIONS)
    
    # Legend
    ax.legend(title="Dataset / Distribution", title_fontsize='11', fontsize='10', loc='upper right')
    
    # Details table as text on plot
    text_info = "Metrics (LLM vs Real):\n\n"
    for r in results:
        text_info += f"{r['Comparison']}:\n  JSD: {r['JSD']:.4f} | V: {r['CramersV']:.4f}\n"
    
    ax.text(0.02, 0.95, text_info, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300)
    print(f"\nPlot successfully saved to: {plot_output_path}")

if __name__ == "__main__":
    df_sav = generate_gabarito()
    llm_answers = parse_llm_results()
    run_analysis(df_sav, llm_answers)
