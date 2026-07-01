"""
Análise de Respostas LLM – Percepção dos Brasileiros Acerca da Democracia
=========================================================================
1. Extrai estatísticas das respostas para cada LLM (P02, P03, P04)
2. Calcula JSD (Jensen-Shannon Divergence) vs. dados reais (SPSS .SAV)
3. Calcula Cramér's V vs. dados reais
4. Compara desempenho entre modelos

Notas sobre o SAV:
  - P02 → colunas P2_1, P2_2, P2_3  (até 3 escolhas; código 1-12, 99=NS/NR)
  - P03 → colunas P3_1…P3_6         (até 6 escolhas; código 1-6, 99=NS/NR)
  - P04 → coluna  P4                 (única escolha; código 1-3, 99=NS/NR)

Notas sobre os LLMs:
  - sabia-4 / sabiazinho-4 / gpt-5-mini: content é JSON {"answer": "x", ...}
  - llama-3.2-3b: content começa com '"<letra>"\n...'
"""

import json
import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyreadstat
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE / "llm_experiments/assets/surveys/results/percepcao_democracia"
SAV_PATH    = BASE / "docs/04832-percepcao-dos-brasileiros-acerca-da-democracia/04832.SAV"

MODELS = {
    "sabia-4":      RESULTS_DIR / "batch_result-sabia-4.jsonl",
    "sabiazinho-4": RESULTS_DIR / "batch_result_sabiazinho-4.jsonl",
    "gpt-5-mini":   RESULTS_DIR / "batch_result_gpt-5-mini.jsonl",
    "llama-3.2-3b": RESULTS_DIR / "batch-result-llama.jsonl",
}

QUESTIONS = ["P02", "P03", "P04"]

# Letras válidas por pergunta (excluindo NS/NR que ficam como 'm','n' no P02 etc.)
QUESTION_OPTIONS = {
    "P02": list("abcdefghijkl"),   # 12 opções válidas (a-l); m=NS, n=NR
    "P03": list("abcdef"),         # 6 opções (a-f); g=NS, h=NR
    "P04": list("abc"),            # 3 opções (a-c); d=NS, e=NR
}

# Colunas SAV para respostas múltiplas
SAV_P02_COLS = ["P2_1", "P2_2", "P2_3"]
SAV_P03_COLS = ["P3_1", "P3_2", "P3_3", "P3_4", "P3_5", "P3_6"]
SAV_P04_COL  = "P4"

# Mapeamento código SAV → letra da survey
# P02: código 1-12 → a-l; 99 → NS/NR
P02_CODE_TO_LETTER = {float(i): chr(ord('a') + i - 1) for i in range(1, 13)}
P02_CODE_TO_LETTER[99.0] = "m"  # NS/NR

# P03: código 1-6 → a-f; 99 → NS/NR
P03_CODE_TO_LETTER = {float(i): chr(ord('a') + i - 1) for i in range(1, 7)}
P03_CODE_TO_LETTER[99.0] = "g"  # NS/NR

# P04: código 1-3 → a-c; 99 → NS/NR
P04_CODE_TO_LETTER = {1.0: "a", 2.0: "b", 3.0: "c", 99.0: "d"}


# ---------------------------------------------------------------------------
# 1. Distribuições Reais (SAV)
# ---------------------------------------------------------------------------

def load_sav(path: Path):
    df, meta = pyreadstat.read_sav(str(path), apply_value_formats=False)
    return df, meta


def real_distribution_multiple(df: pd.DataFrame, cols: list, code_map: dict,
                                 valid_letters: list, n_respondents: int):
    """
    Para perguntas de resposta múltipla (P02, P03):
    cada respondente pode escolher até N opções.
    Retorna proporção de respondentes que escolheu cada opção.
    """
    total_resp = len(df)
    counts = defaultdict(int)
    for col in cols:
        for val in df[col].dropna():
            letter = code_map.get(val)
            if letter:
                counts[letter] += 1

    dist = {}
    for letter in valid_letters:
        dist[letter] = counts.get(letter, 0) / total_resp
    return dist, total_resp


def real_distribution_single(df: pd.DataFrame, col: str, code_map: dict,
                               valid_letters: list):
    """
    Para perguntas de resposta única (P04).
    Retorna proporção por opção (excluindo NS/NR na normalização opcional).
    """
    total_resp = len(df[col].dropna())
    counts = df[col].dropna().value_counts().to_dict()
    dist = {}
    for code, letter in code_map.items():
        if letter in valid_letters:
            dist[letter] = counts.get(code, 0) / total_resp
    return dist, total_resp


# ---------------------------------------------------------------------------
# 2. Parser de JSONL dos LLMs
# ---------------------------------------------------------------------------

def extract_answer_llama(content: str) -> str:
    """
    llama: content começa com '"<letra>"\n...'
    Extrai a letra usando regex.
    """
    m = re.match(r'^\s*["`]?([a-nA-N])["`]?\s*[\n,]?', content)
    if m:
        return m.group(1).lower()
    # fallback: busca JSON
    try:
        obj = json.loads(content)
        return str(obj.get("answer", "")).strip().lower()
    except Exception:
        return ""


def extract_answer_json(content: str) -> str:
    """
    sabia / gpt: content é JSON {"answer": "x", ...}
    """
    try:
        obj = json.loads(content)
        return str(obj.get("answer", "")).strip().lower()
    except Exception:
        # fallback regex
        m = re.search(r'"answer"\s*:\s*"([a-nA-N])"', content)
        if m:
            return m.group(1).lower()
        return ""


def parse_jsonl(path: Path, model_name: str):
    """
    Lê arquivo JSONL de resultado de batch e extrai:
      - question (P02/P03/P04)
      - rep (1-5)
      - answer (letra)
    """
    is_llama = "llama" in model_name.lower()
    records = []
    errors = 0
    total_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
                custom_id = obj.get("custom_id", "")

                # Extrai question do custom_id: <uuid>_P02_rep1
                # formato: <uuid>_P0X_repN
                m = re.search(r'_(P0[234])_rep(\d+)$', custom_id)
                if not m:
                    errors += 1
                    continue
                question = m.group(1)
                rep = m.group(2)

                # Extrai content
                resp = obj.get("response", {})
                if not resp:
                    errors += 1
                    continue
                body = resp.get("body", {})
                if not body:
                    errors += 1
                    continue
                choices = body.get("choices", [])
                if not choices:
                    errors += 1
                    continue

                content_str = choices[0]["message"]["content"]
                if is_llama:
                    answer = extract_answer_llama(content_str)
                else:
                    answer = extract_answer_json(content_str)

                if not answer:
                    errors += 1
                    continue

                records.append({
                    "custom_id": custom_id,
                    "question": question,
                    "rep": rep,
                    "answer": answer,
                })
            except Exception as e:
                errors += 1

    print(f"    Linhas: {total_lines} | Sucesso: {len(records)} | Erros: {errors}")
    return records


# ---------------------------------------------------------------------------
# 3. Distribuição por LLM
# ---------------------------------------------------------------------------

def compute_llm_distribution(records: list, question: str, valid_options: list):
    """
    Para perguntas de escolha única (como implementado nos LLMs — cada
    persona responde 1 opção por pergunta por replicação).
    Retorna dist por opção e total de respostas.
    """
    q_records = [r for r in records if r["question"] == question]
    total = len(q_records)
    counts = defaultdict(int)
    for r in q_records:
        counts[r["answer"]] += 1

    dist = {}
    for opt in valid_options:
        dist[opt] = counts.get(opt, 0) / total if total > 0 else 0.0

    # Respostas fora do esperado
    ns_nr = sum(v for k, v in counts.items() if k not in valid_options and k in "mn")
    invalid = sum(v for k, v in counts.items() if k not in valid_options and k not in "mn")
    dist["_ns_nr"] = ns_nr / total if total > 0 else 0.0
    dist["_invalid"] = invalid / total if total > 0 else 0.0

    return dist, total, counts


# ---------------------------------------------------------------------------
# 4. JSD e Cramér's V
# ---------------------------------------------------------------------------

def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """JSD em base 2; 0 = idêntico, 1 = máxima divergência."""
    eps = 1e-10
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2))


def compute_cramers_v(p_llm: np.ndarray, p_real: np.ndarray, n: int = 2000):
    """
    Cramér's V comparando distribuição LLM vs. distribuição real.
    Monta tabela de contingência 2×K com contagens simuladas.
    """
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
# 5. Ajuste: LLM escolhe 1 opção por replicação
#    Real: pode ter múltiplas escolhas (P02, P03)
#    Para comparação justa, usamos proporção de "votos" por opção.
#    Real P02/P03: soma de menções / total de respondentes (multimarcação)
#    LLM  P02/P03: contagem de escolhas / total de respostas (1 por vez)
#    Ambas ficam como proporção relativa → normalizamos para somar 1 antes do JSD
# ---------------------------------------------------------------------------

def normalize(d: dict, keys: list) -> np.ndarray:
    v = np.array([d.get(k, 0.0) for k in keys], dtype=float)
    s = v.sum()
    if s == 0:
        return v
    return v / s


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------

def run_analysis():
    print("=" * 70)
    print("ANÁLISE: Percepção dos Brasileiros Acerca da Democracia")
    print("=" * 70)

    # --- Carrega SAV ---
    print("\n[1] Carregando dados reais (SAV)...")
    df_sav, meta_sav = load_sav(SAV_PATH)
    n_respondents = len(df_sav)
    print(f"    Respondentes reais: {n_respondents}")

    # --- Distribuições reais ---
    print("\n[2] Calculando distribuições reais...")

    # P02 (multimarcação: até 3 opções)
    real_p02_dist, _ = real_distribution_multiple(
        df_sav, SAV_P02_COLS, P02_CODE_TO_LETTER, QUESTION_OPTIONS["P02"], n_respondents
    )
    # P03 (multimarcação: até 6 opções)
    real_p03_dist, _ = real_distribution_multiple(
        df_sav, SAV_P03_COLS, P03_CODE_TO_LETTER, QUESTION_OPTIONS["P03"], n_respondents
    )
    # P04 (única)
    real_p04_dist, _ = real_distribution_single(
        df_sav, SAV_P04_COL, P04_CODE_TO_LETTER, QUESTION_OPTIONS["P04"]
    )

    real_distributions = {
        "P02": real_p02_dist,
        "P03": real_p03_dist,
        "P04": real_p04_dist,
    }

    for q, dist in real_distributions.items():
        opts = QUESTION_OPTIONS[q]
        total_prop = sum(dist.get(o, 0) for o in opts)
        print(f"\n  {q} (proporção de menções, soma={total_prop:.3f}):")
        for opt in opts:
            bar = "█" * int(dist.get(opt, 0) * 30)
            print(f"    {opt}: {dist.get(opt, 0):.3f}  {bar}")

    # --- Carrega LLMs ---
    print("\n[3] Carregando respostas dos LLMs...")
    llm_records = {}
    for model_name, path in MODELS.items():
        if not path.exists():
            print(f"\n  AVISO: {path.name} não encontrado!")
            continue
        print(f"\n  Modelo: {model_name} ({path.name})")
        records = parse_jsonl(path, model_name)
        llm_records[model_name] = records

    # --- Estatísticas LLM ---
    print("\n" + "=" * 70)
    print("[4] ESTATÍSTICAS POR MODELO")
    print("=" * 70)

    llm_distributions = {}

    for model_name, records in llm_records.items():
        llm_distributions[model_name] = {}
        print(f"\n{'─'*60}")
        print(f"  MODELO: {model_name.upper()}")
        print(f"{'─'*60}")

        for q in QUESTIONS:
            opts = QUESTION_OPTIONS[q]
            dist, total, counts = compute_llm_distribution(records, q, opts)
            llm_distributions[model_name][q] = dist

            print(f"\n  {q} (n={total}):")
            for opt in opts:
                bar = "█" * int(dist.get(opt, 0) * 40)
                print(f"    {opt}: {dist.get(opt, 0):.3f}  {bar}")
            if dist.get("_ns_nr", 0) > 0:
                print(f"    [NS/NR]:    {dist['_ns_nr']:.3f}")
            if dist.get("_invalid", 0) > 0:
                print(f"    [inválidas]:{dist['_invalid']:.3f}")

    # --- JSD e Cramér's V ---
    print("\n" + "=" * 70)
    print("[5] JSD E CRAMÉR'S V (LLM vs. Real)")
    print("=" * 70)

    results_table = []

    for q in QUESTIONS:
        opts = QUESTION_OPTIONS[q]
        real_dist = real_distributions[q]

        # Vetor real normalizado
        p_real_raw = np.array([real_dist.get(o, 0.0) for o in opts])
        p_real = p_real_raw / p_real_raw.sum() if p_real_raw.sum() > 0 else p_real_raw

        print(f"\n{'─'*60}")
        print(f"  PERGUNTA: {q}")
        print(f"  Real (normalizado): {dict(zip(opts, p_real.round(3)))}")
        print(f"{'─'*60}")

        for model_name, qdists in llm_distributions.items():
            if q not in qdists:
                continue
            llm_dist = qdists[q]

            p_llm_raw = np.array([llm_dist.get(o, 0.0) for o in opts])
            p_llm = p_llm_raw / p_llm_raw.sum() if p_llm_raw.sum() > 0 else p_llm_raw

            jsd_val = compute_jsd(p_llm, p_real)
            cv_val, p_val = compute_cramers_v(p_llm, p_real, n=10000)

            results_table.append({
                "Modelo": model_name,
                "Pergunta": q,
                "N_LLM": sum(1 for r in llm_records.get(model_name, []) if r["question"] == q),
                "JSD": round(jsd_val, 4),
                "CramersV": round(cv_val, 4) if not np.isnan(cv_val) else None,
                "p_valor": f"{p_val:.4e}" if not np.isnan(p_val) else None,
            })

            print(f"\n  [{model_name}]")
            print(f"    JSD        = {jsd_val:.4f}  (0=idêntico, 1=totalmente diferente)")
            print(f"    Cramér's V = {cv_val:.4f}  (0=sem assoc, 1=assoc perfeita)")
            print(f"    p-valor    = {p_val:.4e}")
            print(f"    LLM  dist: {dict(zip(opts, p_llm.round(3)))}")
            print(f"    Real dist: {dict(zip(opts, p_real.round(3)))}")

    # --- Tabela resumo ---
    print("\n" + "=" * 70)
    print("[6] TABELA COMPARATIVA DE DESEMPENHO")
    print("=" * 70)

    df_res = pd.DataFrame(results_table)
    if df_res.empty:
        print("  Sem dados para comparar.")
        return df_res

    print(df_res.to_string(index=False))

    # Ranking por JSD médio (menor = melhor)
    print("\n--- Ranking por JSD Médio (menor = melhor) ---")
    ranking_jsd = df_res.groupby("Modelo")["JSD"].mean().sort_values()
    for model, val in ranking_jsd.items():
        bar = "█" * int(val * 50)
        print(f"  {model:<18} {val:.4f}  {bar}")

    # Ranking por Cramér's V médio (maior = melhor alinhamento)
    print("\n--- Ranking por Cramér's V Médio (maior = melhor) ---")
    ranking_cv = df_res.groupby("Modelo")["CramersV"].mean().sort_values(ascending=False)
    for model, val in ranking_cv.items():
        bar = "█" * int(val * 50) if not np.isnan(val) else ""
        print(f"  {model:<18} {val:.4f}  {bar}")

    # Interpretação
    print("\n--- Interpretação ---")
    best_jsd = ranking_jsd.idxmin()
    best_cv  = ranking_cv.idxmax()
    print(f"  Menor JSD médio:      {best_jsd} ({ranking_jsd.min():.4f})")
    print(f"  Maior Cramér's V médio: {best_cv} ({ranking_cv.max():.4f})")

    # Salva CSV
    out_csv = RESULTS_DIR / "analise_comparativa_llms.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n  Resultados salvos em: {out_csv}")

    return df_res, real_distributions, llm_distributions


if __name__ == "__main__":
    run_analysis()
