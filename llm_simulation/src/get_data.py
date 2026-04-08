import pandas as pd
import requests
import pyreadstat
import time

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "llm_simulation/data/dataset_sampled_labeled.csv"
SAV_PATH = "docs/04832 PERCEPÇÃO DOS BRASILEIROS ACERCA DA DEMOCRACIA/04832.SAV"
OUTPUT_PATH = "llm_simulation/outputs/llm_results.csv"
MODEL = "llama3"

# -------------------------------
# LLM CALL
# -------------------------------
def call_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"].strip()


# -------------------------------
# PROMPT
# -------------------------------
def build_prompt(row, question_text, options):
    options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

    return f"""
Você está simulando um respondente brasileiro de uma pesquisa de opinião.

Perfil da pessoa:
- Sexo: {row['SEXO']}
- Idade: {row['IDADE']}
- Escolaridade: {row['ESCOLARIDADE']}
- Renda: {row['REND1']}
- Religião: {row['RELIGIAO']}

Baseie sua resposta nas características do perfil e no comportamento típico de indivíduos semelhantes.

Pergunta:
"{question_text}"

Opções:
{options_str}

Responda com APENAS uma das opções exatamente como escrita.
Não explique sua resposta.
"""


# -------------------------------
# QUESTIONS EXTRACTION
# -------------------------------
def extract_questions(meta):
    questions = {}

    for col, question_text in meta.column_names_to_labels.items():
        if col not in meta.variable_value_labels:
            continue

        # keep only survey questions
        if not col.startswith("P") or col == "PORTE":
            continue

        options = list(meta.variable_value_labels[col].values())

        questions[col] = {
            "text": question_text,
            "options": options
        }

    return questions


# -------------------------------
# OPTIONAL: CLEAN ANSWER
# -------------------------------
def clean_answer(answer, options):
    for opt in options:
        if opt.lower() in answer.lower():
            return opt
    return answer  # fallback


# -------------------------------
# MAIN
# -------------------------------
def main():
    # load data
    df = pd.read_csv(DATA_PATH)

    # load metadata
    _, meta = pyreadstat.read_sav(SAV_PATH, apply_value_formats=True)

    # extract questions
    questions = extract_questions(meta)

    results = []

    for i, row in df.iterrows():
        i+=1
        for q_col, q_data in questions.items():
            if str(row[q_col]) == "nan":
                continue
            prompt = build_prompt(row, q_data["text"], q_data["options"])

            raw_answer = call_llm(prompt)
            answer = clean_answer(raw_answer, q_data["options"])

            results.append({
                "index": i,
                "question": q_col,
                "llm_answer": answer,
                "true_answer": row[q_col]
            })

            print(f"{i} - {q_col}: {answer}")


    # save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()