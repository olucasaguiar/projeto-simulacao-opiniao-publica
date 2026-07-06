from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from importlib import import_module
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--feature_style",
    default="full",
    choices=["full", "demographics_only", "no_religion", "no_income", "no_memory", "no_interest", "politics_only"],
)
parser.add_argument(
    "--model",
    default="llama3",
    choices=["llama3", "gemma3", "qwen3"],
)

DEMOGRAPHICS = {
    "SEXO",
    "ESCOLARIDADE",
    "RELIGIAO",
    "FX_ID",
    "RACA",
    "REND1",
    "REND2",
    "REGIAO",
    "COND",
}

POLITICAL_INTEREST = {
    "P4",
}

VOTING_MEMORY = {
    "P1A",
    "P1B",
    "P1C",
}

FULL = DEMOGRAPHICS | POLITICAL_INTEREST | VOTING_MEMORY
DEMOGRAPHICS_ONLY = DEMOGRAPHICS
NO_RELIGION = FULL - {"RELIGIAO"}
NO_INCOME = FULL - {"REND1", "REND2"}
NO_MEMORY = FULL - VOTING_MEMORY
NO_INTEREST = FULL - POLITICAL_INTEREST
POLITICS_ONLY = POLITICAL_INTEREST | VOTING_MEMORY

FEATURES_BY_FEATURE_STYLE = {
    "full": FULL,
    "demographics_only": DEMOGRAPHICS_ONLY,
    "no_religion": NO_RELIGION,
    "no_income": NO_INCOME,
    "no_memory": NO_MEMORY,
    "no_interest": NO_INTEREST,
    "politics_only": POLITICS_ONLY
}

args = parser.parse_args()

PROMPT_FEATURE_STYLE = args.feature_style
MODEL = args.model

TEST_PATH = "llm_simulation/data/df_test.csv"

OUTPUT_PATH = f"llm_simulation/outputs/{MODEL}_{PROMPT_FEATURE_STYLE}.csv"

llm_builder = import_module(f"llm.{MODEL}")

def call_llm(prompt):
    return llm_builder.call_llm(prompt)

def build_profile(row):
    features = FEATURES_BY_FEATURE_STYLE[PROMPT_FEATURE_STYLE]
    profile = "Você é uma pessoa"

    if "SEXO" in features:
        profile += f" do sexo {row['SEXO']}"

    if "ESCOLARIDADE" in features:
        profile += f", tem escolaridade {row['ESCOLARIDADE']}"

    if "RELIGIAO" in features:
        profile += f", religião {row['RELIGIAO']}"

    if "FX_ID" in features:
        profile += f", é da faixa etária {row['FX_ID']}"

    if "RACA" in features:
        profile += f", é da raça {row['RACA']}"

    if "REND1" in features:
        profile += f", renda pessoal de {row['REND1']} salários mínimos"

    if "REND2" in features:
        profile += f", renda familiar de {row['REND2']} salários mínimos"

    if "REGIAO" in features:
        profile += f", mora na região {row['REGIAO']} do país"

    if "COND" in features:
        profile += f", em um município com condição {row['COND']}"

    profile += ".\n"

    if "P4" in features and row["P4"] != "Não sabe/ Não respondeu":
        profile += f"Você tem {row['P4']} em participar da vida política.\n"

    if "P1A" in features:
        if row["P1A"] == "Sim":
            profile += "Você se lembra em quem votou para Deputado Estadual nas eleições gerais de 2022.\n"
        else:
            profile += "Você não lembra em quem votou para Deputado Estadual nas eleições gerais de 2022.\n"

    if "P1B" in features:
        if row["P1B"] == "Sim":
            profile += "Você se lembra em quem votou para Deputado Federal nas eleições gerais de 2022.\n"
        else:
            profile += "Você não lembra em quem votou para Deputado Federal nas eleições gerais de 2022.\n"

    if "P1C" in features:
        if row["P1C"] == "Sim":
            profile += "Você se lembra em quem votou para Senador nas eleições gerais de 2022.\n"
        else:
            profile += "Você não lembra em quem votou para Senador nas eleições gerais de 2022.\n"

    return profile

# -------------------------------
# PROMPT
# -------------------------------
def build_prompt(row, question_text, options):
    profile = build_profile(row)
    options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    #respondente brasileiro de uma pesquisa de opinião.
    prompt = f"""
{profile}
Baseie sua resposta nas características do perfil e no comportamento típico de indivíduos semelhantes.

Pergunta:
"{question_text}"

Opções:
{options_str}

Responda com APENAS uma das opções exatamente como escrita.
Não explique sua resposta.
"""
    return prompt

# -------------------------------
# QUESTIONS EXTRACTION
# -------------------------------
def extract_questions():
    return {'P2_1': {'text': 'P.02) Qual dessas propostas você acha que deveria ser prioridade de um(a) político(a)?', 'options': ['Reduzir as desigualdades sociais','Combater o preconceito (racismo, homofobia, diferença de classe social, etc.)','Aumentar os impostos de grandes fortunas (ou dos mais ricos)','Incentivar a geração de empregos','Ampliar o uso de energias renováveis','Preservar os valores ligados à família','Defender a igualdade entre homens e mulheres','Melhorar a qualidade da saúde','Melhorar a qualidade da educação','Reduzir a violência','Ampliar os espaços de participação política da população','Não sabe/ Não respondeu']}, 'P3_1': {'text': 'P.03) Quais dessas opções você acredita que poderiam contribuir no combate à divulgação de fake news?', 'options': ['Ampliar a regulamentação, as regras a serem cumpridas pelas plataformas digitais (Facebook, Youtube, WhatsApp, etc.)','Responsabilizar e punir as empresas de tecnologia/comunicação que não removerem postagens com conteúdos falsos','Ampliar a regulamentação para usuários que divulgam fake news, criadas por eles próprios ou por terceiros','Responsabilizar e punir os usuários que divulgam ou compartilham postagens com notícias ou conteúdos falsos','Ampliar a regulamentação para políticos que divulgam fake news, criadas por eles próprios ou por terceiros','Responsabilizar, punir ou caçar políticos que divulgam ou compartilham postagens com notícias ou conteúdos falsos', 'Não sabe/ Não respondeu']}}

# -------------------------------
# OPTIONAL: CLEAN ANSWER
# -------------------------------
def clean_answer(answer, options):
    for opt in options:
        if opt.lower() in answer.lower():
            return opt
    return answer  # fallback

def process_row(i, row, questions):
    local_results = []
    i+=1
    for q_col, q_data in questions.items():
        if pd.isna(row[q_col]):
            continue

        prompt = build_prompt(row, q_data["text"], q_data["options"])
        raw_answer = call_llm(prompt)
        answer = clean_answer(raw_answer, q_data["options"])

        local_results.append({
            "index": i,
            "question": q_col,
            "llm_answer": answer,
            "true_answer": row[q_col]
        })
        print(f"{i} - {q_col}: {answer}")

    return local_results

# -------------------------------
# MAIN
# -------------------------------
def main():
    # load data
    df = pd.read_csv(TEST_PATH)

    # extract questions
    questions = extract_questions()

    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_row, i, row, questions)
            for i, row in df.iterrows()
        ]

        for future in as_completed(futures):
            results.extend(future.result())

    # save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()