from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from importlib import import_module
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--style",
    default="natural",
    choices=["natural", "key_value", "markdown", "json_prompt"],
)
parser.add_argument(
    "--model",
    default="llama3",
    choices=["llama3", "gemma3", "qwen3"],
)

args = parser.parse_args()

PROMPT_STYLE = args.style
MODEL = args.model

TEST_PATH = "llm_simulation/data/df_test.csv"

OUTPUT_PATH = f"outputs/{MODEL}_{PROMPT_STYLE}.csv"

prompt_builder = import_module(f"prompts.{PROMPT_STYLE}")
llm_builder = import_module(f"llm.{MODEL}")

def call_llm(prompt):
    return llm_builder.call_llm(prompt)

# -------------------------------
# PROMPT
# -------------------------------
def build_prompt(row, question_text, options):
    profile = prompt_builder.build_profile(row)
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