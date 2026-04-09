import pandas as pd
import requests
import pyreadstat

# -------------------------------
# CONFIG
# -------------------------------
TEST_PATH = "llm_simulation/data/df_test.csv"
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
- Escolaridade: {row['ESCOLARIDADE']}
- Religião: {row['RELIGIAO']}
- Faixa etária: {row['FX_ID']}
- Raça: {row['RACA']}
- Renda pessoal (em salários mínimos): {row['REND1']}
- Renda familiar (em salários mínimos): {row['REND2']}
- Região do país: {row['REGIAO']}
- Condição do municipio: {row['COND']}
- Lembra em quem votou para Deputado Estadual nas eleições gerais de 2022: {row['P1A']}
- Lembra em quem votou para Deputado Federal nas eleições gerais de 2022: {row['P1B']}
- Lembra em quem votou para Senador nas eleições gerais de 2022? {row['P1C']}
- Nível de interesse em participar da vida política? {row['P4']}

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
def extract_questions():
    return {'P2_1': {'text': 'P.02) Qual dessas propostas você acha que deveria ser prioridade de um(a) político(a)?', 'options': ['Reduzir as desigualdades sociais','Combater o preconceito (racismo, homofobia, diferença de classe social, etc.)','Aumentar os impostos de grandes fortunas (ou dos mais ricos)','Incentivar a geração de empregos','Combater as mudanças climáticas/desmatamento','Ampliar o uso de energias renováveis','Preservar os valores ligados à família','Defender a igualdade entre homens e mulheres', 'Melhorar a qualidade da Saúde', 'Melhorar a qualidade da Educação', 'Reduzir a violência', 'Ampliar os espaços de participação política da população', 'Não sabe', 'Não respondeu']}, 'P3_1': {'text': 'P.03) Quais dessas opções você acredita que poderiam contribuir no combate à divulgação de fake news?', 'options': ['Ampliar a regulamentação, as regras a serem cumpridas pelas plataformas digitais (empresas de tecnologia como Facebook, Youtube, WhatsApp, Twitter/X, etc.)','Responsabilizar e punir as empresas de tecnologia e de comunicação que não removerem postagens com notícias ou conteúdos falsos','Ampliar a regulamentação, as regras a serem cumpridas pelos usuários que divulgam ou compartilham fake news, criadas por eles próprios ou por terceiros','Responsabilizar e punir os usuários que divulgam ou compartilham postagens com notícias ou conteúdos falsos','Ampliar a regulamentação, as regras a serem cumpridas por políticos/candidatos que divulgam ou compartilham fake news, criadas por eles próprios ou por terceiros','Responsabilizar, punir ou cassar políticos/candidatos que divulgam ou compartilham postagens com notícias ou conteúdos falsos', 'Não sabe','Não respondeu']}}


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
    df = pd.read_csv(TEST_PATH)

    # extract questions
    questions = extract_questions()
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