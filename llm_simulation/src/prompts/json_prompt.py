import json


def build_profile(row):
    profile = {
        "sexo": row["SEXO"],
        "escolaridade": row["ESCOLARIDADE"],
        "religiao": row["RELIGIAO"],
        "faixa_etaria": row["FX_ID"],
        "raca": row["RACA"],
        "renda_pessoal": row["REND1"],
        "renda_familiar": row["REND2"],
        "regiao": row["REGIAO"],
        "condicao_municipio": row["COND"],
        "interesse_politica": row["P4"],
        "lembra_dep_estadual": row["P1A"],
        "lembra_dep_federal": row["P1B"],
        "lembra_senador": row["P1C"],
    }

    return json.dumps(profile, ensure_ascii=False, indent=2)