def build_profile(row):
    profile = f"""Perfil:

Sexo: {row['SEXO']}
Escolaridade: {row['ESCOLARIDADE']}
Religião: {row['RELIGIAO']}
Faixa etária: {row['FX_ID']}
Raça: {row['RACA']}
Renda pessoal: {row['REND1']} salários mínimos
Renda familiar: {row['REND2']} salários mínimos
Região: {row['REGIAO']}
Condição do município: {row['COND']}
"""

    if row["P4"] != "Não sabe/ Não respondeu":
        profile += f"Interesse em participar da vida política: {row['P4']}\n"

    profile += f"Lembra voto para Deputado Estadual em 2022: {row['P1A']}\n"
    profile += f"Lembra voto para Deputado Federal em 2022: {row['P1B']}\n"
    profile += f"Lembra voto para Senador em 2022: {row['P1C']}\n"

    return profile