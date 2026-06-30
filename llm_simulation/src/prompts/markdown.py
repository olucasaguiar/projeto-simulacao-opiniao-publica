def build_profile(row):
    profile = f"""| Campo | Valor |
|-------|-------|
| Sexo | {row['SEXO']} |
| Escolaridade | {row['ESCOLARIDADE']} |
| Religião | {row['RELIGIAO']} |
| Faixa etária | {row['FX_ID']} |
| Raça | {row['RACA']} |
| Renda pessoal | {row['REND1']} salários mínimos |
| Renda familiar | {row['REND2']} salários mínimos |
| Região | {row['REGIAO']} |
| Condição do município | {row['COND']} |
"""

    if row["P4"] != "Não sabe/ Não respondeu":
        profile += f"| Interesse em política | {row['P4']} |\n"

    profile += f"| Lembra voto Deputado Estadual | {row['P1A']} |\n"
    profile += f"| Lembra voto Deputado Federal | {row['P1B']} |\n"
    profile += f"| Lembra voto Senador | {row['P1C']} |\n"

    return profile