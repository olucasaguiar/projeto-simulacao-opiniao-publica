def build_profile(row):
    profile = f"""
Você é uma pessoa do sexo {row['SEXO']}, tem escolaridade {row['ESCOLARIDADE']}, religião {row['RELIGIAO']}, é da faixa etária {row['FX_ID']}, é da raça {row['RACA']}, renda pessoal de {row['REND1']} salários mínimos, renda familiar de {row['REND2']} salários mínimos, mora na região {row['REGIAO']} do país, em um município com condição {row['COND']}. 
"""
    if row['P4'] != "Não sabe/ Não respondeu":
        profile = profile +f"""Você tem {row['P4']} em participar da vida política.
"""

    if row['P1A'] == "Sim":
        profile = profile +f"""Você se lembra em quem votou para Deputado Estadual nas eleições gerais de 2022.
"""
    else:
        profile = profile +f"""Você não lembra em quem votou para Deputado Estadual nas eleições gerais de 2022.
"""
    if row['P1B'] == "Sim":
        profile = profile +f"""Você se lembra em quem votou para Deputado Federal nas eleições gerais de 2022.
"""
    else:
        profile = profile +f"""Você não lembra em quem votou para Deputado Federal nas eleições gerais de 2022.
"""
    if row['P1C'] == "Sim":
        profile = profile +f"""Você se lembra em quem votou para Senador nas eleições gerais de 2022.
"""
    else:
        profile = profile +f"""Você não lembra em quem votou para Senador nas eleições gerais de 2022.
"""
    return profile