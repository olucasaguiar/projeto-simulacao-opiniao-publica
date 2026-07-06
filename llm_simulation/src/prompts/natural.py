def build_profile(row, features):
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