# Projeto: Simulação de Opinião Pública

Este projeto consiste no desenvolvimento e comparação de modelos de Linguagem de Grande Escala (LLM) e Aprendizado Supervisionado para simular respostas de questionários de opinião pública.

## Introdução

O projeto utiliza dados do **CESOP** (Centro de Estudos de Opinião Pública) para avaliar a capacidade de modelos computacionais em prever e reproduzir distribuições de opinião pública brasileira.

## Grupo

1. Lucas Nascimento Aguiar
2. Matheus dos Santos Moreira

## Escopo e Objetivos

Baseado no artigo *"Simulating Public Opinion: Comparing Distributional and Individual-Level Predictions from LLMs and Random Forests"*, o trabalho foca em:

- **Simulação:** Executar ao menos 200 simulações (aprox. 10% da base original).
- **Comparação:** Avaliar acurácia, distribuição de respostas e explicabilidade (importância das variáveis).
- **Abrangência:** Incluir o máximo possível de características sociodemográficas e perguntas do questionário.

## Artigo

O artigo final produzido por este trabalho pode ser acessado aqui:

- [**Simulação opinião pública sobre a percepção dos brasileiros acerca da democracia**](docs/simulacao_opiniao_publica_sobre_a_percepcao_dos_brasileiros_acerca_da_democracia.pdf)

## Estrutura do Projeto

- `docs/`: Artigos de referência e documentação do dataset original.
- `ml_simulation/`: Notebooks Jupyter com os modelos de aprendizado supervisionado (Machine Learning).
- `llm_simulation/`: Implementação, dados e resultados da simulação via LLM.

## Ferramentas e Requisitos

- Uso exclusivo de modelos e recursos **open-source**.
- Implementação em **Python** e **Jupyter Notebook**.

## Referências

- [Simulating Public Opinion: Comparing Distributional and Individual-Level Predictions from LLMs and Random Forests](docs/simulating_public_opinion-comparing_distributional_and_individual_level_predictions_from_llms_and_random_forests.pdf)
