# Simulando a Opinião Pública Brasileira: Uma Análise Comparativa entre Modelos de Aprendizado Supervisionado e LLMs

[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=github&logoColor=white)](docs/Simulação_opinião_pública_sobre_a_percepção_dos_brasileiros_acerca_da_democracia.pdf)
[![Presentation](https://img.shields.io/badge/Presentation-PPTX-blue?style=flat&logo=github&logoColor=white)](docs/Simulação%20opinião%20pública%20sobre%20a%20percepçao%20dos%20brasileiros%20acerca%20da%20democracia.pptx)
[![Video](https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=youtube&logoColor=white)](https://youtu.be/k-AyQ2m4o3M)
[![Data](https://img.shields.io/badge/Data-CESOP-blue?style=flat)](https://www.cesop.unicamp.br/v3/portal/estudos/04832)

Este repositório contém o código e os dados da pesquisa que investiga a viabilidade do uso de modelos de aprendizado de máquina supervisionado e modelos de linguagem de grande porte (LLMs) para simular padrões de opinião pública no Brasil.

## Resumo (Abstract)

Este estudo compara o desempenho de um classificador **Random Forest** (estruturado via *Classifier Chain*) com o modelo **LLaMA 3** (via *Silicon Sampling*). A avaliação foi realizada em dois níveis: individual (acurácia) e distribucional (Divergência de Jensen-Shannon - JSD). Os resultados indicam que o modelo supervisionado superou o LLM em ambas as métricas, demonstrando que o treinamento em dados locais captura com maior fidelidade a heterogeneidade e as nuances do cenário sociodemográfico brasileiro.

---

## 1. Metodologia

### 1.1 Base de Dados

Os experimentos utilizaram os dados brutos da pesquisa **CESOP/IPEC/04832** ("Percepção dos Brasileiros acerca da Democracia"), realizada em setembro de 2023 com uma amostra de 2.000 entrevistados.

### 1.2 Atributos e Alvos

- **Atributos (Features):** Variáveis sociodemográficas (Sexo, Idade, Escolaridade, Raça, Região, Renda, Religião) e comportamentais (Interesse político, lembrança de voto).
- **Alvos (Targets):**
  - `P2_1`: Prioridades de políticas públicas segundo o cidadão.
  - `P3_1`: Opções preferenciais para o combate às *fake news*.

### 1.3 Modelagem

- **Aprendizado Supervisionado (ML):** Uso de `RandomForestClassifier` com a técnica de `ClassifierChain` para tratar a natureza multirrótulo do problema. Otimização via `GridSearchCV` e validação cruzada com 5 folds.
- **Modelos de Linguagem (LLM):** `LLaMA 3` (8B) executado localmente via **Ollama**. Foi empregada uma estratégia de *zero-shot prompting* baseada em perfis demográficos sintéticos.

---

## 2. Resultados Principais

Os resultados demonstram uma superioridade estatística do modelo supervisionado na preservação da integridade das distribuições populacionais originais.

| Métrica | Modelo LLM (P2_1) | Modelo ML (P2_1) | Modelo LLM (P3_1) | Modelo ML (P3_1) |
| :--- | :---: | :---: | :---: | :---: |
| **Acurácia (↑)** | 10.5% | **17.0%** | 24.0% | **28.0%** |
| **JSD (↓)** | 0.2674 | **0.0365** | 0.1044 | **0.0268** |

*Nota: (↑) quanto maior melhor; (↓) quanto menor melhor.*

---

## 3. Estrutura do Repositório

- `docs/`: Artigo completo, apresentação e documentação do dataset original.
- `ml_simulation/`: Notebooks contendo o pipeline de pré-processamento, treinamento e avaliação do Random Forest.
- `llm_simulation/`: Scripts para execução das simulações e análise dos resultados via LLM.
- `llm_simulation/outputs/`: Tabelas de métricas e gráficos comparativos gerados.

---

## 4. Como Reproduzir

### Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com/) instalado (para execução do LLaMA 3)

### Instalação

```bash
# Instalar dependências gerais
pip install pandas scikit-learn scipy matplotlib seaborn

# Instalar dependências específicas do módulo LLM
cd llm_simulation
pip install -r requirements.txt
```

### Execução

1. **Modelagem Supervisionada (ML):** Execute os notebooks em `ml_simulation/`.
   - O notebook `simulacao_opiniao_publica_v2.ipynb` contém a configuração exata utilizada para a redação do artigo final.
   - A versão `v3` inclui experimentações adicionais com novos hiperparâmetros.

2. **Simulação via LLM:** Com o servidor Ollama ativo, execute:

   ```bash
   python llm_simulation/src/get_data.py      # Geração de respostas sintéticas
   python llm_simulation/src/compare_data.py  # Cálculo de métricas comparativas
   python llm_simulation/src/display_data.py  # Geração das visualizações
   ```

---

## 5. Dicionário de Variáveis

Abaixo estão as principais variáveis extraídas do dataset **CESOP 04832**, classificadas conforme a metodologia do artigo:

### 5.1 Atributos Preditivos (Features)

| Código | Descrição da Variável (Texto Original) | Tipo |
| :--- | :--- | :--- |
| `FX_ID` | Faixas de idade do respondente | Ordinal |
| `P4` | Grau de vontade de participar da vida política na sua cidade | Ordinal |
| `REND1` | Renda pessoal mensal (em salários mínimos) | Ordinal |
| `REND2` | Renda familiar mensal (em salários mínimos) | Ordinal |
| `SEXO` | Sexo do respondente (Masculino / Feminino) | Nominal |
| `ESCOLARIDADE` | Nível de instrução concluído | Nominal |
| `RACA` | Autodeclaração de raça ou cor | Nominal |
| `RELIGIAO` | Religião ou crença declarada | Nominal |
| `REGIAO` | Região geográfica do Brasil (Norte, NE, SE, Sul, CO) | Nominal |
| `COND` | Condição do município (Capital, Periferia ou Interior) | Nominal |

### 5.2 Variáveis Alvo (Targets)

| Código | Pergunta do Questionário | Objetivo |
| :--- | :--- | :--- |
| **`P2_1`** | "Qual destas propostas você acha que deveria ser prioridade de um(a) político(a)?" | Prioridade Política |
| **`P3_1`** | "Quais dessas opções você acredita que poderiam contribuir no combate às fake news?" | Combate à Desinformação |

---

## 6. Limitações e Considerações Éticas

- **Viés de Treinamento:** Os resultados indicam que LLMs tendem a "homogeneizar" as respostas baseando-se em tendências globais de treinamento, enquanto modelos supervisionados locais captam melhor as disparidades regionais brasileiras.
- **Amostragem:** A simulação foi realizada em um subconjunto de 10% da base original para fins de prova de conceito.
- **Privacidade:** Os dados utilizados são anonimizados e provenientes de repositórios públicos para fins acadêmicos.

---

## Equipe e Orientação

- **Pesquisadores:**
  - [Lucas Nascimento Aguiar](mailto:lucas.naguiar@outlook.com)
  - [Matheus dos Santos Moreira](mailto:matheusmoreira2004@live.com)
- **Orientador:**
  - [Prof. Dr. Rogério de Oliveira](http://lattes.cnpq.br/3067732992972770) (Universidade Presbiteriana Mackenzie)

---

## Citação

```bibtex
@article{aguiar2026simulacao,
  title={Simulando a Opinião Pública Brasileira: Uma Análise Comparativa entre Modelos de Aprendizado Supervisionado e LLMs},
  author={Aguiar, Lucas Nascimento and Moreira, Matheus dos Santos and Oliveira, Rogério},
  institution={Universidade Presbiteriana Mackenzie},
  year={2026},
  url={https://github.com/olucasaguiar/projeto-simulacao-opiniao-publica}
}
```
