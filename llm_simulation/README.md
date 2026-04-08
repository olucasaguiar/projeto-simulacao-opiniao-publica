# Simulação de Opinião Pública com LLM

Este projeto realiza a simulação de respostas de um questionário utilizando um modelo de linguagem (LLM) local, com base em dados reais de uma pesquisa do CESOP.

---

## 📦 Requisitos

* Python 3.10+
* Ollama instalado (https://ollama.com/)
* Ambiente virtual (recomendado)

---

## ⚙️ Setup

### Usando Makefile (recomendado)

```bash
make setup
```

Isso irá:

* criar o ambiente virtual
* instalar todas as dependências

---

## 🤖 Rodar o modelo LLM localmente

Antes de executar a simulação, inicie o modelo:

```bash
ollama run llama3
```

Na primeira execução, o modelo será baixado automaticamente.
Deixe este processo rodando em outro terminal.

---

## ▶️ Executar simulação

```bash
make run
```

---

## ⚙️ Execução manual (opcional)

Caso não utilize o Makefile:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python llm_simulation.py
```

---

## 📁 Estrutura esperada

* `data/` → dataset utilizado
* `llm_simulation.py` → script principal
* `outputs/` → resultados gerados pelo modelo

---

## 🧠 Observação

O modelo LLM é utilizado para simular respostas de indivíduos com base em características demográficas, seguindo a abordagem de *silicon sampling*.

Todos os experimentos utilizam modelos abertos e execução local para garantir reprodutibilidade.
