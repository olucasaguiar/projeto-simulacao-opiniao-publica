# Modelo: LLaMA 3.2

## 1. Informações Técnicas Gerais e Arquitetura

- **Desenvolvedor:** Meta
- **Tamanhos Avaliados:** Foram avaliadas as versões de 1 Bilhão (1B) e 3 Bilhões (3B) de parâmetros (`llama-3.2-1b` e `llama-3.2-3b`). A versão de 3B possui exatamente 3.21 bilhões de parâmetros.
- **Arquitetura:** Dense Transformer decoder (auto-regressivo). A versão 3B possui 24 camadas (layers), dimensão oculta de 3.072 e ativações SwiGLU. Emprega extensivamente o *Grouped-Query Attention (GQA)* para máxima eficiência de memória e velocidade de inferência.
- **Janela de Contexto:** 128.000 tokens de contexto.

## 2. Treinamento e Foco Estratégico

O LLaMA 3.2 de 1B e 3B é um *Small Language Model (SLM)* otimizado especificamente para rodar localmente e "no limite" (*edge devices*, *mobile* com processadores ARM, MediaTek e Qualcomm, além de computadores de uso comum). É fundamental quando se discute privacidade total de dados em simulações locais.

Seu pipeline de treinamento, revelado pela Meta, detalha:

- **Pré-treinamento:** Mistura de dados online públicos com corte de conhecimento em dezembro de 2023. Utiliza pesadamente a técnica de *Knowledge Distillation* (Destilação de Conhecimento), em que as saídas (logits) dos modelos maiores Llama 3.1 (8B e 70B) são incorporadas no treinamento como metas por token (token-level targets), injetando profundo "raciocínio" nesta versão diminuta.
- **Alinhamento (Post-training):** A versão *Instruct* passou por exaustivas rodadas de *Supervised Fine-Tuning (SFT)*, seguidas de *Rejection Sampling (RS)* e *Direct Preference Optimization (DPO)* para alinhar precisão nas instruções e segurança humana.

## 3. Desempenho e Benchmarks

De acordo com os benchmarks divulgados pela Meta, as versões leves do Llama 3.2 possuem uma relação tamanho-performance excepcional para tarefas como resumo, reescrita e seguimento restrito de instruções (*instruction following*). Todavia, diferentemente dos modelos Sabiá, o treinamento foca em inglês e em diversidade cultural ampla de poucas línguas predominantes, tendendo a carregar vieses e padronizações comportamentais globais/ocidentais, o que pode homogeneizar as repostas em cenários que requerem sensibilidade local apurada.

## 4. Plataforma de Execução nos Experimentos

Nos experimentos do projeto (diretório `@llm_experiments`), esses modelos são executados **localmente**, sem depender de ferramentas de empacotamento prontas como o Ollama (o que difere do primeiro experimento `llm_simulation`).

- **Acesso Base:** Via Hugging Face (`meta-llama/Llama-3.2-1B-Instruct`).
- **Adapter:** Operacionalizados diretamente via a biblioteca Transformers no Pytorch (via `llama_adapter.py`).
- **Script Autônomo:** O fluxo base local está isolado em `run_llama_simulation.py`, que baixa, mapeia nos tensores adequados e executa simulações sob as mesmas requisições formadas, suportando, pelo `config.yaml`, quantização caso a memória da GPU não seja suficiente, ou `device_map="auto"` para distribuição em aceleradores disponíveis.
