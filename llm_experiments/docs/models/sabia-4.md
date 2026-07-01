# Modelo: Sabiá-4

## 1. Informações Técnicas Gerais

- **Desenvolvedor:** Maritaca AI
- **Arquitetura e Tamanho:** Baseado na arquitetura Transformer (número exato de parâmetros não é publicamente divulgado devido à política de modelo fechado da Maritaca).
- **Janela de Contexto:** Suporta até 128.000 tokens de contexto, sendo apto a processar longos históricos de diálogos, documentos legais e instruções complexas.

## 2. Foco e Treinamento (Pipeline de 4 Estágios)

Conforme detalhado no Technical Report (arXiv:2603.10213), o modelo é focado em alta proficiência no português do Brasil, desenvolvido sob um pipeline rigoroso de quatro etapas:

1. **Continued Pre-training:** Utilizando corpora gerais da língua portuguesa e corpora jurídicos brasileiros.
2. **Long-context Extension:** Extensão da capacidade de leitura para atingir o limite de 128.000 tokens.
3. **Supervised Fine-Tuning (SFT):** Treinamento em dados de instrução divididos em categorias como conversação (chat), código, tarefas jurídicas e uso de ferramentas (function calling/agentic tasks).
4. **Preference Alignment:** Alinhamento de preferências humanas para melhorar o tom das respostas, seguir restrições de formato com precisão e entender nuances.

## 3. Desempenho e Benchmarks

O relatório técnico submeteu os modelos da geração 4 a 6 rigorosas categorias de *benchmarks*:

- **Capacidades Conversacionais:** Medidas no BRACEval (150 amostras multi-turn).
- **Legislação Brasileira:** OAB-Bench (estilo advogado), Magis-Bench (estilo juiz) e Brazilian Federal Laws.
- **Contexto Longo:** MRCR (*Multi-Round Co-reference Resolution*), mais demandante que o tradicional "Needle in a Haystack".
- **Instruction Following:** Avaliado pelo Multi-IF (instruções acumulativas).
- **Exames Padronizados:** ENEM, CFC, Revalida, CPNU.
- **Capacidades Agênticas:** Uso de ferramentas, avaliação de *pass rate* nos testes Ticket-Bench, Pix-Bench, MARCA e CLIMB.

**Conclusões de Qualidade:** O Sabiá-4 atinge uma posição bastante superior na relação custo-benefício (upper-left region do gráfico preço/acurácia), mostrando substancial avanço, em relação ao Sabiá-3, especialmente na elaboração de peças jurídicas, resolução de diálogos multi-turn e sucesso em completar tarefas agênticas.

## 4. Plataforma de Execução nos Experimentos

No repositório `@llm_experiments`, o Sabiá-4 é executado via chamadas HTTP REST para a API oficial da Maritaca AI. A orquestração da comunicação ocorre de forma nativa e isolada na classe provedora desenvolvida em `maritaca_adapter.py`.
