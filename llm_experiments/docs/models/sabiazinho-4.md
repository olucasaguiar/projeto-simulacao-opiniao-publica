# Modelo: Sabiazinho-4

## 1. Informações Técnicas Gerais

- **Desenvolvedor:** Maritaca AI
- **Arquitetura e Tamanho:** Versão menor, destilada e otimizada baseada na família Sabiá-4 (tamanho exato não divulgado).
- **Janela de Contexto:** Suporta até 128.000 tokens de contexto, mantendo paridade com seu "irmão maior" em capacidade de leitura de documentos extensos.

## 2. Foco Estratégico e Treinamento

Conforme o Technical Report (arXiv:2603.10213), o Sabiazinho-4 compartilha o mesmo rigoroso pipeline de 4 estágios de seu "irmão maior", o Sabiá-4:

1. **Continued pre-training** em corpora do Brasil.
2. **Context Extension** para os mesmos 128K tokens.
3. **SFT** para alinhamento instrucional.
4. **Preference alignment**.

Foi desenhado para atuar como o modelo de **maior custo-benefício** do portfólio da Maritaca AI. Seu objetivo principal é fornecer baixa latência e alta viabilidade econômica para tarefas em massa (como simulação de pesquisas demográficas), ocupando de forma agressiva a região superior-esquerda (upper-left) no gráfico de preço x acurácia perante competidores mundiais.

## 3. Desempenho

O artigo confirma que, apesar do tamanho e custo reduzidos, o Sabiazinho-4 herda notável proficiência em domínios testados (OAB-Bench, Multi-IF, MRCR). De forma surpreendente, mostra aprimoramento contundente frente às antigas gerações (como Sabiá-3.1) para resolução de diálogos longos e tarefas de completude agêntica (ex: Ticket-Bench, Pix-Bench), o que justifica integralmente seu uso para automatizar em larga escala e com baixo custo o preenchimento de pesquisas no simulador.

## 4. Plataforma de Execução nos Experimentos

Nos experimentos, a interação ocorre através do endpoint padrão da API REST da Maritaca AI (via `maritaca_adapter.py`). A execução em larga escala foi orquestrada pela configuração em lote (`generate_openai_batch.py` serviu como estudo para precificar comparativamente essa mesma extração no ecossistema concorrente).
