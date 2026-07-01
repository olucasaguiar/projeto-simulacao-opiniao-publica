# Comparativo entre os Modelos

Este documento analisa as discrepâncias, equivalências e *trade-offs* entre a família **Sabiá-4/Sabiazinho-4** (Maritaca AI), a família **LLaMA 3.2** (Meta) e o **GPT-5-mini** (OpenAI) dentro do escopo de simulação de opinião pública conduzida nos experimentos do repositório.

## 1. Localização e Sensibilidade Sociodemográfica

- **Sabiá-4 e Sabiazinho-4:**
  A maior vantagem destes modelos não é apenas seu suporte formal ao português, mas o foco massivo no contexto brasileiro (comprovado pelo alto desempenho no OAB-Bench e exames nacionais). Sendo treinados desde a fundação com dados locais e refinados via *Preference Alignment* focado nas nuances brasileiras, possuem alto índice de aderência comportamental. Ao assumirem *personas* configuradas pelo IBGE, calibram naturalmente o viés esperado localmente.
- **LLaMA 3.2 (1B/3B):**
  Apresenta forte capacidade fundamental, impulsionada por técnicas avançadas como *Knowledge Distillation* dos modelos maiores (8B/70B). Entretanto, o modelo padece do fenômeno da "homogeneização". Devido ao forte treinamento alinhado com o contexto ocidental genérico e exaustivas rodadas de DPO (*Direct Preference Optimization*) e RS (*Rejection Sampling*) voltadas ao "safety alignment" neutro, o modelo tende a desconsiderar particularidades regionais para opinar em favor do que globalmente é tido como seguro e moralmente aceito, prejudicando a variância simulada.
- **GPT-5-mini:**
  Como modelo de fronteira global, possui excelente raciocínio geral e fidelidade extrema na aderência instrucional (instruction following). Todavia, sofre severamente com a homogeneização de opiniões sob personas específicas. Por conta do extenso alinhamento via RLHF/segurança global da OpenAI, ele tende a adotar respostas muito cautelosas, neutras ou politicamente corretas, "achatando" a polarização e as opiniões sociodemográficas reais que seriam esperadas da população brasileira, servindo como baseline de neutralidade institucional.

## 2. Modelos de Execução e Escalabilidade

- **Sabiá-4 e Sabiazinho-4 (API-based):**
  - **Acesso:** Computação na nuvem gerida via API REST (`maritaca_adapter.py`).
  - **Vantagem:** Evita necessidade de providenciar infraestrutura de GPU local superpotente; a versão Sabiazinho-4 é barata o suficiente para execuções em lote sem pesar no orçamento da pesquisa, com processamento quase instantâneo do lado do provedor.
  - **Desvantagem:** Dependência de rede e potenciais restrições de privacidade, pois a submissão de dados vai a um provedor externo.

- **LLaMA 3.2 (Local-based):**
  - **Acesso:** Inferência local no próprio hardware via biblioteca `transformers` no PyTorch (`llama_adapter.py` / `run_llama_simulation.py`).
  - **Vantagem:** Privacidade total, execução offline (útil para dados sensíveis em pesquisa). Nenhum custo financeiro atrelado ao modelo ou à inferência que não seja o custo da própria eletricidade/servidor local de GPU.
  - **Desvantagem:** Exige GPU compatível, pode ser mais demorado processar um grande "batch", exigindo eventuais perdas com parametrização de quantização dependendo da RAM.

- **GPT-5-mini (API-based / Batch):**
  - **Acesso:** Processamento assíncrono em lote via OpenAI Batch API (`generate_openai_batch.py`).
  - **Vantagem:** Altamente escalável e com custo 50% inferior em relação à inferência em tempo real. O uso obrigatório de *Structured Outputs* (JSON Schema estrito) garante que 100% das respostas cheguem formatadas no padrão exato esperado, sem desperdício de chamadas por erro de parsing.
  - **Desvantagem:** Latência de processamento de até 24 horas (embora costume ser finalizado em minutos ou poucas horas) e dependência de envio de dados para servidores terceiros nos EUA.

## 3. Discrepância na Eficiência Computacional vs Resultados

Todos os modelos otimizam para velocidade e eficiência sob estratégias distintas: o Llama 3B utiliza *Grouped-Query Attention (GQA)* para execução computacional rápida em hardware local (edge); a família Sabiá-4 foca no quadrante superior-esquerdo de custo-performance com APIs regionais de baixa latência; e o GPT-5-mini utiliza a infraestrutura assíncrona da OpenAI Batch API com descontos agressivos de preço por token.

Os testes evidenciam que usar SLMs como o Llama 3B é excelente como prova de conceito (POC) da arquitetura local. No entanto, o melhor balanço geral de simulação sociológica recai sobre o **Sabiazinho-4**. Enquanto o LLaMA exige suas versões de escala gigantesca (ex: 70B) para conseguir representar *nuances* multiculturais sutis (o que inviabiliza localmente o projeto), e o GPT-5-mini exibe alinhamentos homogêneos globais que achatam a diversidade sociodemográfica brasileira real, o Sabiazinho-4 entrega o viés sociológico brasileiro de forma nativa e econômica, sendo a escolha ideal e barata para as execuções em lote no projeto.

