# Modelo: GPT-5-mini

## 1. Informações Técnicas Gerais

- **Desenvolvedor:** OpenAI
- **Arquitetura e Tamanho:** Modelo proprietário de tamanho reduzido e altamente otimizado (tamanho exato de parâmetros não revelado publicamente).
- **Janela de Contexto:** 128.000 tokens de contexto, com suporte estendido para geração de tokens de saída.

## 2. Foco Estratégico e Treinamento

O GPT-5-mini foi projetado pela OpenAI para atuar como o modelo de alta eficiência de sua geração, oferecendo inteligência de fronteira a custos significativamente menores. Seus principais pilares incluem:

- **Eficiência Econômica:** Custo de inferência extremamente reduzido (input de $0.25 por 1M tokens e output de $2.00 por 1M tokens na API padrão), otimizando a execução de pipelines massivos de processamento de texto.
- **Saídas Estruturadas (Structured Outputs):** Suporte nativo a saídas estritamente validadas via *Structured Outputs* (usando JSON Schema com a propriedade `strict: true`), o que mitiga erros de parse de JSON e garante total fidelidade ao esquema especificado.
- **Alinhamento de Preferência:** Refinamento contínuo por meio de RLHF (Aprendizado por Reforço com Feedback Humano) focado na neutralidade de tom, mitigação de preconceitos e seguimento rígido de diretrizes de segurança ("safety alignment").

## 3. Desempenho e Viés em Simulações

Nos experimentos de simulação de opinião pública do repositório:
- **Baseline de Fronteira:** O modelo representa o comportamento de um agente geral de altíssima capacidade lógica e seguimento de instruções, servindo de baseline global para avaliar o comportamento de LLMs mais localizados ou compactos.
- **Efeito de Homogeneização:** Por ser treinado em um corpus massivo global e alinhado sob as diretrizes de segurança estritas da OpenAI, o GPT-5-mini exibe uma tendência de uniformização das opiniões. O modelo tende a favorecer respostas politicamente corretas ou neutras sob a persona atribuída, atenuando a polarização e as nuances regionais brasileiras que são mais evidentes em modelos focados regionalmente (como a família Sabiá).

## 4. Plataforma de Execução nos Experimentos

No repositório `@llm_experiments`, a execução do GPT-5-mini é otimizada para o processamento assíncrono em larga escala:

- **OpenAI Batch API:** O script `generate_openai_batch.py` converte os prompts estruturados em requisições de lote compatíveis com a Batch API da OpenAI, distribuindo-os em arquivos JSONL divididos por repetição (`batch_gpt-5-mini_rep1.jsonl` a `batch_gpt-5-mini_rep5.jsonl`).
- **Redução de Custo de Execução:** A Batch API oferece um desconto de 50% sobre os preços tradicionais (reduzindo o custo de entrada para $0.125/1M tokens e de saída para $1.00/1M tokens), tornando financeiramente viável a execução de 10.000 requisições por rodada de pesquisa (2.000 personas em 5 repetições).
- **Garantia de Esquema:** A orquestração do lote impõe o uso de `response_format` com JSON Schema estrito, garantindo o retorno consistente dos campos `answer` (a letra da opção) e `explanation` (justificativa de até 85 palavras).
