# Execução Experimental e Configurações de Inferência

Este documento detalha o desenho prático e a infraestrutura de execução das simulações de opinião pública, complementando a fundamentação de inferência descrita em [inferencia.md](file:///home/lucasaguiar/projetos/olucasaguiar/projeto-simulacao-opiniao-publica/llm_experiments/docs/methodology/inferencia.md).

## 1. Modelos de Linguagem Utilizados

Para avaliar a consistência das simulações de opinião pública sob diferentes arquiteturas e capacidades cognitivas, as inferências foram realizadas utilizando os seguintes modelos de linguagem:

- **gpt-5-mini** (OpenAI)
- **sabia-4** (Maritaca AI)
- **sabiazinho-4** (Maritaca AI)
- **llama-3.2** (Meta)

Esses modelos foram integrados e configurados de acordo com as especificações contidas na infraestrutura de adapters em [infrastructure/llm](file:///home/lucasaguiar/projetos/olucasaguiar/projeto-simulacao-opiniao-publica/llm_experiments/src/infrastructure/llm).

## 2. População Sintética e Protocolo de Repetições

O desenho do experimento visa mitigar o ruído estocástico das escolhas dos modelos e garantir a validade estatística dos resultados:

- **Volume Populacional:** O experimento é executado sobre uma base de **2.000 personas (indivíduos sintéticos)**. Este conjunto reproduz de forma precisa a distribuição demográfica e socioeconômica nacional, baseando-se nos microdados do Censo IBGE e da PNAD Contínua.
- **Protocolo de Repetições:** Cada pergunta do questionário é aplicada **5 vezes para cada persona**. A repetição é fundamental para mensurar a variação e estabilidade da inferência sob a temperatura configurada (`temperature = 0.7`), permitindo traçar uma curva de distribuição probabilística fidedigna das preferências e atitudes de cada perfil sintético, em vez de depender de respostas pontuais.

## 3. Estruturação Completa dos Prompts

Para condicionar os modelos a emularem fielmente a identidade sociodemográfica designada, o sistema gera chamadas estruturadas compostas por mensagens de sistema (*System Prompt*), do usuário (*User Prompt*) e esquemas rígidos de saída (*JSON Schema*).

### Exemplo de Prompt de Inferência

Abaixo é apresentado um caso real de chamada estruturada enviada aos modelos de linguagem:

#### A. System Prompt (Mensagem de Sistema)
Gera o condicionamento de persona do módulo [prompt_builder.py](file:///home/lucasaguiar/projetos/olucasaguiar/projeto-simulacao-opiniao-publica/llm_experiments/src/features/opinion_simulation/prompt_builder.py) a partir do modelo [Persona](file:///home/lucasaguiar/projetos/olucasaguiar/projeto-simulacao-opiniao-publica/llm_experiments/src/features/generate_persona/models.py#L36):

```text
Você é um cidadão brasileiro com o seguinte perfil:
- Idade: 30 a 39 anos
- Gênero: Feminino
- Raça/Cor: Parda
- Região do Brasil: Nordeste (Zona Urbana)
- Escolaridade: Ensino Médio Completo
- Situação de emprego: Ocupada
- Faixa de renda: De 1 a 2 salários mínimos
- Estado civil: Casada
- Religião: Católica
- Autoavaliação de saúde: Bom

Sua tarefa é responder a perguntas de opinião pública como se você fosse essa pessoa. Considere as condições socioeconômicas e demográficas descritas para formar sua opinião, refletindo o contexto real do Brasil. Não saia do personagem. Responda de forma sincera baseando-se nas experiências prováveis de alguém com este exato perfil.
```

#### B. User Prompt (Mensagem do Usuário)
Injeta a pergunta e alternativas de resposta oriundas da definição em YAML da pesquisa (por exemplo, [survey_percepcao_democracia.yaml](file:///home/lucasaguiar/projetos/olucasaguiar/projeto-simulacao-opiniao-publica/llm_experiments/assets/surveys/survey_percepcao_democracia.yaml)):

```text
Com qual destas frases você está mais de acordo?

Opções:
a: A democracia é preferível a qualquer outra forma de governo
b: Em algumas circunstâncias, um governo autoritário pode ser preferível a um democrático
c: Para pessoas como eu, tanto faz um governo democrático como um não democrático

O JSON deve ter as chaves 'answer' (apenas a letra da opção escolhida) e 'explanation' (resumo do porquê).
```

#### C. Esquema de Saída Rígido (JSON Schema)
Esquema estruturado exigido na chamada de inferência para assegurar a leitura computacional das respostas:

```json
{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "description": "A letra da alternativa escolhida para a sua resposta"
    },
    "explanation": {
      "type": "string",
      "description": "Justificativa em relação a alternativa escolhida"
    }
  },
  "required": ["answer", "explanation"]
}
```

#### D. Resposta de Inferência do Modelo (Output)
```json
{
  "answer": "a",
  "explanation": "Acredito que a democracia garante direitos básicos de expressão e melhores oportunidades para trabalhadores de classe média e baixa, mesmo com os desafios políticos do país."
}
```
