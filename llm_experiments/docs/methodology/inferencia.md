# Metodologia: Inferência e Prompting

## 1. Fundamentação Teórica

O fluxo de inferência sustenta a segunda etapa do modelo de **Silicon Sampling** para simulação da opinião pública. A técnica metodológica principal não avalia o modelo apenas de forma pontual (Acurácia), já que LLMs sofrem do fenômeno de *homogeneização* — tendem a escolher invariavelmente uma única resposta politicamente correta/esperada ao mesmo *prompt*.

Para extrair o valor social verdadeiro, o experimento é desenhado para medir a variância distribucional, observada pelas respostas agrupadas do lote inteiro e checada matematicamente via Divergência Jensen-Shannon (JSD).

## 2. Engenharia de Prompts (Prompt Building)

Para condicionar o LLM a raciocinar *através das lentes* do perfil sintético e não como um assistente virtual neutro, a configuração dos prompts isola perfeitamente o contexto das instruções através das chamadas *System Messages*.

### Construção do System Prompt

O módulo `prompt_builder.py` executa a função *zero-shot roleplay* formatando a classe `Persona` em um bloco rígido de identidade. Exemplo do template utilizado no código:

> *"Você é um cidadão brasileiro com o seguinte perfil:*
> *- Idade: 30 a 39 anos*
> *- Gênero: Feminino*
> *- Raça/Cor: Parda*
> *- Região do Brasil: Sudeste (Zona Urbana)*
> *- Escolaridade: Ensino Médio Completo*
> *- Faixa de renda: 2 a 3 salários mínimos*
> *- Religião: Católica*
> [...]
> *Sua tarefa é responder a perguntas de opinião pública como se você fosse essa pessoa. Considere as condições socioeconômicas e demográficas descritas para formar sua opinião, refletindo o contexto real do Brasil. Não saia do personagem."*

### Injeção da Pesquisa

O sistema então envia um *User Prompt* contendo os dados carregados das definições YAML do questionário (ex: `survey_percepcao_democracia.yaml`), apresentando a Pergunta e enumerando rigidamente as Alternativas disponíveis.

## 3. Parâmetros de Inferência

As configurações enviadas ao LLM (seja pela chamada REST para os servidores da Maritaca AI, seja localmente pela execução das classes Pytorch `generation_config`) determinam profundamente a eficácia da pesquisa sociológica emula:

- **Temperature (Temperatura = 0.7):** Um valor alto-médio (maior que zero) é de rigor na metodologia para impedir o determinismo puro. Para o *roleplay* ser condizente com humanos, certa aleatoriedade permite que o modelo flutue suas repostas em distribuições (simulando que pessoas idênticas podem ter divergências menores nas decisões), emulando as flutuações amostrais genuínas da população em massa.
- **Tokens Máximos (`max_output_tokens = 200`):** Foi imposto para limitar devaneios (alucinações). Ao forçar o modelo a retornar a estrutura curta, ele prioriza imediatamente a extração da opção final solicitada.
- **Reproduções (`repetitions`):** Controladas pelo arquivo `config.yaml` (`reproducoes`), permite passar a mesma pergunta à mesma persona $N$ vezes, aumentando a validade interna do processo estocástico do LLM, permitindo verificar que a distribuição das preferências daquele indivíduo se mantêm logicamente atrelada ao seu *background* econômico inserido no prompt e não meramente a flutuações aleatórias cegas do *softmax*.
