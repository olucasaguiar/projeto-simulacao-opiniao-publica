import json
import logging
from typing import List
from datetime import datetime
from collections import defaultdict

from features.generate_persona.models import Persona
from infrastructure.llm import LLMFactory
from .models import (
    FormResults,
    FormResponse,
    PersonaSimulationResult,
    SimulationConfig,
)
from .prompt_builder import build_persona_system_prompt

logger = logging.getLogger(__name__)


def run_simulation(
    config: SimulationConfig,
    personas: List[Persona],
) -> List[PersonaSimulationResult]:
    """
    Executa a simulação baseada na configuração e lista de personas.
    O loop é feito por: Modelo -> Persona -> Repetição -> Pergunta.
    O context manager de cada client cuida de liberar recursos.
    """
    factory = LLMFactory()

    # Criar um logger de arquivo para auditoria
    file_handler = logging.FileHandler(
        f"{config.results_path}/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Armazena todas as respostas coletadas.
    # A estrutura aqui é um dicionário temporário para agregar e formar os FormResponse.
    # Map: persona_id -> question_id -> list of ModelAnswer (across repetitions & models)
    # Actually, the requirement asks to compute FormResults per persona, per question.
    # We will accumulate by (persona_id, question_id).
    results_map = defaultdict(lambda: defaultdict(list))

    # Iterate for each model
    for model_id in config.models:
        client = factory.provide(model_id)
        if not client:
            logger.error(
                f"Modelo '{model_id}' não suportado/encontrado no factory. Pulando."
            )
            continue

        logger.info(f"Iniciando simulação com o modelo: {model_id}")

        with client:
            for persona in personas:
                system_prompt = build_persona_system_prompt(persona)

                for rep in range(config.repetitions):
                    # Histórico limpo a cada nova repetição
                    messages = []

                    for survey in config.surveys:
                        for question in survey.questions:
                            logger.debug(
                                f"[{model_id}] Persona {str(persona.id)[:8]} | Rep {rep + 1} | Pergunta {question.id}"
                            )

                            try:
                                answer = client.question_answer(
                                    system=system_prompt,
                                    question=question.text,
                                    options=question.options,
                                    messages=messages,
                                )

                                # Atualizar o histórico para a próxima pergunta da mesma repetição
                                options_str = "\n".join(
                                    [f"{k}) {v}" for k, v in question.options.items()]
                                )
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": f"Pergunta: {question.text}\nAlternativas:\n{options_str}",
                                    }
                                )
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": json.dumps(
                                            {
                                                "answer_key": answer.answer[0],
                                                "answer_value": answer.answer[1],
                                                "explanation": answer.explanation,
                                            },
                                            ensure_ascii=False,
                                        ),
                                    }
                                )

                                # Logar auditoria
                                logger.info(
                                    f"RESULT | Model={model_id} | Persona={str(persona.id)[:8]} | Rep={rep + 1} | Q={question.id} | Key={answer.answer[0]}"
                                )

                                # Salvar
                                results_map[persona.id][question.id].append(answer)

                            except Exception as e:
                                logger.error(
                                    f"Erro ao processar modelo={model_id}, persona={str(persona.id)[:8]}, Q={question.id}: {e}"
                                )

    logger.removeHandler(file_handler)
    file_handler.close()

    # Agregar os resultados e calcular as distribuições (FormResults)
    persona_results = []

    # Build list of all questions to simplify building responses
    all_questions = {}
    for survey in config.surveys:
        for q in survey.questions:
            all_questions[q.id] = q

    for persona in personas:
        responses = []
        for q_id, q_obj in all_questions.items():
            answers = results_map[persona.id].get(q_id, [])

            # Calcular distribuição
            distribution = {}
            total = len(answers)
            if total > 0:
                counts = defaultdict(int)
                for a in answers:
                    counts[a.answer[0]] += 1
                for key in q_obj.options.keys():
                    distribution[key] = counts.get(key, 0) / total
            else:
                for key in q_obj.options.keys():
                    distribution[key] = 0.0

            result = FormResults(distribution=distribution)
            responses.append(
                FormResponse(
                    question=q_obj.text,
                    options=q_obj.options,
                    answers=answers,
                    result=result,
                )
            )

        persona_results.append(
            PersonaSimulationResult(persona=persona, responses=responses)
        )

    return persona_results
