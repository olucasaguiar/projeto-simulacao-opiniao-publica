import json
import logging
from typing import List, Dict, Any
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


def _setup_file_logger(results_path: str) -> logging.FileHandler:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"{results_path}/simulation_{timestamp}.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    return file_handler


def _calculate_answer_distribution(
    answers: List[Any], options: Dict[str, str]
) -> Dict[str, float]:
    distribution = {key: 0.0 for key in options.keys()}
    total_answers = len(answers)
    
    if total_answers == 0:
        return distribution
        
    counts = defaultdict(int)
    for answer in answers:
        counts[answer.answer[0]] += 1
        
    for key in options.keys():
        distribution[key] = counts.get(key, 0) / total_answers
        
    return distribution


def _build_message_history_update(question: Any, answer: Any) -> List[Dict[str, str]]:
    options_str = "\n".join([f"{k}) {v}" for k, v in question.options.items()])
    return [
        {
            "role": "assistant",
            "content": f"Pergunta: {question.text}\nAlternativas:\n{options_str}",
        },
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
    ]


def _compile_simulation_results(
    personas: List[Persona], 
    config: SimulationConfig, 
    results_map: Dict[Any, Dict[Any, List[Any]]]
) -> List[PersonaSimulationResult]:
    all_questions = {q.id: q for survey in config.surveys for q in survey.questions}
    persona_results = []

    for persona in personas:
        responses = []
        for question_id, question in all_questions.items():
            answers = results_map[persona.id].get(question_id, [])
            distribution = _calculate_answer_distribution(answers, question.options)
            
            result = FormResults(distribution=distribution)
            responses.append(
                FormResponse(
                    question=question.text,
                    options=question.options,
                    answers=answers,
                    result=result,
                )
            )

        persona_results.append(
            PersonaSimulationResult(persona=persona, responses=responses)
        )

    return persona_results


def run_simulation(
    config: SimulationConfig,
    personas: List[Persona],
) -> List[PersonaSimulationResult]:
    factory = LLMFactory()
    file_handler = _setup_file_logger(config.results_path)
    
    results_map = defaultdict(lambda: defaultdict(list))

    for model_id in config.models:
        client = factory.provide(model_id)
        if not client:
            logger.error(f"Modelo '{model_id}' não suportado ou não encontrado no factory. Pulando.")
            continue

        logger.info(f"Iniciando simulação com o modelo: {model_id}")

        with client:
            for persona in personas:
                system_prompt = build_persona_system_prompt(persona)

                for rep in range(config.repetitions):
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
                                    max_output_tokens=200,
                                )

                                messages.extend(_build_message_history_update(question, answer))
                                
                                logger.info(
                                    f"RESULTADO | Modelo={model_id} | Persona={str(persona.id)[:8]} | "
                                    f"Rep={rep + 1} | Q={question.id} | Resposta={answer.answer[0]}"
                                )

                                results_map[persona.id][question.id].append(answer)

                            except Exception as e:
                                logger.error(
                                    f"Erro ao processar (modelo={model_id}, persona={str(persona.id)[:8]}, "
                                    f"Q={question.id}): {e}"
                                )

    logger.removeHandler(file_handler)
    file_handler.close()

    return _compile_simulation_results(personas, config, results_map)
