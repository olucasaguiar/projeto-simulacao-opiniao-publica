import json
import logging
from typing import List
from features.generate_persona.models import Persona
from infrastructure.llm import BaseLLMClient
from .models import (
    OpinionQuestion,
    ModelOpinionResponse,
    PersonaSimulationResult,
    SimulationComparison,
)
from .prompt_builder import build_persona_system_prompt, build_question_prompt

logger = logging.getLogger(__name__)


def parse_model_response(raw_resp: str, options: List[str]) -> tuple[str, str]:
    raw_resp_clean = raw_resp.strip()

    # 1. Try parsing whole response as JSON
    try:
        data = json.loads(raw_resp_clean)
        if isinstance(data, dict) and "chosen_option" in data:
            return str(data["chosen_option"]), str(data.get("explanation", ""))
    except json.JSONDecodeError:
        pass

    # 2. Try finding substring starting with { and ending with }
    # Try sliding window from outside in
    start_idx = 0
    while True:
        start_idx = raw_resp_clean.find("{", start_idx)
        if start_idx == -1:
            break
        end_idx = raw_resp_clean.rfind("}")
        while end_idx > start_idx:
            try:
                json_str = raw_resp_clean[start_idx : end_idx + 1]
                data = json.loads(json_str)
                if isinstance(data, dict) and "chosen_option" in data:
                    return str(data["chosen_option"]), str(data.get("explanation", ""))
            except json.JSONDecodeError:
                pass
            end_idx = raw_resp_clean.rfind("}", start_idx, end_idx)
        start_idx += 1

    # 3. Fallback: case-insensitive search of option names in text
    for opt in options:
        if opt.lower() in raw_resp_clean.lower():
            return opt, raw_resp_clean

    return "Erro/Indefinido", raw_resp_clean


def run_simulation(
    personas: List[Persona],
    questions: List[OpinionQuestion],
    clients: List[BaseLLMClient],
) -> List[SimulationComparison]:
    """
    Run the opinion simulation for a batch of personas across multiple models.
    Loads and runs one model at a time, then frees its memory to save GPU resources.
    """
    responses_map = {}

    for client in clients:
        logger.info(f"Starting execution for model: {client.model_id}")

        for question in questions:
            # Schema for parsing JSON from models
            json_schema = {
                "type": "object",
                "properties": {
                    "chosen_option": {
                        "type": "string",
                        "description": "The exact text of the chosen option.",
                        "enum": question.options,
                    },
                    "explanation": {
                        "type": "string",
                        "description": "The reasoning behind the choice in the first person.",
                    },
                },
                "required": ["chosen_option", "explanation"],
                "additionalProperties": False,
            }

            user_prompt = build_question_prompt(question)

            for persona in personas:
                system_prompt = build_persona_system_prompt(persona)

                try:
                    logger.debug(
                        f"Querying {client.model_id} for persona {str(persona.id)[:8]}..."
                    )
                    raw_resp = client.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_schema=json_schema,
                    )

                    chosen, explanation = parse_model_response(
                        raw_resp, question.options
                    )

                    # Validate chosen option against available options (case-insensitive alignment)
                    matched_option = "Erro/Indefinido"
                    for opt in question.options:
                        if chosen.lower() == opt.lower():
                            matched_option = opt
                            break

                    # If we couldn't match, check if raw explanation contains it
                    if matched_option == "Erro/Indefinido":
                        for opt in question.options:
                            if opt.lower() in explanation.lower():
                                matched_option = opt
                                break

                    response_obj = ModelOpinionResponse(
                        model_id=client.model_id,
                        question_id=question.id,
                        persona_id=str(persona.id),
                        chosen_option=matched_option,
                        explanation=explanation,
                    )

                except Exception as e:
                    logger.error(f"Error querying {client.model_id}: {e}")
                    response_obj = ModelOpinionResponse(
                        model_id=client.model_id,
                        question_id=question.id,
                        persona_id=str(persona.id),
                        chosen_option="Erro",
                        explanation=str(e),
                    )

                responses_map[(client.model_id, question.id, str(persona.id))] = (
                    response_obj
                )

        # Free GPU memory immediately after processing all questions for this model
        if hasattr(client, "free_memory"):
            logger.info(f"Deallocating model from memory: {client.model_id}")
            try:
                client.free_memory()
            except Exception as e:
                logger.error(f"Error deallocating model {client.model_id}: {e}")

    # Restructure results into original List[SimulationComparison] layout
    comparisons = []
    for question in questions:
        persona_results = []
        for persona in personas:
            model_responses = []
            for client in clients:
                resp = responses_map.get(
                    (client.model_id, question.id, str(persona.id))
                )
                if resp:
                    model_responses.append(resp)
            persona_results.append(
                PersonaSimulationResult(persona=persona, responses=model_responses)
            )
        comparisons.append(
            SimulationComparison(question=question, persona_results=persona_results)
        )

    return comparisons
