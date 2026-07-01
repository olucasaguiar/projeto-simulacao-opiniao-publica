import sys
import argparse
import json
import logging
from pathlib import Path
import re
import yaml
from jinja2 import Template

# Ensure the src directory is in sys.path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

uuid_pattern = re.compile(
    r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def get_persona_id_from_custom_id(custom_id: str) -> str:
    match = uuid_pattern.match(custom_id)
    return match.group(1) if match else custom_id


def is_continuation_of(response_id: str, history_id: str) -> bool:
    return response_id == history_id or response_id.startswith(history_id + "_")


response_cache = {}


def load_responses(path_str: str) -> dict:
    """
    Loads responses from a batch results jsonl file and caches them.
    Returns a dictionary mapping persona_id -> {custom_id: content}.
    """
    if path_str in response_cache:
        return response_cache[path_str]

    path = Path(path_str)
    if not path.exists():
        logger.error(f"Arquivo de resposta do assistant não encontrado: {path}")
        sys.exit(1)

    logger.info(f"Carregando respostas do assistant de {path}...")
    
    # Dict of dicts: persona_id -> {custom_id: content}
    grouped_responses = {}
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                logger.error(f"Erro ao decodificar JSON na linha {line_num} de {path}: {e}")
                continue

            custom_id = data.get("custom_id")
            if not custom_id:
                continue

            # check error
            if data.get("error") is not None:
                continue

            resp_obj = data.get("response")
            if not resp_obj or resp_obj.get("status_code") != 200:
                continue

            choices = resp_obj.get("body", {}).get("choices", [])
            if not choices:
                continue

            content = choices[0].get("message", {}).get("content")
            if content is not None:
                persona_id = get_persona_id_from_custom_id(custom_id)
                if persona_id not in grouped_responses:
                    grouped_responses[persona_id] = {}
                grouped_responses[persona_id][custom_id] = content

    response_cache[path_str] = grouped_responses
    return grouped_responses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera arquivo batch para OpenAI API a partir de personas e um template YAML/Jinja2."
    )
    parser.add_argument(
        "template_file",
        type=str,
        help="Caminho para o arquivo YAML de template.",
    )
    parser.add_argument(
        "personas_file",
        type=str,
        help="Caminho para o arquivo JSONL de personas.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Caminho para o arquivo JSONL de saída do batch.",
    )
    args = parser.parse_args()

    # Load the YAML template
    template_path = Path(args.template_file)
    if not template_path.exists():
        logger.error(f"Arquivo de template não encontrado: {template_path}")
        sys.exit(1)

    logger.info(f"Lendo template de {template_path}...")
    with open(template_path, "r", encoding="utf-8") as f:
        try:
            template_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erro ao ler YAML do template: {e}")
            sys.exit(1)

    if not isinstance(template_config, dict):
        logger.error("O arquivo de template deve ser um YAML válido com um dicionário na raiz.")
        sys.exit(1)

    # Extract prompt fields
    system_template = template_config.pop("system", None)
    messages_templates = template_config.pop("messages", [])

    if not isinstance(messages_templates, list):
        logger.error("A chave 'messages' no template deve ser uma lista.")
        sys.exit(1)

    # Resolve parameters from template or fallbacks
    model = template_config.pop("model", "sabiazinho-4")
    temperature = template_config.pop("temperature", 0.7)
    max_tokens = template_config.pop("max_tokens", 1000)

    # Compile templates
    try:
        system_tmpl = Template(system_template) if system_template else None
    except Exception as e:
        logger.error(f"Erro ao compilar template do system prompt: {e}")
        sys.exit(1)

    msg_tmpls = []
    for idx, msg in enumerate(messages_templates):
        role = msg.get("role")
        content = msg.get("content")
        path = msg.get("path")
        if not role:
            logger.error(f"Mensagem no índice {idx} inválida. Deve conter 'role'.")
            sys.exit(1)

        if role == "assistant" and path is not None:
            msg_tmpls.append({
                "role": role,
                "path": path,
                "content_tmpl": None
            })
        else:
            if content is None:
                logger.error(f"Mensagem no índice {idx} inválida. Deve conter 'content'.")
                sys.exit(1)
            try:
                msg_tmpls.append({
                    "role": role,
                    "path": None,
                    "content_tmpl": Template(content)
                })
            except Exception as e:
                logger.error(f"Erro ao compilar template da mensagem no índice {idx}: {e}")
                sys.exit(1)

    # Read personas and write batch records
    personas_path = Path(args.personas_file)
    if not personas_path.exists():
        logger.error(f"Arquivo de personas não encontrado: {personas_path}")
        sys.exit(1)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processando personas de {personas_path}...")
    records_written = 0

    with open(personas_path, "r", encoding="utf-8") as pf, open(output_path, "w", encoding="utf-8") as out_f:
        for line_num, line in enumerate(pf, 1):
            line = line.strip()
            if not line:
                continue
            try:
                persona = json.loads(line)
            except Exception as e:
                logger.error(f"Erro ao decodificar JSON na linha {line_num} de {personas_path}: {e}")
                continue

            persona_id = persona.get("id")
            if not persona_id:
                logger.warning(f"Persona na linha {line_num} sem chave 'id'. Ignorando.")
                continue

            # Build Jinja context
            demo = persona.get("demographic", {})
            eco = persona.get("economic", {})
            health = persona.get("health", {})
            social = persona.get("social", {})

            context = {
                "persona": persona,
                # Demographic
                "age_group": demo.get("age_group"),
                "gender": demo.get("gender"),
                "race": demo.get("race"),
                "region": demo.get("region"),
                "urban": demo.get("urban"),
                "urban_str": "Urbana" if demo.get("urban") else "Rural",
                # Economic
                "education_level": eco.get("education_level"),
                "employment_status": eco.get("employment_status"),
                "income_per_capita": eco.get("income_per_capita"),
                "inflation_rate": eco.get("inflation_rate"),
                # Health
                "health_self_assessment": health.get("health_self_assessment"),
                "has_chronic_disease": health.get("has_chronic_disease"),
                # Social
                "marital_status": social.get("marital_status"),
                "religion": social.get("religion"),
            }

            # Render system prompt
            system_messages = []
            if system_tmpl:
                try:
                    system_messages.append({
                        "role": "system",
                        "content": system_tmpl.render(context)
                    })
                except Exception as e:
                    logger.error(f"Erro ao renderizar system prompt para persona {persona_id}: {e}")
                    continue

            histories = [{"custom_id": str(persona_id), "messages": system_messages}]

            # Process prompt messages and follow-up branches
            skip_persona = False
            for idx, msg_info in enumerate(msg_tmpls):
                if msg_info["path"] is not None:
                    path_str = msg_info["path"]
                    try:
                        grouped_resps = load_responses(path_str)
                    except Exception as e:
                        logger.error(f"Erro ao carregar respostas do assistente: {e}")
                        skip_persona = True
                        break
                    
                    persona_resps = grouped_resps.get(str(persona_id), {})
                    new_histories = []
                    for hist in histories:
                        hist_id = hist["custom_id"]
                        for resp_id, resp_content in persona_resps.items():
                            if is_continuation_of(resp_id, hist_id):
                                new_histories.append({
                                    "custom_id": resp_id,
                                    "messages": hist["messages"] + [{"role": "assistant", "content": resp_content}]
                                })
                    histories = new_histories
                else:
                    try:
                        content_tmpl = msg_info["content_tmpl"]
                        rendered_content = content_tmpl.render(context)
                        for hist in histories:
                            hist["messages"] = hist["messages"] + [{
                                "role": msg_info["role"],
                                "content": rendered_content
                            }]
                    except Exception as e:
                        logger.error(f"Erro ao renderizar mensagem {idx} para persona {persona_id}: {e}")
                        skip_persona = True
                        break

            if skip_persona or not histories:
                continue

            # Write all generated history branches to the batch file
            for hist in histories:
                body = {
                    "model": model,
                    "messages": hist["messages"],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **template_config  # This passes response_format and any other custom parameter
                }

                record = {
                    "custom_id": hist["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

    logger.info(f"Concluído! {records_written} registros salvos em {output_path}.")


if __name__ == "__main__":
    main()
