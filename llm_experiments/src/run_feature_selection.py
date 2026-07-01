import os
import sys
import json
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure the src directory is in sys.path
sys.path.append(str(Path(__file__).parent))

from settings import settings
from infrastructure.llm.maritaca_adapter import MaritacaAdapter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    prompts_dir = project_root / "assets" / "prompts"
    output_dir = project_root / "assets" / "feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate the adapter to access configuration and backoff helper
    # This also validates that MARITACA_API_KEY is present
    try:
        adapter = MaritacaAdapter(model_id="sabiazinho-4")
    except Exception as e:
        logger.error(f"Erro ao inicializar o MaritacaAdapter: {e}")
        sys.exit(1)

    # Prompt configurations to run
    prompt_files = [
        "feature_selection_political_priorities.yaml",
        "feature_selection_fake_news.yaml",
        "feature_selection_political_participation.yaml",
    ]

    headers = {
        "Authorization": f"Key {adapter.maritaca_api_key}",
        "Content-Type": "application/json",
    }
    endpoint = f"{adapter.base_url}/v1/responses"

    for prompt_file in prompt_files:
        prompt_path = prompts_dir / prompt_file
        if not prompt_path.exists():
            logger.warning(f"Arquivo de prompt não encontrado: {prompt_path}. Pulando.")
            continue

        logger.info(f"Lendo prompt de: {prompt_path.name}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f)

        # Map response_format to the schema structure expected by Maritaca AI
        schema = prompt_data.get("response_format", {}).get("json_schema", {}).get("schema")
        if not schema:
            logger.error(f"Esquema JSON não encontrado no arquivo {prompt_file}. Pulando.")
            continue

        payload = {
            "model": prompt_data.get("model", adapter.model_id),
            "instructions": prompt_data.get("system"),
            "input": prompt_data.get("messages"),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "feature_selection_schema",
                    "schema": schema,
                    "strict": True,
                }
            },
            "temperature": prompt_data.get("temperature", 0.3),
            "max_output_tokens": prompt_data.get("max_tokens", 150),
        }

        logger.info(f"Enviando requisição de feature selection para {prompt_file}...")
        try:
            response_data = adapter._post_with_backoff(endpoint, headers, payload)
            content = response_data["output"][0]["content"][0]["text"]

            # Parse response content
            parsed_json = None
            try:
                parsed_json = json.loads(content.strip())
            except Exception:
                import re
                match = re.search(r"\{.*\}", content.strip(), re.DOTALL)
                if match:
                    parsed_json = json.loads(match.group(0))
                else:
                    raise ValueError(f"Não foi possível extrair um JSON válido da resposta: {content}")

            # Save to disk
            output_file_name = prompt_path.stem + ".json"
            output_file_path = output_dir / output_file_name
            with open(output_file_path, "w", encoding="utf-8") as out_f:
                json.dump(parsed_json, out_f, ensure_ascii=False, indent=2)

            logger.info(f"Feature selection salva com sucesso em: {output_file_path}")

        except Exception as e:
            logger.error(f"Erro ao processar a seleção de features para {prompt_file}: {e}")


if __name__ == "__main__":
    main()
