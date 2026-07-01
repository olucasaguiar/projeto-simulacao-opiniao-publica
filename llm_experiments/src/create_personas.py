import sys
import argparse
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure the src directory is in sys.path
sys.path.append(str(Path(__file__).parent))

from settings import settings
from infrastructure.sidra import SidraClient
from infrastructure.cache import DistributionCache
from features.generate_persona import generate_batch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Gerador de personas usando dados do IBGE/SIDRA."
    )
    parser.add_argument(
        "count",
        type=int,
        help="Número de personas a serem geradas.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Caminho do arquivo .jsonl de saída.",
    )
    args = parser.parse_args()

    # Resolve database path relative to project root (llm_experiments)
    project_root = Path(__file__).parent.parent
    db_path = project_root / settings.paths.distributions_db

    logger.info(f"Conectando ao banco de distribuições em: {db_path}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with SidraClient() as client, DistributionCache(db_path) as cache:
        logger.info(f"Gerando {args.count} persona(s)...")
        personas = generate_batch(client=client, cache=cache, count=args.count)
        logger.info("Personas geradas com sucesso.")

    logger.info(f"Salvando personas em {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for persona in personas:
            # Pydantic v2 serialization
            persona_json = json.dumps(persona.model_dump(mode="json"), ensure_ascii=False)
            f.write(persona_json + "\n")

    logger.info("Concluído!")


if __name__ == "__main__":
    main()
