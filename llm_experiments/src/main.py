import argparse
import yaml
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from infrastructure.sidra import SidraClient
from infrastructure.cache import DistributionCache
from features.generate_persona import generate_batch
from settings import settings

from features.opinion_simulation.models import SimulationConfig, Survey, SurveyQuestion
from features.opinion_simulation.handler import run_simulation


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_yaml_config(filepath: str) -> SimulationConfig:
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Translate from YAML PT-BR keys to SimulationConfig keys
    cenario = data.get("cenario", {})
    resultados = data.get("resultados", {})
    pesquisas_data = data.get("pesquisas", [])

    surveys = []
    for p in pesquisas_data:
        questions = []
        for q in p.get("perguntas", []):
            questions.append(
                SurveyQuestion(
                    id=str(q["id"]),
                    topic=q["topico"],
                    text=q["texto"],
                    options=q["alternativas"],
                )
            )
        surveys.append(Survey(id=str(p["id"]), title=p["titulo"], questions=questions))

    config = SimulationConfig(
        personas=cenario.get("personas", 1),
        repetitions=cenario.get("reproducoes", 1),
        models=cenario.get("modelos", []),
        results_path=resultados.get("caminho", "./resultados"),
        surveys=surveys,
    )
    return config


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Executar simulação de opinião com LLMs."
    )
    parser.add_argument(
        "--survey",
        required=True,
        help="Caminho para o arquivo YAML de configuração da pesquisa.",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.survey)
    if not config_path.exists():
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        return

    config = parse_yaml_config(str(config_path))

    # Ensure results directory exists
    results_dir = Path(config.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate Personas
    project_root = Path(__file__).parent.parent
    db_path = project_root / settings.paths.distributions_db

    with SidraClient() as client, DistributionCache(db_path) as cache:
        logger.info(f"Gerando {config.personas} persona(s) usando IBGE/SIDRA...")
        personas = generate_batch(client=client, cache=cache, count=config.personas)
        logger.info("Personas geradas com sucesso.")

    # Run Simulation
    logger.info("Iniciando simulação com LLMs...")
    results = run_simulation(config=config, personas=personas)

    # Save Results
    output_file = results_dir / f"resultados_{config_path.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        # Pydantic v2 dump
        out_data = [r.model_dump(mode="json") for r in results]
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Simulação concluída! Resultados salvos em {output_file}")


if __name__ == "__main__":
    main()
