from pathlib import Path
from infrastructure.sidra import SidraClient
from infrastructure.cache import DistributionCache
from features.generate_persona import generate_batch
from settings import settings


def main() -> None:
    # Resolve the database path relative to project root
    project_root = Path(__file__).parent.parent
    db_path = project_root / settings.paths.distributions_db

    with SidraClient() as client, DistributionCache(db_path) as cache:
        print("Generating personas using IBGE/SIDRA data pipeline...")
        amount = settings.persona_generation.default_amount
        indent = settings.persona_generation.json_indent
        personas = generate_batch(client=client, cache=cache, count=amount)
        for i, persona in enumerate(personas, 1):
            print(f"\n=== PERSONA {i} ===")
            print(persona.model_dump_json(indent=indent))


if __name__ == "__main__":
    main()
