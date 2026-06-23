from settings import settings
from infrastructure.cache import DistributionCache
from infrastructure.sidra import SidraClient, SidraQueryBuilder
from ..exceptions import MaritalStatusDistributionError
from .utils import get_cached_or_fetch, safe_float


def get_marital_status_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.civil.marital_status
    period = cfg.period

    def fetch():
        try:
            query = (
                SidraQueryBuilder(cfg.table_id)
                .select_periods(period)
                .select_variables(cfg.variable_id)
                .select_locations(cfg.location_level)
            )
            for classification in cfg.classifications:
                query.select_classification(
                    classification.id, classification.categories
                )

            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise MaritalStatusDistributionError(
                    "No data returned from SIDRA for marital status distribution"
                )

            raw = {}
            for res in data[0].resultados:
                status_name = None
                for c in res.classificacoes:
                    if c.id == "9832":
                        for cat_name in c.categoria.values():
                            if cat_name != "Total":
                                status_name = cat_name

                if status_name:
                    for serie in res.series:
                        val_str = serie.serie.get(period)
                        val = safe_float(val_str)
                        if val is not None:
                            raw[status_name] = val

            total = sum(raw.values())
            if total <= 0:
                raise MaritalStatusDistributionError(
                    "Marital status distribution total is zero or negative"
                )

            # Apply mappings if any exist
            mapped_raw = {}
            for k, v in raw.items():
                mapped_k = cfg.mappings.get(k, k)
                mapped_raw[mapped_k] = mapped_raw.get(mapped_k, 0.0) + v

            return {k: (v / total) * 100 for k, v in mapped_raw.items()}
        except Exception as e:
            if isinstance(e, MaritalStatusDistributionError):
                raise
            raise MaritalStatusDistributionError(
                "Failed to fetch or parse marital status distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, period, fetch
    )
