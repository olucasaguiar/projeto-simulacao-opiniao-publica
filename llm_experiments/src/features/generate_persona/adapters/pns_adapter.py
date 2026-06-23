from infrastructure.cache import DistributionCache
from infrastructure.sidra import SidraClient, SidraQueryBuilder
from settings import settings
from ..exceptions import (
    HealthAssessmentDistributionError,
    ChronicDiseaseDistributionError,
)
from .utils import get_cached_or_fetch, safe_float


def get_health_assessment_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.pns.health_assessment
    splits = settings.pns.scale_splits

    def fetch():
        try:
            query = (
                SidraQueryBuilder(cfg.table_id)
                .select_periods(cfg.period)
                .select_variables(cfg.variable_id)
                .select_locations(cfg.location_level)
            )
            for c in cfg.classifications:
                query.select_classification(c.id, c.categories)

            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise HealthAssessmentDistributionError(
                    "No data returned from SIDRA for health assessment distribution"
                )

            raw = {}
            for res in data[0].resultados:
                status_name = None
                for c in res.classificacoes:
                    if c.id == str(cfg.classifications[0].id):
                        for cat_name in c.categoria.values():
                            if cat_name != "Total":
                                status_name = cat_name

                if status_name:
                    for serie in res.series:
                        val_str = serie.serie.get(cfg.period)
                        val = safe_float(val_str)
                        if val is not None:
                            raw[status_name] = val

            total = sum(raw.values())
            if total <= 0:
                raise HealthAssessmentDistributionError(
                    "Health assessment distribution total is zero or negative"
                )

            dist = {k: (v / total) * 100 for k, v in raw.items()}
            final_dist = {}

            # Map the 3-point scale from PNS to the 5-point scale in our target schema using settings splits
            if "Muito bom e bom" in dist:
                val = dist["Muito bom e bom"]
                final_dist["Muito bom"] = val * splits.get("Muito bom", 0.35)
                final_dist["Bom"] = val * splits.get("Bom", 0.65)
            if "Regular" in dist:
                final_dist["Regular"] = dist["Regular"]
            if "Ruim e muito ruim" in dist:
                val = dist["Ruim e muito ruim"]
                final_dist["Ruim"] = val * splits.get("Ruim", 0.70)
                final_dist["Muito ruim"] = val * splits.get("Muito ruim", 0.30)

            return final_dist
        except Exception as e:
            if isinstance(e, HealthAssessmentDistributionError):
                raise
            raise HealthAssessmentDistributionError(
                "Failed to fetch or parse health assessment distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, cfg.period, fetch
    )


def get_chronic_disease_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.pns.chronic_disease

    def fetch():
        try:
            query = (
                SidraQueryBuilder(cfg.table_id)
                .select_periods(cfg.period)
                .select_variables(cfg.variable_id)
                .select_locations(cfg.location_level)
            )
            for c in cfg.classifications:
                query.select_classification(c.id, c.categories)

            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise ChronicDiseaseDistributionError(
                    "No data returned from SIDRA for chronic disease rate"
                )

            resultado = data[0].resultados[0]
            val_str = resultado.series[0].serie.get(cfg.period)
            if not val_str:
                raise ChronicDiseaseDistributionError(
                    "Chronic disease rate value is empty"
                )

            pct = float(val_str)
            return {"Sim": pct, "Não": 100.0 - pct}
        except Exception as e:
            if isinstance(e, ChronicDiseaseDistributionError):
                raise
            raise ChronicDiseaseDistributionError(
                "Failed to fetch or parse chronic disease rate"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, cfg.period, fetch
    )
