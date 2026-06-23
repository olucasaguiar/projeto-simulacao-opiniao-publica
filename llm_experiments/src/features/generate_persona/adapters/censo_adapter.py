from settings import settings
from infrastructure.cache import DistributionCache
from infrastructure.sidra import SidraClient, SidraQueryBuilder
from ..exceptions import (
    RaceDistributionError,
    GenderDistributionError,
    RegionDistributionError,
    UrbanRuralDistributionError,
    ReligionDistributionError,
    IBGEDataError,
)
from .utils import get_cached_or_fetch, safe_float


def _build_query_from_config(cfg) -> SidraQueryBuilder:
    qb = (
        SidraQueryBuilder(cfg.table_id)
        .select_periods(cfg.period)
        .select_variables(cfg.variable_id)
        .select_locations(cfg.location_level)
    )
    for classification in cfg.classifications:
        qb.select_classification(classification.id, classification.categories)
    return qb


def get_race_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.censo.race
    period = cfg.period

    def fetch():
        try:
            query = _build_query_from_config(cfg)
            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise RaceDistributionError(
                    "No data returned from SIDRA for race distribution"
                )

            raw = {}
            for res in data[0].resultados:
                race_name = None
                for c in res.classificacoes:
                    if c.id == "86":
                        for cat_name in c.categoria.values():
                            if cat_name != "Total":
                                race_name = cat_name

                if race_name:
                    for serie in res.series:
                        val_str = serie.serie.get(period)
                        val = safe_float(val_str)
                        if val is not None:
                            raw[race_name] = val

            total = sum(raw.values())
            if total <= 0:
                raise RaceDistributionError(
                    "Race distribution total population is zero or negative"
                )

            # Map raw labels using mappings, defaulting to the original label if no mapping exists
            mapped_raw = {}
            for k, v in raw.items():
                mapped_k = cfg.mappings.get(k, k)
                mapped_raw[mapped_k] = mapped_raw.get(mapped_k, 0.0) + v

            return {k: (v / total) * 100 for k, v in mapped_raw.items()}
        except Exception as e:
            if isinstance(e, RaceDistributionError):
                raise
            raise RaceDistributionError(
                "Failed to fetch or parse race distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, period, fetch
    )


def get_gender_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.censo.gender
    period = cfg.period

    def fetch():
        try:
            query = _build_query_from_config(cfg)
            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise GenderDistributionError(
                    "No data returned from SIDRA for gender distribution"
                )

            raw = {}
            for res in data[0].resultados:
                gender_name = None
                for c in res.classificacoes:
                    if c.id == "2":
                        for cat_name in c.categoria.values():
                            if cat_name != "Total":
                                gender_name = cat_name

                if gender_name:
                    for serie in res.series:
                        val_str = serie.serie.get(period)
                        val = safe_float(val_str)
                        if val is not None:
                            raw[gender_name] = val

            total = sum(raw.values())
            if total <= 0:
                raise GenderDistributionError(
                    "Gender distribution total population is zero or negative"
                )

            # Apply mappings
            mapped_raw = {}
            for k, v in raw.items():
                mapped_k = cfg.mappings.get(k, k)
                mapped_raw[mapped_k] = mapped_raw.get(mapped_k, 0.0) + v

            return {k: (v / total) * 100 for k, v in mapped_raw.items()}
        except Exception as e:
            if isinstance(e, GenderDistributionError):
                raise
            raise GenderDistributionError(
                "Failed to fetch or parse gender distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, period, fetch
    )


def get_region_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.censo.region
    period = cfg.period

    def fetch():
        try:
            query = _build_query_from_config(cfg)
            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise RegionDistributionError(
                    "No data returned from SIDRA for region distribution"
                )

            raw = {}
            for res in data[0].resultados:
                for serie in res.series:
                    region_name = serie.localidade.nome
                    val_str = serie.serie.get(period)
                    val = safe_float(val_str)
                    if val is not None and region_name:
                        if region_name.startswith("Região "):
                            region_name = region_name[7:]
                        raw[region_name] = val

            total = sum(raw.values())
            if total <= 0:
                raise RegionDistributionError(
                    "Region distribution total population is zero or negative"
                )

            # Apply mappings
            mapped_raw = {}
            for k, v in raw.items():
                mapped_k = cfg.mappings.get(k, k)
                mapped_raw[mapped_k] = mapped_raw.get(mapped_k, 0.0) + v

            return {k: (v / total) * 100 for k, v in mapped_raw.items()}
        except Exception as e:
            if isinstance(e, RegionDistributionError):
                raise
            raise RegionDistributionError(
                "Failed to fetch or parse region distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, period, fetch
    )


def get_age_group_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.censo.age_group
    period = cfg.period

    def fetch():
        try:
            query = _build_query_from_config(cfg)
            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise IBGEDataError(
                    "No data returned from SIDRA for age group distribution"
                )

            age_counts = {}
            for res in data[0].resultados:
                age_cat_name = None
                for c in res.classificacoes:
                    if c.id == "287":
                        for cat_name in c.categoria.values():
                            if cat_name != "Total":
                                age_cat_name = cat_name

                if age_cat_name:
                    for serie in res.series:
                        val_str = serie.serie.get(period)
                        val = safe_float(val_str)
                        if val is not None:
                            age_counts[age_cat_name] = val

            brackets = {
                "18-24": 0,
                "25-34": 0,
                "35-44": 0,
                "45-54": 0,
                "55-64": 0,
                "65+": 0,
            }

            for cat_name, count in age_counts.items():
                age = None
                if (
                    cat_name == "Menos de 1 ano"
                    or "mês" in cat_name
                    or "meses" in cat_name
                ):
                    age = 0
                elif "ano" in cat_name:
                    parts = cat_name.split()
                    if parts[0].isdigit():
                        age = int(parts[0])

                if age is not None:
                    if 18 <= age <= 24:
                        brackets["18-24"] += count
                    elif 25 <= age <= 34:
                        brackets["25-34"] += count
                    elif 35 <= age <= 44:
                        brackets["35-44"] += count
                    elif 45 <= age <= 54:
                        brackets["45-54"] += count
                    elif 55 <= age <= 64:
                        brackets["55-64"] += count
                    elif age >= 65:
                        brackets["65+"] += count

            total = sum(brackets.values())
            if total <= 0:
                raise IBGEDataError("Age group distribution total is zero or negative")

            # Apply mappings to brackets if any exist
            mapped_brackets = {}
            for k, v in brackets.items():
                mapped_k = cfg.mappings.get(k, k)
                mapped_brackets[mapped_k] = mapped_brackets.get(mapped_k, 0.0) + v

            return {k: (v / total) * 100 for k, v in mapped_brackets.items()}
        except Exception as e:
            if isinstance(e, IBGEDataError):
                raise
            raise IBGEDataError(
                "Failed to fetch or parse age group distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, period, fetch
    )


def get_urban_rural_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.censo.urban_rural
    period = cfg.period

    def fetch():
        try:
            query = _build_query_from_config(cfg)
            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise UrbanRuralDistributionError(
                    "No data returned from SIDRA for urban/rural distribution"
                )

            raw = {}
            for res in data[0].resultados:
                status_name = None
                for c in res.classificacoes:
                    if c.id == "1":
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
                raise UrbanRuralDistributionError(
                    "Urban/Rural distribution total population is zero or negative"
                )

            # Apply mappings
            mapped_raw = {}
            for k, v in raw.items():
                mapped_k = cfg.mappings.get(k, k)
                mapped_raw[mapped_k] = mapped_raw.get(mapped_k, 0.0) + v

            return {k: (v / total) * 100 for k, v in mapped_raw.items()}
        except Exception as e:
            if isinstance(e, UrbanRuralDistributionError):
                raise
            raise UrbanRuralDistributionError(
                "Failed to fetch or parse urban/rural distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, period, fetch
    )


def get_religion_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.censo.religion
    period = cfg.period

    def fetch():
        try:
            query = _build_query_from_config(cfg)
            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise ReligionDistributionError(
                    "No data returned from SIDRA for religion distribution"
                )

            raw = {}
            for res in data[0].resultados:
                rel_name = None
                for c in res.classificacoes:
                    if c.id == "133":
                        for cat_name in c.categoria.values():
                            if cat_name != "Total":
                                rel_name = cat_name

                if rel_name:
                    for serie in res.series:
                        val_str = serie.serie.get(period)
                        val = safe_float(val_str)
                        if val is not None:
                            raw[rel_name] = val

            mapped = {
                "Católica": 0.0,
                "Evangélica": 0.0,
                "Espírita": 0.0,
                "Umbanda e Candomblé": 0.0,
                "Sem religião": 0.0,
                "Outras": 0.0,
            }

            for name, val in raw.items():
                # Check mapping via dict or fallback search
                mapped_name = cfg.mappings.get(name)
                if mapped_name:
                    mapped[mapped_name] += val
                elif "Católica Apostólica Romana" in name:
                    mapped["Católica"] += val
                elif "Evangélica" in name or name == "Evangélicas":
                    mapped["Evangélica"] += val
                elif "Espírita" in name:
                    mapped["Espírita"] += val
                elif "Umbanda" in name or "Candomblé" in name:
                    mapped["Umbanda e Candomblé"] += val
                elif "Sem religião" in name:
                    mapped["Sem religião"] += val
                else:
                    mapped["Outras"] += val

            total = sum(mapped.values())
            if total <= 0:
                raise ReligionDistributionError(
                    "Religion distribution total is zero or negative"
                )
            return {k: (v / total) * 100 for k, v in mapped.items()}
        except Exception as e:
            if isinstance(e, ReligionDistributionError):
                raise
            raise ReligionDistributionError(
                "Failed to fetch or parse religion distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, period, fetch
    )
