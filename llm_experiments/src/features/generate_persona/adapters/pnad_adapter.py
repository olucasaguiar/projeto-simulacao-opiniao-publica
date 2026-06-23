from infrastructure.cache import DistributionCache
from infrastructure.sidra import SidraClient, SidraQueryBuilder
from settings import settings
from ..exceptions import (
    EmploymentDistributionError,
    EducationDistributionError,
    IncomeDistributionError,
)
from .utils import get_cached_or_fetch, safe_float


def get_employment_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.pnad.employment

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
                raise EmploymentDistributionError(
                    "No data returned from SIDRA for employment distribution"
                )

            res_0 = data[0].resultados[0]
            actual_period = next(iter(res_0.series[0].serie))

            raw = {}
            for res in data[0].resultados:
                emp_name = None
                for c in res.classificacoes:
                    if c.id == str(cfg.classifications[0].id):
                        for cat_name in c.categoria.values():
                            if cat_name != "Total":
                                for map_key, map_val in cfg.mappings.items():
                                    if map_key in cat_name.lower():
                                        emp_name = map_val
                                        break
                                else:
                                    emp_name = cat_name

                if emp_name:
                    for serie in res.series:
                        val_str = serie.serie.get(actual_period)
                        val = safe_float(val_str)
                        if val is not None:
                            raw[emp_name] = val

            total = sum(raw.values())
            if total <= 0:
                raise EmploymentDistributionError(
                    "Employment distribution total population is zero or negative"
                )
            return {k: (v / total) * 100 for k, v in raw.items()}, actual_period
        except Exception as e:
            if isinstance(e, EmploymentDistributionError):
                raise
            raise EmploymentDistributionError(
                "Failed to fetch or parse employment distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, cfg.period, fetch
    )


def get_education_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.pnad.education

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
                raise EducationDistributionError(
                    "No data returned from SIDRA for education distribution"
                )

            res_0 = data[0].resultados[0]
            actual_period = next(iter(res_0.series[0].serie))

            raw = {}
            for res in data[0].resultados:
                edu_name = None
                for c in res.classificacoes:
                    if c.id == str(cfg.classifications[0].id):
                        for cat_name in c.categoria.values():
                            if cat_name != "Total" and cat_name != "Não determinado":
                                edu_name = cat_name

                if edu_name:
                    for serie in res.series:
                        val_str = serie.serie.get(actual_period)
                        val = safe_float(val_str)
                        if val is not None:
                            raw[edu_name] = val

            total = sum(raw.values())
            if total <= 0:
                raise EducationDistributionError(
                    "Education distribution total population is zero or negative"
                )
            return {k: (v / total) * 100 for k, v in raw.items()}, actual_period
        except Exception as e:
            if isinstance(e, EducationDistributionError):
                raise
            raise EducationDistributionError(
                "Failed to fetch or parse education distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, cfg.period, fetch
    )


def get_income_distribution(
    client: SidraClient, cache: DistributionCache
) -> dict[str, float]:
    cfg = settings.pnad.income

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
                raise IncomeDistributionError(
                    "No data returned from SIDRA for income distribution"
                )

            raw = {}
            for res in data[0].resultados:
                income_bracket = None
                for c in res.classificacoes:
                    if c.id == str(cfg.classifications[0].id):
                        for cat_name in c.categoria.values():
                            if cat_name != "Total":
                                income_bracket = cat_name

                if income_bracket:
                    for serie in res.series:
                        val_str = serie.serie.get(cfg.period)
                        val = safe_float(val_str)
                        if val is not None:
                            raw[income_bracket] = val

            total = sum(raw.values())
            if total <= 0:
                raise IncomeDistributionError(
                    "Income distribution total is zero or negative"
                )
            return {k: (v / total) * 100 for k, v in raw.items()}
        except Exception as e:
            if isinstance(e, IncomeDistributionError):
                raise
            raise IncomeDistributionError(
                "Failed to fetch or parse income distribution"
            ) from e

    return get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, cfg.period, fetch
    )
