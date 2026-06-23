from infrastructure.cache import DistributionCache
from infrastructure.sidra import SidraClient, SidraQueryBuilder
from settings import settings
from ..exceptions import InflationDataError
from .utils import get_cached_or_fetch


def get_monthly_inflation(client: SidraClient, cache: DistributionCache) -> float:
    cfg = settings.ipca.inflation

    def fetch():
        try:
            query = (
                SidraQueryBuilder(cfg.table_id)
                .select_periods(cfg.period)
                .select_variables(cfg.variable_id)
                .select_locations(cfg.location_level)
            )
            data = client.get_data(query)
            if not data or not data[0].resultados:
                raise InflationDataError("No inflation data returned from SIDRA")

            resultado = data[0].resultados[0]
            if not resultado.series:
                raise InflationDataError("No inflation time series returned from SIDRA")

            serie = resultado.series[0]
            period = next(iter(serie.serie), None)
            if not period:
                raise InflationDataError("No period found in inflation series data")

            val_str = serie.serie[period]
            if not val_str:
                raise InflationDataError("Inflation rate value is missing or empty")

            # We return it wrapped in a dictionary so it can be cached
            return {"rate": float(val_str)}, period
        except Exception as e:
            if isinstance(e, InflationDataError):
                raise
            raise InflationDataError(
                "Failed to fetch or parse inflation rate from SIDRA"
            ) from e

    dist = get_cached_or_fetch(
        cache, cfg.cache_key, cfg.table_id, cfg.cache_ttl_days, cfg.period, fetch
    )
    return dist["rate"]
