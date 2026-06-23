from datetime import datetime, timezone
from infrastructure.cache import DistributionCache, CacheEntry
from settings import settings


def get_cached_or_fetch(
    cache: DistributionCache,
    key: str,
    table_id: int,
    ttl_days: int,
    period: str,
    fetch_fn,
) -> dict[str, float]:
    entry = cache.get(key)
    if entry:
        return entry.distribution

    res = fetch_fn()
    if isinstance(res, tuple):
        dist, resolved_period = res
    else:
        dist, resolved_period = res, period

    new_entry = CacheEntry(
        source=settings.cache.default_source,
        table_id=table_id,
        fetched_at=datetime.now(timezone.utc),
        ttl_days=ttl_days,
        period=resolved_period,
        distribution=dist,
    )
    cache.put(key, new_entry)
    return dist


def safe_float(val_str: str) -> float | None:
    if not val_str:
        return None
    try:
        return float(val_str)
    except ValueError:
        return None
