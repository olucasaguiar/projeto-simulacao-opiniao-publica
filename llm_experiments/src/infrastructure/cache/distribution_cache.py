import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


from settings import settings


@dataclass(frozen=True)
class CacheEntry:
    source: str
    table_id: int
    fetched_at: datetime
    ttl_days: int
    period: str
    distribution: dict[str, float]

    @property
    def is_expired(self) -> bool:
        fetched = self.fetched_at
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        expires_at = fetched + timedelta(days=self.ttl_days)
        return datetime.now(timezone.utc) > expires_at


class DistributionCache:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {settings.cache.table_name} (
                key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                table_id INTEGER NOT NULL,
                fetched_at TEXT NOT NULL,
                ttl_days INTEGER NOT NULL,
                period TEXT NOT NULL,
                distribution TEXT NOT NULL
            )
        """)

    def get(self, key: str) -> CacheEntry | None:
        row = self._conn.execute(
            f"SELECT source, table_id, fetched_at, ttl_days, period, distribution "
            f"FROM {settings.cache.table_name} WHERE key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        entry = CacheEntry(
            source=row[0],
            table_id=row[1],
            fetched_at=datetime.fromisoformat(row[2]),
            ttl_days=row[3],
            period=row[4],
            distribution=json.loads(row[5]),
        )
        return None if entry.is_expired else entry

    def put(self, key: str, entry: CacheEntry) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO {settings.cache.table_name} VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                entry.source,
                entry.table_id,
                entry.fetched_at.isoformat(),
                entry.ttl_days,
                entry.period,
                json.dumps(entry.distribution, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
