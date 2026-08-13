import json
import os
import sqlite3
from pathlib import Path
from typing import Any, cast

from distill_lab.canonical import canonical_json


class CacheIntegrityError(RuntimeError):
    pass


class SQLiteResultCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        os.chmod(path, 0o600)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            "namespace TEXT NOT NULL, cache_key TEXT NOT NULL, payload BLOB NOT NULL, "
            "PRIMARY KEY(namespace, cache_key))"
        )
        self._connection.commit()

    def get(self, namespace: str, cache_key: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload FROM results WHERE namespace = ? AND cache_key = ?",
            (namespace, cache_key),
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise CacheIntegrityError("cached result is not an object")
        return cast(dict[str, Any], value)

    def put(self, namespace: str, cache_key: str, value: dict[str, Any]) -> None:
        payload = canonical_json(value)
        self._connection.execute(
            "INSERT OR IGNORE INTO results(namespace, cache_key, payload) VALUES (?, ?, ?)",
            (namespace, cache_key, payload),
        )
        self._connection.commit()
        row = self._connection.execute(
            "SELECT payload FROM results WHERE namespace = ? AND cache_key = ?",
            (namespace, cache_key),
        ).fetchone()
        if row is None or row[0] != payload:
            raise CacheIntegrityError("cache key collision produced different content")

    def close(self) -> None:
        self._connection.close()
