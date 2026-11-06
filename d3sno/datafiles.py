"""Where the package data tables live, and the stage-10 row updates (``s10_*.tsv``).

``20_corrections9.md`` ships row updates for five tables. They are applied at load time by key:
a row with the same key replaces the base row in place, a new key is added at the end. The
shipped base tables stay as they were delivered.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"

# base table -> (update table, key columns)
UPDATES = {
    "ps3_attributes.tsv": ("s10_ps3_attributes.tsv", ("ps3_id",)),
    "cov_enums_core.tsv": ("s10_enums_core.tsv", ("enum", "value")),
    "cov_enums_geometry.tsv": ("s10_enums_geometry.tsv", ("record", "field", "kind", "value")),
    "cov_enums_records.tsv": ("s10_enums_records.tsv", ("record", "field", "kind", "value")),
    "cov_tags.tsv": ("s10_tags.tsv", ("table", "tag_hex")),
}


def _read(name: str) -> list[dict[str, str]]:
    with open(DATA / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _key(row: dict[str, str], columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row[c].strip().lower() if c == "tag_hex" else row[c].strip() for c in columns)


@lru_cache(maxsize=None)
def table_rows(name: str) -> tuple[dict[str, str], ...]:
    """The rows of a data table with its ``s10`` update applied (other tables: as shipped)."""
    rows = _read(name)
    if name not in UPDATES:
        return tuple(rows)
    update, columns = UPDATES[name]
    where = {_key(row, columns): i for i, row in enumerate(rows)}
    for row in _read(update):
        key = _key(row, columns)
        if key in where:
            rows[where[key]] = row
        else:
            where[key] = len(rows)
            rows.append(row)
    return tuple(rows)
