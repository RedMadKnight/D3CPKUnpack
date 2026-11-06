"""x360 struct fields (``x360_struct_fields.tsv``) with the stage-10 corrections (G2-S3, G2-S4).

These tables are data: no reader takes its layout from them (Q292). They give, per x360 struct,
each field's offset, type, Switch counterpart and status.

- G2-S3: the ``ExperienceLevel`` bounty block (body offsets 120 to 152) is renamed by
  ``BOUNTY_BLOCK``; ``BountyCampsXP`` and ``BountyCampsGold`` become the Switch fields without an
  x360 counterpart (ABSENT), replacing ``GoblinHuntBonusGold`` and ``BountyKillUniqueXP``.
- G2-S4: ``x360_struct_names.tsv`` proposes a display name for fields that are OPEN. The
  executable holds no field names, so these are display names only; fields are still read by
  offset and type, and ``X360Field.name`` is never replaced by a proposal.
- V9-S4: several rows share one offset (the ``ScriptEventVariantParam`` variants at 0, 4 and 8,
  two bit fields of the ``DevilsHandRedeemableSet`` word at 28). A proposal is looked up by struct
  and field index (``proposal``), or by offset and name; by offset alone only when one row has
  that offset (``proposals_at`` gives them all).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from .datafiles import table_rows

# G2-S3: body offset -> name (SUPPORTED).
BOUNTY_BLOCK = {
    120: "GoblinHuntBonusGold",
    124: "BountyKillUniqueXP",
    128: "BountyKillBossXP",
    132: "BountyCompleteEventXP",
    136: "BountyClearDungeonXP",
    140: "BountyKillUniqueGold",
    144: "BountyKillBossGold",
    148: "BountyCompleteEventGold",
    152: "BountyClearDungeonGold",
}
BOUNTY_ABSENT = ("BountyCampsXP", "BountyCampsGold")
_BOUNTY_EVIDENCE = "G2-S3: reward ratios inside the x360 row follow the Switch pattern (x360_xp_bounty.tsv)"


def _int(text: str) -> int | None:
    text = (text or "").strip()
    return None if text in ("", "-") else int(text, 0)


@dataclass(frozen=True)
class X360Field:
    struct: str
    x360_size: int
    switch_struct: str
    switch_size: int | None
    field_index: int | None  # None: a Switch field with no x360 counterpart (ABSENT)
    x360_offset: int | None
    x360_bytes: int | None
    type: str
    element: str
    count: str
    extra: str
    name: str
    switch_offset: int | None
    switch_type: str
    status: str
    evidence: str


def _rows() -> list[X360Field]:
    return [
        X360Field(
            row["struct"], int(row["x360_size"]), row["switch_struct"], _int(row["switch_size"]),
            _int(row["field_index"]), _int(row["x360_offset"]), _int(row["x360_bytes"]),
            row["type"], row["element"], row["count"], row["extra"], row["name"],
            _int(row["switch_offset"]), row["switch_type"], row["status"], row["evidence"],
        )
        for row in table_rows("x360_struct_fields.tsv")
    ]


def _correct_bounty(fields: list[X360Field]) -> list[X360Field]:
    switch = {f.name: (f.switch_offset, f.switch_type) for f in fields}
    template = next(f for f in fields if f.field_index is None)
    result = []
    for f in fields:
        if f.field_index is None and f.name in ("GoblinHuntBonusGold", "BountyKillUniqueXP"):
            continue  # now present on x360
        if f.x360_offset in BOUNTY_BLOCK and f.field_index is not None:
            name = BOUNTY_BLOCK[f.x360_offset]
            offset, kind = switch.get(name, (None, ""))
            f = replace(f, name=name, switch_offset=offset, switch_type=kind or f.switch_type,
                        status="SUPPORTED", evidence=_BOUNTY_EVIDENCE)
        result.append(f)
    for name in BOUNTY_ABSENT:
        offset, kind = switch[name]
        result.append(replace(template, name=name, switch_offset=offset, switch_type=kind,
                              status="ABSENT", evidence="G2-S3: Switch field with no x360 counterpart"))
    return result


@lru_cache(maxsize=None)
def x360_struct_fields() -> dict[str, tuple[X360Field, ...]]:
    """Every x360 struct's fields, in table order, with G2-S3 applied."""
    found: dict[str, list[X360Field]] = {}
    for f in _rows():
        found.setdefault(f.struct, []).append(f)
    found["ExperienceLevel"] = _correct_bounty(found["ExperienceLevel"])
    return {struct: tuple(fields) for struct, fields in found.items()}


def x360_field(struct: str, offset: int) -> X360Field | None:
    """The first field of ``struct`` at ``offset`` (variants share offsets: see
    ``x360_field_by_index``)."""
    for f in x360_struct_fields().get(struct, ()):
        if f.x360_offset == offset:
            return f
    return None


def x360_field_by_index(struct: str, field_index: int) -> X360Field | None:
    for f in x360_struct_fields().get(struct, ()):
        if f.field_index == field_index:
            return f
    return None


@dataclass(frozen=True)
class X360FieldName:
    """A proposed display name (G2-S4)."""

    struct: str
    field_index: int | None
    x360_offset: int | None
    type: str
    proposed_name: str
    role: str
    evidence: str
    status: str  # SUPPORTED or OPEN


@lru_cache(maxsize=None)
def x360_struct_names() -> tuple[X360FieldName, ...]:
    return tuple(
        X360FieldName(row["struct"], _int(row["field_index"]), _int(row["x360_offset"]), row["type"],
                      row["proposed_name"], row["role"], row["evidence"], row["status"])
        for row in table_rows("x360_struct_names.tsv")
    )


@lru_cache(maxsize=None)
def _names_by_index() -> dict[tuple[str, int], X360FieldName]:
    return {(r.struct, r.field_index): r for r in x360_struct_names() if r.field_index is not None}


def proposal(struct: str, field_index: int) -> X360FieldName | None:
    """V9-S4: the proposed name of a field, by struct and field index."""
    return _names_by_index().get((struct, field_index))


def proposals_at(struct: str, offset: int) -> list[X360FieldName]:
    """Every proposal at an offset (several for variants and bit fields)."""
    return [r for r in x360_struct_names() if r.struct == struct and r.x360_offset == offset]


def display_name(struct: str, offset: int | None = None, *, field_index: int | None = None,
                 name: str | None = None) -> str | None:
    """A field's display name: the proposed name of G2-S4 when there is one, else the table name.
    For display only; never a constant.

    V9-S4: pass ``field_index`` (preferred), or ``offset`` with ``name`` (the proposed name or
    the table name) to pick one of several rows at an offset. With ``offset`` alone the proposal
    is used only when it is the only row at that offset; otherwise the table name is given."""
    if field_index is not None:
        found = proposal(struct, field_index)
        if found is not None:
            return found.proposed_name
        field_ = x360_field_by_index(struct, field_index)
        return field_.name if field_ else None
    if offset is None:
        raise ValueError("display_name needs an offset or a field index")
    rows = proposals_at(struct, offset)
    if name is not None:
        for row in rows:
            table = x360_field_by_index(struct, row.field_index) if row.field_index is not None else None
            if name in (row.proposed_name, table.name if table else None):
                return row.proposed_name
        rows = []
    if len(rows) == 1:
        return rows[0].proposed_name
    found = x360_field(struct, offset)
    return found.name if found else None
