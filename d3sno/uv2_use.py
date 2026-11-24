"""``44`` W20-1 to W20-3, ``45`` X1 to X6, ``47`` Y6 to Y8: what a primitive's second texture coordinate
(UV2) feeds.

Package tables (``d3sno/data``) say which shader passes feed UV2 to which texture stage:
``shader_uv2.tsv`` (one row per build, pass and stage that UV2 feeds; ``build`` ``ps3`` serves ps3
and switch, ``x360`` serves x360), ``shader_known.txt`` (every ps3 and switch shader the table was
derived from) and ``shader_known_x360.txt`` (every x360 shader). Shader names compare without case.
A shader that is not in its build's list is unknown: nothing is concluded about its UV2. When the
row table or a build's list is missing, that build uses no table and every shader is unknown.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .material import SLOT_DIFFUSE, SLOT_LIGHTMAP, SLOT_NAMES

__all__ = [
    "KNOWN_TABLE", "KNOWN_TABLES", "KNOWN_TABLE_X360", "ROW_TABLE", "TABLE_BUILDS", "USES", "Uv2Row", "Uv2Rows", "Uv2Tables",
    "layer_entries", "layer_rows", "load_tables", "rows_for", "slot_name", "table_build", "tables", "uv2_use",
]

ROW_TABLE = "shader_uv2.tsv"
KNOWN_TABLE = "shader_known.txt"
KNOWN_TABLE_X360 = "shader_known_x360.txt"
TABLE_BUILDS = ("ps3", "x360")  # X6: the values of the ``build`` column
KNOWN_TABLES = {"ps3": KNOWN_TABLE, "x360": KNOWN_TABLE_X360}
DATA_DIR = Path(__file__).parent / "data"
DIVISOR_LAYER = 8192
DIVISOR_ODD = 512
# ``47`` Y7: a slot type with no name in ``10`` on which no material block has a texture: its entries
# name no slot and no texture (and nothing for ``missing_dependency``)
SLOTS_WITHOUT_TEXTURE = frozenset({39})
# W20-2, in the order the cases are tried
USES = ("lightmap", "layer", "open", "lightmap_without_texture", "unused", "unknown")
# W20-2: ``uv2_scale`` of each ``uv2_use`` (the lightmap case: ``"open"`` when the texture is missing)
UV2_SCALE = {"lightmap": "lightmap", "layer": "layer", "open": "open", "lightmap_without_texture": "open",
             "unused": "unused", "unknown": "open"}


def table_build(build: str | None) -> str:
    """X6: the table rows a build uses (switch and ps3 use ``ps3``)."""
    return "x360" if build == "x360" else "ps3"


@dataclass(frozen=True)
class Uv2Row:
    shader: str  # as the table writes it
    pass_index: int
    stage: int
    slot_type: int | None  # None: the table leaves it empty (a few x360 rows)
    divisor: int
    lightmap_v_scale: int
    build: str = "ps3"

    @property
    def key(self) -> tuple:
        return self.shader.lower(), self.pass_index, self.stage


@dataclass(frozen=True)
class Uv2Tables:
    rows: dict  # table build -> {lower-case shader name -> tuple of Uv2Row}
    known: dict  # table build -> frozenset of lower-case shader names (a build without its list: absent)

    def for_build(self, build: str | None):
        """(rows, known) of a build, or None when its list of known shaders is missing."""
        key = table_build(build)
        if key not in self.known:
            return None
        return self.rows.get(key, {}), self.known[key]


def _int_or_none(text: str | None) -> int | None:
    text = (text or "").strip()
    return int(text) if text else None


def _read_rows(path: Path) -> dict:
    out: dict[str, dict[str, list[Uv2Row]]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            build = (row.get("build") or "ps3").strip()
            item = Uv2Row(row["shader"].strip(), int(row["pass"]), int(row["stage"]), _int_or_none(row["slot_type"]),
                          int(row["divisor"]), int(row["lightmap_v_scale"]), build)
            out.setdefault(build, {}).setdefault(item.shader.lower(), []).append(item)
    return {build: {name: tuple(rows) for name, rows in names.items()} for build, names in out.items()}


def _read_known(path: Path) -> frozenset:
    with open(path, encoding="utf-8") as handle:
        return frozenset(line.strip().lower() for line in handle if line.strip())


def load_tables(folder: Path | str | None = None) -> Uv2Tables | None:
    """The tables from ``folder`` (default: the package data), or None when the row table is
    missing or no build has its list of known shaders. A build whose list is missing is left out."""
    folder = Path(folder) if folder is not None else DATA_DIR
    rows_path = folder / ROW_TABLE
    if not rows_path.is_file():
        return None
    known = {build: _read_known(folder / name) for build, name in KNOWN_TABLES.items() if (folder / name).is_file()}
    if not known:
        return None
    return Uv2Tables(_read_rows(rows_path), known)


@lru_cache(maxsize=None)
def tables() -> Uv2Tables | None:
    """The package tables, read once (None when the row table or every list is missing)."""
    return load_tables(DATA_DIR)


@dataclass(frozen=True)
class Uv2Rows:
    """W20-1: the table rows of a primitive's shaders, and whether every shader is known."""

    rows: tuple = ()
    all_known: bool = False  # at least one shader, and every shader in its build's list

    @property
    def lightmap(self) -> tuple:
        return tuple(r for r in self.rows if r.slot_type == SLOT_LIGHTMAP)

    @property
    def layer(self) -> tuple:
        """Divisor 8192 on a slot other than 2, ``lightmap_v_scale`` 1 included (X6)."""
        return tuple(r for r in self.rows if r.slot_type != SLOT_LIGHTMAP and r.divisor == DIVISOR_LAYER)

    @property
    def odd(self) -> tuple:
        return tuple(r for r in self.rows if r.divisor == DIVISOR_ODD)


def rows_for(names, table: Uv2Tables | None, build: str | None = "ps3") -> Uv2Rows:
    """``names``: the shader names of every use of the material's shader map (None for a shader
    without a name), or None when the material has no shaders (handle -1, or a shader map that does
    not resolve). ``build``: the archive build; its own table rows are looked up (X6)."""
    found = table.for_build(build) if table is not None else None
    if found is None or not names:
        return Uv2Rows()
    by_name, known = found
    rows: list[Uv2Row] = []
    all_known = True
    for name in names:
        if name is None:
            all_known = False
            continue
        key = _bare(name)
        if key not in known:
            all_known = False
        rows.extend(by_name.get(key, ()))
    return Uv2Rows(tuple(rows), all_known)


def _bare(name: str) -> str:
    """The table key of a shader name: the file name without folder and ``.shd``, lower case."""
    base = name.replace("/", "\\").rpartition("\\")[2].lower()
    return base[:-4] if base.endswith(".shd") else base


def uv2_use(rows: Uv2Rows, has_lightmap: bool) -> str:
    """W20-2: the first case that applies."""
    if has_lightmap:
        return "lightmap"
    if rows.layer:
        return "layer"
    if rows.odd:
        return "open"
    if rows.lightmap:
        return "lightmap_without_texture"
    if rows.all_known and not rows.rows:
        return "unused"
    return "unknown"


def slot_name(slot_type: int | None) -> str | None:
    """W20-3: the slot name of ``10``; a slot type it does not name is ``layer_<n>``; no slot type, or
    one of ``SLOTS_WITHOUT_TEXTURE`` (Y7): None."""
    if slot_type is None or slot_type in SLOTS_WITHOUT_TEXTURE:
        return None
    return SLOT_NAMES.get(slot_type) or f"layer_{slot_type}"


def layer_rows(rows: Uv2Rows) -> list[Uv2Row]:
    """W20-3: the layer rows and the odd rows, sorted by shader name, pass and stage, without
    duplicates."""
    seen: dict[tuple, Uv2Row] = {}
    for row in (*rows.layer, *rows.odd):
        seen.setdefault(row.key, row)
    return [seen[k] for k in sorted(seen)]


def entry_texcoord(row: Uv2Row, written: str | None, v_divide: float | None):
    """X3: (``texcoord``, ``v_divide``) of an entry. ``written``: the ``TEXCOORD_1`` the primitive has
    (``"layer"``, ``"lightmap"`` or None); ``v_divide``: S of the lightmap coordinate."""
    if written is None or row.divisor != DIVISOR_LAYER:
        return None, None
    if row.lightmap_v_scale:  # Y6: ``TEXCOORD_1`` as written (V scaled by the lightmap scale in the game)
        return 1, None
    if written == "lightmap":
        return 1, v_divide
    return 1, None


def _texture(block, slot_type):
    if block is None or slot_type is None or slot_type in SLOTS_WITHOUT_TEXTURE:
        return None
    handle = block.texture(slot_type)
    return handle if handle is not None and handle != -1 else None


def layer_entries(rows: Uv2Rows, block, texture_uri, written: str | None, v_divide: float | None = None) -> list[dict]:
    """W20-3 with X3: the material's ``uv2_layers`` entries. ``block`` is the sub-object's material
    block (or None); ``texture_uri(handle)`` names a texture; ``written`` and ``v_divide`` as in
    ``entry_texcoord``."""
    out = []
    for row in layer_rows(rows):
        handle = _texture(block, row.slot_type)
        texcoord, divide = entry_texcoord(row, written, v_divide)
        entry = {"shader": row.shader, "pass": row.pass_index, "stage": row.stage, "slot_type": row.slot_type,
                 "slot_name": slot_name(row.slot_type), "divisor": row.divisor,
                 "texture": texture_uri(handle) if handle is not None else None, "texcoord": texcoord}
        if divide is not None:
            entry["v_divide"] = divide
        if row.lightmap_v_scale:  # X6
            entry["v_scale"] = "lightmap"
        if row.slot_type == SLOT_DIFFUSE:  # X3
            entry["also_base_color"] = True
        out.append(entry)
    return out


def layer_texture_handles(rows: Uv2Rows, block) -> list[int]:
    """The texture handles that the layer entries name (for ``missing_dependency``)."""
    found = (_texture(block, r.slot_type) for r in layer_rows(rows))
    return [h for h in found if h is not None]
