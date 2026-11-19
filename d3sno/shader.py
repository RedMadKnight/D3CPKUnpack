"""Shader containers, compiled program blocks, program replacement and shader maps
(``30`` W15-A13, W15-A14).

``.shd`` and ``.shm`` are read as generic trees (``sno_layout``) and laid out again with
``sno_layout.write_tree``. Body (record) offsets, ps3 / switch and x360:

- ``.shd``: flag word 0x0C; pass count 0x10 / 0x14; passes pair 0x14 / 0x18 (200 / 224-byte
  passes); compiled block pair 0x20 / 0x28, the block ending the file.
- pass: stage count 0x6C (= stage list length); stage list pair 0x90 / 0x70 (24-byte stages:
  slot type, three address modes, filter, float mip bias); tag map pair 0x74 / 0x84; vertex and
  pixel key 0xC0, 0xC4 / 0xD8, 0xDC; switch and x360 two always-empty byte arrays 0xB8, 0xC8.
- block: 2 x passes pairs {offset, size} relative to the table end, contiguous from 0 to the
  block end; record 2k is the vertex and 2k+1 the pixel program of pass k. Heads: ps3 code
  length at +0x10, code at the record end; switch {0, 0, size - 16, 0} then a program whose word
  +0x44 is its own size and +0x48 is 0x2D0; x360 {h, n, a, b} (``31`` V18-1: h 0 or 1, 1 only
  on vertex programs; a and b multiples of 4; size - 16 - a - b a multiple of 8). Records are
  kept as stored: a writer keeps h, and a program replacement copies the donor's heads, h
  included. h = 1 on a pixel record is listed in ``ShaderFile.problems`` (not a read error).
- ``.shm``: a tag map (body 0x10) of type-2 shader handles keyed by use.

GPU programs are not compiled. Pass keys cannot be computed: they are copied with the program.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Sequence

from . import sno_layout
from .binary import ByBuild, ByteView, FormatError, Pair, build_order
from .power import TAG_SNO, read_tag_map
from .presentation import parse_shader_block

__all__ = [
    "ShaderFile", "ShaderPass", "ShaderProgram", "TextureStage", "read_shader", "write_shader",
    "shader_records", "shader_block", "replace_shader_programs", "sampled_slot_types",
    "read_shader_map", "shader_map_links", "set_shader_map_link", "check_shader_map_links",
    "unused_pass_key", "x360_head_problems", "SWITCH_PROGRAM_MARK",
]

GROUP_SHADERS = 37
FLAGS_AT = 0x0C
_PASS_COUNT_AT = ByBuild({"ps3": 0x10, "switch": 0x14})
_PASSES_AT = ByBuild({"ps3": 0x14, "switch": 0x18})
_BLOCK_AT = ByBuild({"ps3": 0x20, "switch": 0x28})
_PASS_SIZE = ByBuild({"ps3": 200, "switch": 224})
STAGE_COUNT_AT = 0x6C
_STAGES_AT = ByBuild({"ps3": 0x90, "switch": 0x70})
_PASS_TAGS_AT = ByBuild({"ps3": 0x74, "switch": 0x84})
_KEYS_AT = ByBuild({"ps3": (0xC0, 0xC4), "switch": (0xD8, 0xDC)})
_EMPTY_ARRAYS_AT = ByBuild({"ps3": (), "switch": (0xB8, 0xC8)})
# Render parameters (switch, x360: inline RenderParams of 96 bytes at pass +0x08). ps3: not
# established, so they are not copied there.
_RENDER_PARAMS = ByBuild({"switch": (0x08, 0x68)})
STAGE_SIZE = 24
SWITCH_PROGRAM_MARK = 0x2D0
SHM_TAGS_AT = 0x10


# --- Records of the compiled block (W15-A13) -----------------------------------------------------

@dataclass
class ShaderProgram:
    pass_index: int
    stage: str  # vertex or pixel
    offset: int  # relative to the table end
    size: int
    head: tuple  # switch / x360: the four head words; ps3: the code length
    code: bytes
    raw: bytes


def _pass_count(passes) -> int:
    return passes if isinstance(passes, int) else len(passes)


def shader_records(block: bytes, passes, build: str, strict: bool = True) -> list[ShaderProgram]:
    """W15-A13: the 2 x passes program records of a compiled block (x360 heads {h, n, a, b}
    included, h 0 or 1: V18-1; see ``x360_head_problems`` for h = 1 on a pixel record).
    ``passes`` is the pass count or the pass list. With ``strict`` the records must be
    contiguous from 0 to the block end, and a switch program must hold its own size at +0x44
    and 0x2D0 at +0x48 (after the 16-byte head)."""
    count = _pass_count(passes)
    order = build_order(build)
    pairs = parse_shader_block(bytes(block), count, build)
    view = ByteView(bytes(block), order)
    table_end = count * 16
    cursor = 0
    found = []
    for i in range(count * 2):
        offset, size = view.unpack("II", i * 8)
        if strict and offset != cursor:
            raise FormatError(f"shader record {i} at +0x{offset:X}, expected +0x{cursor:X}", i * 8)
        cursor = offset + size
        record = pairs[i // 2][i % 2]
        r = ByteView(record.raw, order)
        if build == "ps3":
            head = (r.u32(0x10),)
        else:
            head = r.unpack("4I", 0)
            if strict and build == "switch":
                program = size - 16
                words = r.unpack("II", 16 + 0x44, "switch program words") if size >= 16 + 0x4C else None
                if words != (program, SWITCH_PROGRAM_MARK):
                    raise FormatError(f"switch program {i}: words +0x44/+0x48 {words}, expected "
                                      f"({program}, 0x{SWITCH_PROGRAM_MARK:X})", table_end + offset)
        found.append(ShaderProgram(i // 2, "vertex" if i % 2 == 0 else "pixel", offset, size, head,
                                   record.code, record.raw))
    if strict and table_end + cursor != len(block):
        raise FormatError(f"shader records end at +0x{table_end + cursor:X}, the block at "
                          f"+0x{len(block):X}", 0)
    return found


def x360_head_problems(programs: Sequence[ShaderProgram]) -> list[str]:
    """V18-1: h = 1 is seen only on vertex-program records; a pixel record with h = 1 is
    reported (the record is still read and written as stored)."""
    return [f"pass {p.pass_index} pixel program: head word h = {p.head[0]} (seen only on vertex "
            f"programs, V18-1)" for p in programs if p.stage == "pixel" and p.head and p.head[0] != 0
            and len(p.head) == 4]


def shader_block(records: Sequence[bytes], build: str) -> bytes:
    """Lay out a compiled block from 2 x passes record bytes (vertex, pixel per pass). Records
    are copied as given (x360: head word h included, V18-1)."""
    if len(records) % 2:
        raise FormatError(f"{len(records)} records; a block holds two per pass")
    order = build_order(build)
    table = bytearray()
    cursor = 0
    for raw in records:
        table += struct.pack(order + "II", cursor, len(raw))
        cursor += len(raw)
    return bytes(table) + b"".join(bytes(r) for r in records)


# --- Container (W15-A13) -------------------------------------------------------------------------

@dataclass
class TextureStage:
    slot_type: int
    address_modes: tuple[int, int, int]
    filter: int
    mip_bias: float


@dataclass
class ShaderPass:
    index: int
    record: sno_layout.Record
    stage_count: int
    stages: list[TextureStage]
    tag_map: bytes
    vertex_key: int
    pixel_key: int


@dataclass
class ShaderFile:
    build: str
    tree: sno_layout.SnoTree
    flags: int
    pass_count: int
    passes: list[ShaderPass]
    block: bytes
    programs: list[ShaderProgram]
    problems: list[str] = field(default_factory=list)


def _block_data(record: sno_layout.Record, offset: int) -> bytes:
    value = record.pairs.get(offset)
    if value is None or value.block is None:
        return b""
    if value.block.elements is not None:
        return b"".join(bytes(e.raw) for e in value.block.elements) + value.block.extra
    return bytes(value.block.data or b"")


def _pass_records(tree: sno_layout.SnoTree) -> list[sno_layout.Record]:
    value = tree.top.pairs.get(_PASSES_AT[tree.build])
    if value is None or value.block is None:
        return []
    return list(value.block.elements or [])


def _read_pass(record: sno_layout.Record, index: int, build: str, order: str) -> tuple[ShaderPass, list[str]]:
    problems = []
    stages_raw = _block_data(record, _STAGES_AT[build])
    if len(stages_raw) % STAGE_SIZE:
        problems.append(f"pass {index}: stage list of {len(stages_raw)} bytes")
    stages = []
    for k in range(len(stages_raw) // STAGE_SIZE):
        words = struct.unpack_from(order + "5If", stages_raw, k * STAGE_SIZE)
        stages.append(TextureStage(words[0], tuple(words[1:4]), words[4], words[5]))
    stage_count = struct.unpack_from(order + "I", record.raw, STAGE_COUNT_AT)[0]
    if stage_count != len(stages):
        problems.append(f"pass {index}: stage count {stage_count}, {len(stages)} stages")
    for at in _EMPTY_ARRAYS_AT[build]:
        if _block_data(record, at):
            problems.append(f"pass {index}: byte array at +0x{at:X} is not empty")
    vertex_at, pixel_at = _KEYS_AT[build]
    shader_pass = ShaderPass(
        index, record, stage_count, stages, _block_data(record, _PASS_TAGS_AT[build]),
        struct.unpack_from(order + "I", record.raw, vertex_at)[0],
        struct.unpack_from(order + "I", record.raw, pixel_at)[0],
    )
    return shader_pass, problems


def _from_tree(tree: sno_layout.SnoTree, source_length: int | None = None) -> ShaderFile:
    build = tree.build
    order = build_order(build)
    top = tree.top
    pass_count = struct.unpack_from(order + "I", top.raw, _PASS_COUNT_AT[build])[0]
    problems = list(tree.notes)
    passes = []
    for i, record in enumerate(_pass_records(tree)):
        shader_pass, found = _read_pass(record, i, build, order)
        passes.append(shader_pass)
        problems += found
    if pass_count != len(passes):
        problems.append(f"pass count {pass_count}, {len(passes)} passes")
    block_value = top.pairs.get(_BLOCK_AT[build])
    block = _block_data(top, _BLOCK_AT[build])
    if source_length is not None and block:
        end = sno_layout.BODY + block_value.stored[0] + block_value.stored[1]
        if end != source_length:
            problems.append(f"the compiled block ends at 0x{end:X}, the file at 0x{source_length:X}")
    programs = shader_records(block, len(passes), build) if block else []
    if build == "x360":
        problems += x360_head_problems(programs)
    flags = struct.unpack_from(order + "I", top.raw, FLAGS_AT)[0]
    return ShaderFile(build, tree, flags, pass_count, passes, block, programs, problems)


def read_shader(file: bytes, build: str | None = None) -> ShaderFile:
    """W15-A13: read a ``.shd``: passes (stage lists, tag maps, keys) and the compiled block's
    program records. Departures that do not stop reading are listed in ``problems``."""
    tree = sno_layout.parse_tree(bytes(file), "shd", build)
    return _from_tree(tree, len(file))


def write_shader(shader: ShaderFile) -> bytes:
    """Lay the file out again (W11); an unchanged shader gives its bytes back."""
    return sno_layout.write_tree(shader.tree)


def sampled_slot_types(shader: ShaderFile) -> set[int]:
    """The texture slot types the shader's passes sample (their stage lists)."""
    return {stage.slot_type for p in shader.passes for stage in p.stages}


# --- Replacement (W15-A14) -----------------------------------------------------------------------

def _copy_pair(value: sno_layout.PairValue | None) -> sno_layout.PairValue | None:
    if value is None or value.block is None:
        return None
    block = value.block
    if block.elements is not None:
        elements = [sno_layout.Record(e.name, bytearray(e.raw)) for e in block.elements]
        return sno_layout.PairValue(sno_layout.Block(elements=elements, extra=block.extra))
    return sno_layout.PairValue(sno_layout.Block(bytes(block.data or b"")))


def replace_shader_programs(shader: ShaderFile, donor: ShaderFile, copy_stages: bool = False,
                            copy_render_params: bool = False) -> bytes:
    """W15-A14: give ``shader`` the whole compiled block of ``donor`` (same build and pass count)
    with the donor's vertex and pixel pass keys, and lay the file out again; returns the bytes
    and updates ``shader``. Nothing else is derived. The donor's records are copied whole, so
    their heads come with them (x360: h included, V18-1).

    ``copy_stages`` also copies each pass's stage list and stage count, ``copy_render_params``
    the render parameters (switch, x360: pass +0x08..+0x68); both are advised by W15-A14 and off
    by default, so every other byte stays as it was."""
    if shader.build != donor.build:
        raise FormatError(f"donor is {donor.build}, the shader {shader.build}: keys are build specific")
    if len(shader.passes) != len(donor.passes):
        raise FormatError(f"donor has {len(donor.passes)} passes, the shader {len(shader.passes)}")
    if not donor.block:
        raise FormatError("the donor holds no compiled block")
    build = shader.build
    if copy_render_params and build == "ps3":
        raise FormatError("ps3 render parameters are not established")
    for mine, theirs in zip(shader.passes, donor.passes):
        for at in _KEYS_AT[build]:
            mine.record.raw[at:at + 4] = theirs.record.raw[at:at + 4]
        if copy_stages:
            mine.record.pairs[_STAGES_AT[build]] = _copy_pair(theirs.record.pairs.get(_STAGES_AT[build]))
            mine.record.raw[STAGE_COUNT_AT:STAGE_COUNT_AT + 4] = \
                theirs.record.raw[STAGE_COUNT_AT:STAGE_COUNT_AT + 4]
        if copy_render_params:
            start, end = _RENDER_PARAMS[build]
            mine.record.raw[start:end] = theirs.record.raw[start:end]
    shader.tree.top.pairs[_BLOCK_AT[build]] = sno_layout.PairValue(sno_layout.Block(bytes(donor.block)))
    out = sno_layout.write_tree(shader.tree)
    fresh = read_shader(out, build)
    shader.__dict__.update(fresh.__dict__)
    return out


def unused_pass_key(used: Sequence[int] | set[int], start: int = 1) -> int:
    """W15-A14: a key that no program of the build uses (a new program needs one). ``used`` are
    the build's vertex and pixel keys, for example from ``read_shader`` over every ``.shd``."""
    taken = set(used)
    key = start
    while key in taken:
        key += 1
    if key > 0xFFFFFFFF:
        raise FormatError("no free pass key")
    return key


# --- Shader maps (W15-A14) -----------------------------------------------------------------------

def read_shader_map(data: bytes, build: str | None = None) -> sno_layout.SnoTree:
    return sno_layout.parse_tree(bytes(data), "shm", build)


def _shm_records(shm: sno_layout.SnoTree):
    if shm.ext != "shm":
        raise FormatError(f"not a shader map tree (.{shm.ext})")
    raw = _block_data(shm.top, SHM_TAGS_AT)
    if not raw:
        return raw, []
    view = ByteView(bytes(0x10) + raw, build_order(shm.build))
    return raw, read_tag_map(view, Pair(0, len(raw)), "shader map", 0x10, type_7=True).records


def shader_map_links(shm: sno_layout.SnoTree) -> dict[int, int]:
    """Use tag -> shader handle of every type-2 record."""
    return {r.tag: r.value for r in _shm_records(shm)[1] if r.type == TAG_SNO}


def set_shader_map_link(shm: sno_layout.SnoTree, use_tag: int, handle: int) -> int:
    """W15-A14: change a use by overwriting its type-2 handle in place; returns the old one."""
    raw, records = _shm_records(shm)
    for record in records:
        if record.tag == use_tag:
            if record.type != TAG_SNO:
                raise FormatError(f"use 0x{use_tag:X} is a type-{record.type} record, not a handle")
            data = bytearray(raw)
            struct.pack_into(build_order(shm.build) + "i", data, record.offset - 0x10 + 8, handle)
            shm.top.pairs[SHM_TAGS_AT] = sno_layout.PairValue(sno_layout.Block(bytes(data)))
            return record.value
    raise FormatError(f"the shader map has no use 0x{use_tag:X}")


def check_shader_map_links(shm: sno_layout.SnoTree, toc) -> list[str]:
    """W15-A14: every shader-map link must name a shader (group 37) of the TOC (-1: none).
    ``toc`` as for ``material_write.toc_keys``; a ``CoreToc`` or a prepared set of keys is not
    copied (V18-9)."""
    from .material_write import toc_keys

    keys = toc_keys(toc)
    return [f"use 0x{tag:X}: shader {handle} is not in the TOC"
            for tag, handle in shader_map_links(shm).items()
            if handle != -1 and (GROUP_SHADERS, handle) not in keys]
