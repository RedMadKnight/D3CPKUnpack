"""``.mat`` materials and the material path of an ``.app``."""

from __future__ import annotations

from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair
from .sno import SnoHeader, parse_header

SLOT_DIFFUSE = 1
SLOT_LIGHTMAP = 2
SLOT_NAMES = {
    1: "diffuse", 2: "lightmap", 3: "bump", 4: "environment", 5: "gloss", 6: "glow",
    7: "reflection", 8: "irradiance", 9: "refraction", 10: "distortion", 19: "alpha",
}
_BLOCK_SIZE = {"switch": 104, "ps3": 72}
_FIXED = {"switch": 0x88, "ps3": 0x68}
_TEXTURE_LIST_AT = {"switch": 0x4C, "ps3": 0x2C}
_TEXTURE_SIZE = {"switch": 32, "ps3": 24}
_TEXTURE_FIELDS = {"switch": (0x08, 0x18), "ps3": (0x04, 0x14)}  # animation pair, handle
_BLOCK_AT = 0x20

# File positions in an .app: look count, material count, materials pair, looks pair.
_APP_FIELDS = {"switch": (0x178, 0x17C, 0x180, 0x190), "ps3": (0x188, 0x18C, 0x190, 0x1A0)}
_MATERIAL_RECORD = {"switch": 24, "ps3": 20}
_LOOK_RECORD = {"switch": 248, "ps3": 184}
_LOOK_BLOCK_AT = 0x18
_LOOK_HANDLE_AT = {"switch": 0x80, "ps3": 0x60}


@dataclass(frozen=True)
class Colour:
    rgba: tuple  # 4 floats on switch, 4 bytes on ps3
    scale: float | None  # ps3


@dataclass(frozen=True)
class TextureSlot:
    type: int
    animation_raw: bytes
    handle: int

    @property
    def name(self) -> str | None:
        return SLOT_NAMES.get(self.type)


@dataclass
class MaterialBlock:
    shader_map: int
    ambient: Colour
    diffuse: Colour
    specular: Colour
    emissive: Colour
    power: float
    sort_priority_raw: int
    textures: list[TextureSlot]

    def texture(self, slot_type: int) -> int | None:
        return next((t.handle for t in self.textures if t.type == slot_type), None)


def read_material_block(view: ByteView, position: int, build: str) -> MaterialBlock:
    """Read the block at file ``position``."""
    size = _BLOCK_SIZE[build]
    raw = view.take(position, size, "material block")
    r = ByteView(raw, view.order)
    if build == "switch":
        colours = [Colour(r.f32s(0x04 + i * 16, 4), None) for i in range(4)]
        tail = 0x44
    else:
        colours = [
            Colour(tuple(raw[0x04 + i * 8:0x08 + i * 8]), r.f32(0x08 + i * 8)) for i in range(4)
        ]
        tail = 0x24
    list_at = _TEXTURE_LIST_AT[build]
    pair = r.pair(list_at)
    width = _TEXTURE_SIZE[build]
    if pair.size % width:
        raise FormatError(f"texture list of {pair.size} bytes", position + list_at)
    view.pair_data(pair, position + list_at, "texture list")
    animation_at, handle_at = _TEXTURE_FIELDS[build]
    textures = []
    for i in range(pair.size // width):
        at = pair.file_offset + i * width
        animation = view.pair_data(view.pair(at + animation_at), at + animation_at, "texture animation")
        textures.append(TextureSlot(view.u32(at), animation, view.i32(at + handle_at)))
    return MaterialBlock(
        r.i32(0), colours[0], colours[1], colours[2], colours[3], r.f32(tail), r.u32(tail + 4),
        textures,
    )


@dataclass
class MaterialFile:
    header: SnoHeader
    build: str
    flags: int
    tag_map_raw: bytes
    block: MaterialBlock


def parse_material(data: bytes, build: str | None = None) -> MaterialFile:
    header = parse_header(data, build)
    view = ByteView(data, header.order)
    view.need(0, 0x10 + _FIXED[header.build], "material fixed part")
    tags = view.pair_data(view.pair(0x20), 0x20, "tag map")
    return MaterialFile(
        header, header.build, view.u32(0x1C), tags,
        read_material_block(view, 0x10 + _BLOCK_AT, header.build),
    )


@dataclass
class AppMaterial:
    hash: int
    per_look: list[tuple[MaterialBlock, int]]  # block and material handle, per look


@dataclass
class AppMaterials:
    looks: list[int] = field(default_factory=list)  # look hashes
    materials: list[AppMaterial] = field(default_factory=list)

    def find(self, material_hash: int) -> AppMaterial | None:
        return next((m for m in self.materials if m.hash == material_hash), None)


def read_app_materials(view: ByteView, build: str) -> AppMaterials:
    """Read the looks and materials of an ``.app``."""
    look_count_at, material_count_at, materials_at, looks_at = _APP_FIELDS[build]
    look_count = view.i32(look_count_at, "look count")
    material_count = view.i32(material_count_at, "material count")
    looks_pair = view.pair(looks_at)
    if look_count < 0 or looks_pair.size != look_count * 4:
        raise FormatError(f"looks of {looks_pair.size} bytes for {look_count}", looks_at)
    looks_raw = view.pair_data(looks_pair, looks_at, "looks")
    record = _MATERIAL_RECORD[build]
    pair = view.pair(materials_at)
    if material_count < 0 or pair.size != material_count * record:
        raise FormatError(f"materials of {pair.size} bytes for {material_count}", materials_at)
    view.pair_data(pair, materials_at, "materials")
    result = AppMaterials(list(ByteView(looks_raw, view.order).unpack(f"{look_count}I", 0)))
    look_size = _LOOK_RECORD[build]
    for i in range(material_count):
        at = pair.file_offset + i * record
        list_pair = view.pair(at + 8)
        if list_pair.size % look_size:
            raise FormatError(f"per-look list of {list_pair.size} bytes", at + 8)
        view.pair_data(list_pair, at + 8, "per-look list")
        per_look = []
        for k in range(list_pair.size // look_size):
            base = list_pair.file_offset + k * look_size
            per_look.append((
                read_material_block(view, base + _LOOK_BLOCK_AT, build),
                view.i32(base + _LOOK_HANDLE_AT[build]),
            ))
        result.materials.append(AppMaterial(view.u32(at), per_look))
    return result


def sub_object_texture(app, sub, look: int = 0, slot_type: int = SLOT_DIFFUSE) -> int | None:
    """The texture handle of a sub-object for one look, or None."""
    material = app.material_data.find(sub.material_hash)
    if material is None or not 0 <= look < len(material.per_look):
        return None
    return material.per_look[look][0].texture(slot_type)
