"""Level layout: ``.scn`` beyond navigation, ``.mrk`` marker sets and ``.wrl`` worlds."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair
from .power import TAG_FLOAT, TagMap, read_tag_map
from .sno import SnoHeader, parse_header, resolve_build
from .transform import Quaternion, Vector, quat_multiply, rotate

SCENE_FIXED = 0x218
MARKER_SET_FIXED = 0xC8
MARKER_SIZE = 88
CIRCLE_SIZE = 12
LINK_SIZE = 8
LABEL_ALWAYS = -1
SCALE_TAG = 0x80000
SCENE_GROUP = 33
TYPE_ACTOR = 0
TYPE_SUB_SCENE = 16
ACTOR_SCALE_FLOOR = 1e-6
_WORLD_FIXED = {"switch": 0xC0, "ps3": 0x100}
_BLOCK_SIZE = {"switch": 144, "ps3": 112}
_ENTRY_SIZE = {"switch": 256, "ps3": 240}
_TILE_SIZE = {"switch": 80, "ps3": 72}
_GENERATOR_SIZE = 120
_LIST_SIZE = 24

MARKER_TYPES = {
    0: "actor", 1: "light", 3: "camera", 4: "audio_volume", 5: "weather_volume",
    6: "ambient_sound", 7: "particle", 10: "encounter", 13: "script", 14: "adventure",
    16: "sub_scene_slot", 28: "map", 29: "event", 30: "conductor", 31: "effect_group",
    **{50 + i: f"gizmo_location_{chr(65 + i)}" for i in range(26)},
    **{i: f"location_{i}" for i in range(76, 102)},
}
# Marker type: the group its reference belongs to.
MARKER_GROUPS = {0: 1, 1: 23, 5: 47, 6: 5, 7: 27, 10: 15, 13: 35, 14: 2, 29: 31, 31: 14}
EXIT_NORTH, EXIT_SOUTH, EXIT_EAST, EXIT_WEST = 1, 2, 4, 8

Bounds = tuple[Vector, Vector]


def _body(body: int) -> int:
    return 0x10 + body


def _bounds(view: ByteView, body: int) -> Bounds:
    values = view.f32s(_body(body), 6, "bounds")
    return values[:3], values[3:]


def _u32_list(view: ByteView, body: int, what: str) -> list[int]:
    position = _body(body)
    raw = view.pair_data(view.pair(position), position, what)
    if len(raw) % 4:
        raise FormatError(f"{what}: {len(raw)} bytes is not a list of u32", position)
    return list(ByteView(raw, view.order).unpack(f"{len(raw) // 4}I", 0))


@dataclass
class SceneInfo:
    header: SnoHeader
    build: str
    bounds: Bounds
    marker_bounds: Bounds
    exclusions: list[int]
    inclusions: list[int]
    marker_sets: list[int]
    look_hash: int
    xp_density: float | None
    appearance: int
    collision_mesh: int


def parse_scene_info(data: bytes, build: str | None = None, strict: bool = False) -> SceneInfo:
    header = parse_header(data, build)
    resolved = resolve_build("scn", header, build, strict)
    view = ByteView(data, header.order)
    view.need(0, _body(SCENE_FIXED), "scene fixed part")
    if resolved == "switch":
        xp, appearance, collision = view.f32(_body(0x208)), view.i32(_body(0x20C)), view.i32(_body(0x210))
    else:
        xp, appearance, collision = None, view.i32(_body(0x210)), view.i32(_body(0x214))
    return SceneInfo(
        header, resolved, _bounds(view, 0x10), _bounds(view, 0x28),
        _u32_list(view, 0xA0, "exclusions"), _u32_list(view, 0xE0, "inclusions"),
        _u32_list(view, 0x120, "marker sets"), view.u32(_body(0x160)), xp, appearance, collision,
    )


@dataclass
class Marker:
    name_hash: int
    type: int
    rotation: Quaternion
    position: Vector
    group: int
    handle: int
    tag_map_raw: bytes
    links: list[tuple[int, int]]
    order: str

    @property
    def type_name(self) -> str | None:
        return MARKER_TYPES.get(self.type)

    @property
    def tags(self) -> TagMap:
        """The tag map. Type 7 is a 4-byte value here, and ``error`` says where reading stopped."""
        if not self.tag_map_raw:
            return TagMap("marker", Pair(0, 0))
        view = ByteView(bytes(0x10) + self.tag_map_raw, self.order)
        return read_tag_map(view, Pair(0, len(self.tag_map_raw)), "marker", lenient=True)

    @property
    def scale(self) -> float:
        """Tag 0x80000, which defaults to 1.0."""
        record = self.tags.get(SCALE_TAG)
        return record.value if record is not None and record.type == TAG_FLOAT else 1.0

    def actor_scale(
        self, actor_flags: int, scale_minimum: float, scale_delta: float, random_fraction=None,
    ) -> float:
        """The scale an actor marker ends up with.

        ``scale_minimum`` and ``scale_delta`` are actor tags 0x10007 and 0x10008.
        ``random_fraction()`` is only called for a non-zero delta.
        """
        scale = self.scale
        if self.type != TYPE_ACTOR or (scale >= ACTOR_SCALE_FLOOR and not actor_flags & 1):
            return scale
        if scale_delta == 0:
            return scale_minimum
        if random_fraction is None:
            raise FormatError("a scale delta needs a random source")
        return scale_minimum + random_fraction() * scale_delta


@dataclass
class MarkerSet:
    header: SnoHeader
    build: str
    markers: list[Marker]
    circles: list[tuple[float, float, float]]
    bounds: Bounds
    flags: int
    label: int
    special_indices: list[int]

    @property
    def always_active(self) -> bool:
        return self.label == LABEL_ALWAYS


def _read_marker(view: ByteView, at: int) -> Marker:
    raw = view.take(at, MARKER_SIZE, "marker")
    r = ByteView(raw, view.order)
    tag_pair = r.pair(0x30)
    link_count = r.i32(0x40)
    link_pair = r.pair(0x44)
    if link_count < 0 or link_pair.size != link_count * LINK_SIZE:
        raise FormatError(f"links size {link_pair.size} for {link_count}", at + 0x44)
    links_raw = view.pair_data(link_pair, at + 0x44, "marker links")
    links = list(struct.iter_unpack(view.order + "II", links_raw))
    return Marker(
        r.u32(0), r.u32(8), r.f32s(0x0C, 4), r.f32s(0x1C, 3), r.i32(0x28), r.i32(0x2C),
        view.pair_data(tag_pair, at + 0x30, "marker tag map"), links, view.order,
    )


def parse_marker_set(data: bytes, build: str | None = None) -> MarkerSet:
    header = parse_header(data, build)
    view = ByteView(data, header.order)
    view.need(0, _body(MARKER_SET_FIXED), "marker set fixed part")
    pair = view.pair(_body(0x0C))
    if pair.size % MARKER_SIZE:
        raise FormatError(f"markers of {pair.size} bytes", _body(0x0C))
    view.pair_data(pair, _body(0x0C), "markers")
    markers = [
        _read_marker(view, pair.file_offset + i * MARKER_SIZE)
        for i in range(pair.size // MARKER_SIZE)
    ]
    circles_raw = view.pair_data(view.pair(_body(0x50)), _body(0x50), "no-spawn circles")
    if len(circles_raw) % CIRCLE_SIZE:
        raise FormatError(f"circles of {len(circles_raw)} bytes", _body(0x50))
    special_count = view.i32(_body(0xB0))
    special_pair = view.pair(_body(0xB4))
    if special_count < 0 or special_pair.size != special_count * 2:
        raise FormatError(f"special index size {special_pair.size}", _body(0xB4))
    special_raw = view.pair_data(special_pair, _body(0xB4), "special indices")
    return MarkerSet(
        header, header.build, markers,
        list(struct.iter_unpack(view.order + "3f", circles_raw)), _bounds(view, 0x90),
        view.u32(_body(0xA8)), view.i32(_body(0xAC)),
        list(ByteView(special_raw, view.order).unpack(f"{special_count}H", 0)),
    )


@dataclass
class SceneEntry:
    group: int
    handle: int
    rotation: Quaternion
    position: Vector  # the scene's minimum corner
    level_areas: tuple[int, ...]
    cluster_id: int


@dataclass(frozen=True)
class Tile:
    exits: int
    tile_set: int
    scene: int
    weight: int


@dataclass
class Generator:
    tiles: list[Tile]
    command_count: int
    commands_raw: bytes
    raw: bytes


@dataclass
class ServerBlock:
    generators: list[Generator]
    scene_lists: list[list[SceneEntry]]
    next_cluster_id: int
    cluster_count: int
    clusters_raw: bytes
    navigation_powers: tuple[int, ...]
    script_handle: int
    seed: int


@dataclass
class WorldFile:
    header: SnoHeader
    build: str
    is_generated: int
    server: ServerBlock | None
    marker_sets: list[int]
    skybox_actor: int | None  # open on ps3, where the word is exposed as skybox_word
    skybox_word: int
    music: int
    ambient: int
    reverb: int
    weather: int
    warnings: list[str] = field(default_factory=list)

    @property
    def scene_entries(self) -> list[SceneEntry]:
        if self.server is None:
            return []
        return [entry for entries in self.server.scene_lists for entry in entries]


def _read_entries(view: ByteView, record: ByteView, build: str, position: int) -> list[SceneEntry]:
    pair = record.pair(0)
    count = record.i32(8)
    size = _ENTRY_SIZE[build]
    if count < 0 or pair.size != count * size:
        raise FormatError(f"scene chunk size {pair.size} for {count} entries", position)
    raw = view.pair_data(pair, position, "scene chunk")
    entries = []
    cluster_at = 0xAC if build == "switch" else 0x9C
    for i in range(count):
        e = ByteView(raw[i * size:(i + 1) * size], view.order)
        entries.append(SceneEntry(
            e.i32(0), e.u32(4), e.f32s(8, 4), e.f32s(0x18, 3), e.unpack("4i", 0x30),
            e.i32(cluster_at),
        ))
    return entries


def _read_generator(view: ByteView, raw: bytes, build: str, position: int) -> Generator:
    r = ByteView(raw, view.order)
    size = _TILE_SIZE[build]
    tiles_raw = view.pair_data(r.pair(0), position, "tiles")
    if len(tiles_raw) % size:
        raise FormatError(f"tiles of {len(tiles_raw)} bytes", position)
    tiles = [
        Tile(*ByteView(tiles_raw[i:i + size], view.order).unpack("4i", 0))
        for i in range(0, len(tiles_raw), size)
    ]
    return Generator(
        tiles, r.i32(0x40), view.pair_data(r.pair(0x44), position, "commands"), raw,
    )


def _read_server(view: ByteView, pair: Pair, build: str, warnings: list[str]) -> ServerBlock:
    size = _BLOCK_SIZE[build]
    if pair.size != size:
        raise FormatError(f"server block of {pair.size} bytes", _body(0x10))
    raw = view.pair_data(pair, _body(0x10), "server block")
    r = ByteView(raw, view.order)
    generators_raw = view.pair_data(r.pair(0), pair.file_offset, "generators")
    if len(generators_raw) % _GENERATOR_SIZE:
        raise FormatError(f"generators of {len(generators_raw)} bytes", pair.file_offset)
    generators = [
        _read_generator(view, generators_raw[i:i + _GENERATOR_SIZE], build, pair.file_offset)
        for i in range(0, len(generators_raw), _GENERATOR_SIZE)
    ]
    lists_raw = view.pair_data(r.pair(0x10), pair.file_offset + 0x10, "scene lists")
    if len(lists_raw) % _LIST_SIZE:
        raise FormatError(f"scene lists of {len(lists_raw)} bytes", pair.file_offset + 0x10)
    scene_lists = [
        _read_entries(view, ByteView(lists_raw[i:i + _LIST_SIZE], view.order), build,
                      pair.file_offset + 0x10)
        for i in range(0, len(lists_raw), _LIST_SIZE)
    ]
    if generators and any(scene_lists):
        warnings.append("world has both scene lists and generator data")
    script_at, seed_at = (0x74, 0x78) if build == "switch" else (0x68, 0x6C)
    return ServerBlock(
        generators, scene_lists, r.i32(0x38), r.i32(0x40),
        view.pair_data(r.pair(0x50), pair.file_offset + 0x50, "clusters"), r.unpack("4i", 0x58),
        r.i32(script_at), r.i32(seed_at),
    )


def parse_world(data: bytes, build: str | None = None, strict: bool = False) -> WorldFile:
    header = parse_header(data, build)
    resolved = header.build
    view = ByteView(data, header.order)
    view.need(0, _body(_WORLD_FIXED[resolved]), "world fixed part")
    skybox, music, ambient, reverb, weather = (
        (0x84, 0x88, 0x98, 0x9C, 0xA0) if resolved == "switch" else (0xD4, 0xD8, 0xDC, 0xE0, 0xE4)
    )
    warnings: list[str] = []
    pair = view.pair(_body(0x10), "server block pair")
    server = None if pair.size == 0 else _read_server(view, pair, resolved, warnings)
    skybox_word = view.i32(_body(skybox))
    world = WorldFile(
        header, resolved, view.u32(_body(0x0C)), server, _u32_list(view, 0x20, "world marker sets"),
        skybox_word if resolved == "switch" else None, skybox_word,
        view.i32(_body(music)), view.i32(_body(ambient)),
        view.i32(_body(reverb)), view.i32(_body(weather)), warnings,
    )
    if strict and warnings:
        raise FormatError(warnings[0], _body(0x10))
    return world


@dataclass
class PlacedMarker:
    marker: Marker
    marker_set: int
    scene: SceneEntry | None  # None for a set on the world itself
    position: Vector
    rotation: Quaternion
    conditional: bool


@dataclass
class WorldLayout:
    scenes: list[SceneEntry]
    markers: list[PlacedMarker]
    sub_scene_slots: list[PlacedMarker]
    missing_scenes: list[int]
    missing_marker_sets: list[int]


def _place(layout: WorldLayout, placed: PlacedMarker) -> None:
    layout.markers.append(placed)
    if placed.marker.type == TYPE_SUB_SCENE:
        layout.sub_scene_slots.append(placed)


def lay_out_world(
    world: WorldFile, scenes: dict[int, SceneInfo], marker_sets: dict[int, MarkerSet],
    active_labels: dict[int, set[int]] | None = None, origin: tuple[float, float] = (0.0, 0.0),
) -> WorldLayout:
    """Place every marker of a static world. Both dictionaries are keyed by handle.

    With ``active_labels``, per scene handle, a scene's labelled sets are made only if listed,
    as in normal play. Without it every set is made, as in editor modes, and flagged
    ``conditional``. Sets on the world are never filtered, and move by minus ``origin``.
    """
    layout = WorldLayout(world.scene_entries, [], [], [], [])
    for entry in world.scene_entries:
        scene = scenes.get(entry.handle)
        if scene is None:
            layout.missing_scenes.append(entry.handle)
            continue
        for set_handle in scene.marker_sets:
            marker_set = marker_sets.get(set_handle)
            if marker_set is None:
                layout.missing_marker_sets.append(set_handle)
                continue
            if active_labels is not None and not marker_set.always_active:
                if marker_set.label not in active_labels.get(entry.handle, ()):
                    continue
            for marker in marker_set.markers:
                moved = rotate(entry.rotation, marker.position)
                _place(layout, PlacedMarker(
                    marker, set_handle, entry,
                    (
                        entry.position[0] + moved[0],
                        entry.position[1] + moved[1],
                        entry.position[2] + moved[2],
                    ),
                    quat_multiply(entry.rotation, marker.rotation),
                    not marker_set.always_active,
                ))
    for set_handle in world.marker_sets:
        marker_set = marker_sets.get(set_handle)
        if marker_set is None:
            layout.missing_marker_sets.append(set_handle)
            continue
        for marker in marker_set.markers:
            position = (
                marker.position[0] - origin[0], marker.position[1] - origin[1], marker.position[2],
            )
            _place(layout, PlacedMarker(
                marker, set_handle, None, position, marker.rotation, False,
            ))
    return layout
