"""Cloth records of appearance sub-objects (``20_corrections9.md`` G1-4).

One cloth record per cloth sub-object (``ClothStructure`` on Switch and x360, ``ClothStructureP3``
on PS3). Array lengths come from the pair sizes; the count fields equal them, except
``nFaceCount``, which may be smaller than the face array (the extra entries are kept and written
unchanged; V9-G1-4: the face rules apply to the first ``nFaceCount`` faces only).

``cloth_problems`` checks the rules that hold in every record of ``s10_cloth_rules.tsv``. Rules
with exceptions there are not checked: the attachment step (6 / 2 / 2 records), the parent being a
distance neighbour, the bending pairs, the unfolded bending length, and the face order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair

RECORD_SIZE = {"switch": 144, "x360": 144, "ps3": 100}
# Field offsets: count, pair for vertices, faces, staples, distance, bending; islands, area, map.
_FIELDS = {
    "switch": dict(vertices=(0x00, 0x04), faces=(0x18, 0x1C), staples=(0x30, 0x34),
                   distance=(0x48, 0x4C), bending=(0x60, 0x64), islands=0x78, area=0x7C, map=0x80),
    "ps3": dict(vertices=(0x00, 0x04), faces=(0x10, 0x14), staples=(0x20, 0x24),
                distance=(0x30, 0x34), bending=(0x40, 0x44), islands=0x50, area=0x54, map=0x58),
}
VERTEX_SIZE, FACE_SIZE, STAPLE_SIZE, CONSTRAINT_SIZE = 44, 16, 28, 16
HORIZONTAL_SLOPE = 0.5


@dataclass(frozen=True)
class ClothVertex:
    wp: tuple[float, float, float]  # rest position in cloth axes
    wp0: tuple[float, float, float]
    inv_mass: float
    parent: int
    attachment_distance: int
    fat_vertex: int  # a render vertex mapped to this cloth vertex
    island: int


@dataclass(frozen=True)
class ClothFace:
    rest_area: float
    vertices: tuple[int, int, int]


@dataclass(frozen=True)
class ClothStaple:
    vertex: int
    bones: tuple[int, int, int]
    weights: tuple[float, float, float]


@dataclass(frozen=True)
class ClothConstraint:
    vertices: tuple[int, int]
    rest_length_squared: float
    horizontal: float


@dataclass
class ClothRecord:
    vertex_count: int
    face_count: int  # may be below len(faces) (G1-4)
    staple_count: int
    distance_count: int
    bending_count: int
    island_count: int
    area: float
    vertices: list[ClothVertex] = field(default_factory=list)
    faces: list[ClothFace] = field(default_factory=list)
    staples: list[ClothStaple] = field(default_factory=list)
    distance: list[ClothConstraint] = field(default_factory=list)
    bending: list[ClothConstraint] = field(default_factory=list)
    mapping: list[int] = field(default_factory=list)  # render vertex -> cloth vertex
    problems: list[str] = field(default_factory=list)  # layout problems found while reading

    @property
    def pinned(self) -> set[int]:
        return {s.vertex for s in self.staples}


def model_position(wp: tuple[float, float, float]) -> tuple[float, float, float]:
    """G1-4: render (model) axes from cloth axes: (wp.z, wp.x, wp.y)."""
    return (wp[2], wp[0], wp[1])


def _array(view: ByteView, record: ByteView, base: int, count_at: int, pair_at: int,
           element: int, name: str, problems: list[str], allow_more: bool = False) -> tuple[int, bytes]:
    count = record.i32(count_at)
    pair: Pair = record.pair(pair_at)
    raw = view.pair_data(pair, base + pair_at, f"cloth {name}")
    if len(raw) % element:
        problems.append(f"cloth {name}: pair of {len(raw)} bytes is not a multiple of {element}")
    length = len(raw) // element
    if count != length and not (allow_more and 0 <= count <= length):
        problems.append(f"cloth {name}: count {count}, array of {length}")
    return count, raw[:length * element]


def read_cloth(view: ByteView, pair: Pair, build: str, position: int) -> ClothRecord:
    """Read the cloth record a sub-object's cloth pair points at."""
    layout = _FIELDS["ps3" if build == "ps3" else "switch"]
    size = RECORD_SIZE[build]
    raw = view.pair_data(pair, position, "cloth record")
    if len(raw) != size:
        raise FormatError(f"cloth pair of {len(raw)} bytes, one record is {size}", position)
    record = ByteView(raw, view.order)
    base = pair.file_offset
    problems: list[str] = []
    o = view.order
    n_v, verts = _array(view, record, base, *layout["vertices"], VERTEX_SIZE, "vertices", problems)
    n_f, faces = _array(view, record, base, *layout["faces"], FACE_SIZE, "faces", problems, True)
    n_s, staples = _array(view, record, base, *layout["staples"], STAPLE_SIZE, "staples", problems)
    n_d, dist = _array(view, record, base, *layout["distance"], CONSTRAINT_SIZE, "distance constraints", problems)
    n_b, bend = _array(view, record, base, *layout["bending"], CONSTRAINT_SIZE, "bending constraints", problems)
    map_pair = record.pair(layout["map"])
    map_raw = view.pair_data(map_pair, base + layout["map"], "cloth vertex map")
    if len(map_raw) % 2:
        problems.append(f"cloth vertex map of {len(map_raw)} bytes")
    cloth = ClothRecord(n_v, n_f, n_s, n_d, n_b, record.i32(layout["islands"]),
                        record.f32(layout["area"]), problems=problems)
    v = ByteView(verts, o)
    for i in range(len(verts) // VERTEX_SIZE):
        at = i * VERTEX_SIZE
        cloth.vertices.append(ClothVertex(
            v.f32s(at, 3), v.f32s(at + 12, 3), v.f32(at + 24), *v.unpack("4i", at + 28)))
    f = ByteView(faces, o)
    for i in range(len(faces) // FACE_SIZE):
        at = i * FACE_SIZE
        cloth.faces.append(ClothFace(f.f32(at), f.unpack("3i", at + 4)))
    s = ByteView(staples, o)
    for i in range(len(staples) // STAPLE_SIZE):
        at = i * STAPLE_SIZE
        cloth.staples.append(ClothStaple(s.i32(at), s.unpack("3i", at + 4), s.f32s(at + 16, 3)))
    for raw_c, target in ((dist, cloth.distance), (bend, cloth.bending)):
        c = ByteView(raw_c, o)
        for i in range(len(raw_c) // CONSTRAINT_SIZE):
            at = i * CONSTRAINT_SIZE
            target.append(ClothConstraint(c.unpack("2i", at), c.f32(at + 8), c.f32(at + 12)))
    m = ByteView(map_raw, o)
    cloth.mapping = list(m.unpack(f"{len(map_raw) // 2}H", 0)) if len(map_raw) >= 2 else []
    return cloth


# --- Rules -----------------------------------------------------------------------------------

def _close(a: float, b: float, rel: float, absolute: float = 1e-9) -> bool:
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), absolute)


def _distance_squared(a, b) -> float:
    return sum((a[k] - b[k]) ** 2 for k in range(3))


def _area(a, b, c) -> float:
    u = [b[k] - a[k] for k in range(3)]
    w = [c[k] - a[k] for k in range(3)]
    cross = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
    return 0.5 * math.sqrt(sum(x * x for x in cross))


def components(count: int, edges) -> list[int]:
    """Connected-component label of each of ``count`` vertices (labels in first-seen order)."""
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if 0 <= a < count and 0 <= b < count:
            parent[find(a)] = find(b)
    labels: dict[int, int] = {}
    return [labels.setdefault(find(i), len(labels)) for i in range(count)]


def cloth_problems(cloth: ClothRecord, render: list[tuple[float, float, float]] | None,
                   tolerance: tuple[float, float, float], bone_count: int, where: str) -> list[str]:
    """The G1-4 rules that hold in every record. ``render`` are the positions of the render
    vertices (fat, or packed rigid), in model axes; ``tolerance`` per axis for the axis rule."""
    found = [f"{where}: {p}" for p in cloth.problems]
    n = len(cloth.vertices)
    pinned = cloth.pinned
    for i, vertex in enumerate(cloth.vertices):
        if vertex.wp != vertex.wp0:
            found.append(f"{where}: cloth vertex {i}: wp differs from wp0")
            break
    # Staples.
    staple_vertices = [s.vertex for s in cloth.staples]
    if any(not 0 <= s < n for s in staple_vertices):
        found.append(f"{where}: staple vertex past {n} cloth vertices")
    if len(set(staple_vertices)) != len(staple_vertices):
        found.append(f"{where}: staple vertices are not distinct")
    for s in cloth.staples:
        if any(b != 0 for b in s.bones) if bone_count == 0 else any(not 0 <= b < bone_count for b in s.bones):
            found.append(f"{where}: staple bone index past {bone_count} bones")
            break
    for s in cloth.staples:
        if list(s.weights) != sorted(s.weights, reverse=True):
            found.append(f"{where}: staple weights not descending")
            break
    # Pins, parents, islands.
    for i, vertex in enumerate(cloth.vertices):
        stapled = i in pinned
        if (vertex.inv_mass == 0) != stapled or vertex.inv_mass < 0:
            found.append(f"{where}: cloth vertex {i}: inverse mass {vertex.inv_mass}, stapled {stapled}")
            break
        if (vertex.parent == -1) != stapled or not -1 <= vertex.parent < n:
            found.append(f"{where}: cloth vertex {i}: parent {vertex.parent}, stapled {stapled}")
            break
        if (vertex.attachment_distance == 0) != stapled:
            found.append(f"{where}: cloth vertex {i}: attachment distance "
                         f"{vertex.attachment_distance}, stapled {stapled}")
            break
    islands = [v.island for v in cloth.vertices]
    if any(not 0 <= k < cloth.island_count for k in islands):
        found.append(f"{where}: island index outside 0..{cloth.island_count - 1}")
    elif n and len(set(islands)) != cloth.island_count:
        found.append(f"{where}: {len(set(islands))} islands used, nIslandCount {cloth.island_count}")
    if cloth.distance:
        labels = components(n, [c.vertices for c in cloth.distance])
        if len(set(zip(labels, islands))) != len(set(labels)) or len(set(labels)) != len(set(islands)):
            found.append(f"{where}: islands are not the distance-constraint components")
    elif cloth.island_count != n:
        found.append(f"{where}: no distance constraints, nIslandCount {cloth.island_count} for {n} vertices")
    # Faces (V9-G1-4): the rules hold for the first nFaceCount faces only. Entries after them are
    # kept and written unchanged but not checked (usually zero, or a triangle whose stored rest
    # area is not its area).
    counted_faces = cloth.faces[:cloth.face_count]
    area = 0.0
    for k, face in enumerate(counted_faces):
        if any(not 0 <= x < n for x in face.vertices):
            found.append(f"{where}: face {k} vertex past {n}")
            break
        a, b, c = (cloth.vertices[x].wp for x in face.vertices)
        if not _close(face.rest_area, _area(a, b, c), 1e-3, 1e-6):
            found.append(f"{where}: face {k} rest area differs from its triangle")
            break
        area += face.rest_area
    if counted_faces and not _close(cloth.area, area, 1e-4, 1e-6):
        found.append(f"{where}: flArea {cloth.area} is not the sum of the counted face rest areas {area}")
    # Constraints.
    for kind, constraints in (("distance", cloth.distance), ("bending", cloth.bending)):
        for k, con in enumerate(constraints):
            if any(not 0 <= x < n for x in con.vertices):
                found.append(f"{where}: {kind} constraint {k} vertex past {n}")
                break
            if kind == "bending":
                if con.horizontal != 0.0:
                    found.append(f"{where}: bending constraint {k}: flHorizontal {con.horizontal}")
                    break
                continue
            a, b = (cloth.vertices[x].wp for x in con.vertices)
            length2 = _distance_squared(a, b)
            if not _close(con.rest_length_squared, length2, 1e-3, 1e-9):
                found.append(f"{where}: distance constraint {k}: rest length differs from wp")
                break
            slope = abs(a[1] - b[1]) / math.sqrt(length2) if length2 else 0.0
            if con.horizontal != (1.0 if slope < HORIZONTAL_SLOPE else 0.0):
                found.append(f"{where}: distance constraint {k}: flHorizontal {con.horizontal}, "
                             f"slope {slope:.3f}")
                break
    if cloth.distance:
        pairs = {frozenset(c.vertices) for c in cloth.distance}
        edges = set()
        for face in counted_faces:  # V9-G1-4: the counted faces only
            a, b, c = face.vertices
            edges |= {frozenset((a, b)), frozenset((b, c)), frozenset((c, a))}
        if pairs != {e for e in edges if not e <= pinned}:
            found.append(f"{where}: distance pairs are not the face edges without pinned-pinned edges")
    # Mapping and axes.
    if render is not None:
        mapping = cloth.mapping
        if len(mapping) != len(render):
            found.append(f"{where}: vertex map of {len(mapping)} entries for {len(render)} render vertices")
        elif any(m >= n for m in mapping):
            found.append(f"{where}: vertex map value past {n} cloth vertices")
        else:
            if set(mapping) != set(range(n)):
                found.append(f"{where}: a cloth vertex is not mapped")
            for i, vertex in enumerate(cloth.vertices):
                j = vertex.fat_vertex
                if not 0 <= j < len(mapping) or mapping[j] != i:
                    found.append(f"{where}: cloth vertex {i}: nFatVertexIndex {j} is not mapped to it")
                    break
            for j, m in enumerate(mapping):
                expected = model_position(cloth.vertices[m].wp)
                if any(abs(render[j][k] - expected[k]) > tolerance[k] for k in range(3)):
                    found.append(f"{where}: render vertex {j} is not cloth vertex {m} in model axes")
                    break
    return found
