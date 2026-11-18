"""Stage 14: derived values of ``.phm`` collision meshes (``28_derived_values.md`` W14-12 to W14-15)
and a writer for parsed collision files.

- W14-12 (bounds): centre = f32((min + max) / 2) per axis; stored vertex = f32(position - centre)
  with a 4th float 0; half extent = f32(raw max + pad), pad 0.164042 (switch form) or 0.164 (ps3);
  scale = f32(half / 32767) (switch form) or f32(half / 32766) (ps3). ps3: the triangle table has
  three more entries than the triangle count, copies of the last used triangle (PROVEN tail).
- W14-13 (tree): pre-order binary tree, leaves of 1 to 4 triangles, leaf order = table order;
  split on the axis of largest centre extent, left child = the prefix of the sorted triangles with
  the least area(L) * |L| + area(R) * |R| (SUPPORTED choice; shape and leaf order PROVEN).
- W14-14: switch node bounds (SUPPORTED) and ps3 split values (PROVEN).
- W14-15: triangle neighbours (PROVEN for slots with at most one candidate).

``build_collision_mesh`` derives everything from positions and triangles. ``write_collision``
rebuilds a file from a parsed :class:`~d3sno.collision.CollisionFile` with the generic tree writer
(W11); ``derive=False`` (default) stores every value as held by the model, so an unchanged
canonical file is rebuilt byte for byte; ``derive="proven"`` recomputes only the PROVEN values on
the stored tree (counts, neighbours, ps3 split values, ps3 table tail, tree height);
``derive=True`` rebuilds each mesh with ``build_collision_mesh``.

f32 arithmetic: every sum with a constant is done with the constant rounded to single precision
(the operation of two singles computed in double and rounded once equals the single-precision
operation). See the notes for the choices this implies.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable, Sequence

from .binary import FormatError, build_order
from .collision import _FIELDS, LEAF, CollisionFile, CollisionMesh, TreeNode, Triangle
from .sno import SNO_SIGNATURE, SnoHeader, expected_version
from . import sno_layout as L

__all__ = [
    "HALF_PAD", "QUANT_MAX", "NODE_PAD", "SPLIT_PAD", "f32", "form_of",
    "mesh_bounds", "build_tree", "triangle_neighbours", "switch_node_bounds", "ps3_split_values",
    "tree_height", "tree_nodes", "leaf_runs", "build_collision_mesh", "new_collision_file",
    "write_collision", "derive_mesh", "non_manifold_slots", "proven_neighbours",
]

HALF_PAD = {"switch": 0.164042, "ps3": 0.164}
QUANT_MAX = {"switch": 32767, "ps3": 32766}
NODE_PAD = 0.0328084  # switch form node bounds
SPLIT_PAD = 0.0164  # ps3 split values
LEAF_MAX = 4

_F = struct.Struct("<f")


def f32(x: float) -> float:
    """Round to single precision."""
    return _F.unpack(_F.pack(x))[0]


def form_of(build: str) -> str:
    """"ps3" or "switch" (x360 uses the switch form, big-endian)."""
    if build not in ("ps3", "switch", "x360"):
        raise FormatError(f"unknown build {build!r}")
    return "ps3" if build == "ps3" else "switch"


def _i16(value: int) -> int:
    return max(-32768, min(32767, value))


# --- W14-12 -------------------------------------------------------------------------------------

def mesh_bounds(positions: Sequence[Sequence[float]], build: str):
    """W14-12: (centre, stored vertex rows, half extent, scale) of a list of positions."""
    form = form_of(build)
    pad = f32(HALF_PAD[form])
    centre = []
    for axis in range(3):
        values = [p[axis] for p in positions]
        centre.append(f32((min(values) + max(values)) / 2) if values else 0.0)
    rows = [(f32(p[0] - centre[0]), f32(p[1] - centre[1]), f32(p[2] - centre[2]), 0.0) for p in positions]
    half, scale = [], []
    for axis in range(3):
        top = max((row[axis] for row in rows), default=0.0)
        h = f32(top + pad)
        half.append(h)
        scale.append(f32(h / QUANT_MAX[form]))
    return tuple(centre), rows, tuple(half), tuple(scale)


# --- W14-13 -------------------------------------------------------------------------------------

class _Node:
    __slots__ = ("tris", "depth", "axis", "leaf", "right", "index", "first")

    def __init__(self, tris, depth):
        self.tris, self.depth = tris, depth
        self.axis = LEAF
        self.leaf = True
        self.right = 0
        self.index = 0
        self.first = 0


def _boxes(vertices, triangles):
    boxes = []
    for tri in triangles:
        pts = [vertices[v] for v in tri]
        boxes.append((tuple(min(p[a] for p in pts) for a in range(3)),
                      tuple(max(p[a] for p in pts) for a in range(3))))
    return boxes


def _area(lo, hi) -> float:
    dx, dy, dz = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    return 2.0 * (dx * dy + dy * dz + dz * dx)


def _split(tris: list[int], boxes) -> tuple[int, list[int], int]:
    """W14-13 split of a node: (axis, triangles sorted on it, prefix length of the left child)."""
    centres = {t: tuple((boxes[t][0][a] + boxes[t][1][a]) / 2 for a in range(3)) for t in tris}
    extents = [max(centres[t][a] for t in tris) - min(centres[t][a] for t in tris) for a in range(3)]
    axis = extents.index(max(extents))  # first axis on a tie
    ordered = sorted(tris, key=lambda t: centres[t][axis])  # stable: ties keep node order
    n = len(ordered)
    inf = float("inf")
    prefix = []
    lo, hi = [inf] * 3, [-inf] * 3
    for t in ordered:
        b = boxes[t]
        lo = [min(lo[a], b[0][a]) for a in range(3)]
        hi = [max(hi[a], b[1][a]) for a in range(3)]
        prefix.append(_area(lo, hi))
    suffix = [0.0] * n
    lo, hi = [inf] * 3, [-inf] * 3
    for i in range(n - 1, -1, -1):
        b = boxes[ordered[i]]
        lo = [min(lo[a], b[0][a]) for a in range(3)]
        hi = [max(hi[a], b[1][a]) for a in range(3)]
        suffix[i] = _area(lo, hi)
    best, best_k = inf, 1
    for k in range(1, n):
        cost = prefix[k - 1] * k + suffix[k] * (n - k)
        if cost < best:  # first minimum on a tie
            best, best_k = cost, k
    return axis, ordered, best_k


def build_tree(vertices, triangles) -> tuple[list[_Node], list[int]]:
    """W14-13: nodes in pre-order and the leaf order (input triangle indices, table order).
    ``vertices`` are the stored (centred) vertex rows."""
    if not triangles:
        return [], []
    boxes = _boxes(vertices, triangles)
    nodes: list[_Node] = []
    order: list[int] = []
    pending = [(_Node(list(range(len(triangles))), 1), None)]
    while pending:
        node, parent = pending.pop()
        node.index = len(nodes)
        nodes.append(node)
        if parent is not None:
            parent.right = node.index - parent.index
        if len(node.tris) <= LEAF_MAX:
            node.first = len(order)
            order.extend(node.tris)
            continue
        axis, ordered, k = _split(node.tris, boxes)
        node.leaf, node.axis = False, axis
        left, right = _Node(ordered[:k], node.depth + 1), _Node(ordered[k:], node.depth + 1)
        pending.append((right, node))
        pending.append((left, None))
    return nodes, order


def tree_height(nodes: Sequence[TreeNode], build: str) -> int:
    """W14-13: depth of the deepest node, root 1 (0 for an empty tree)."""
    shift = 2 if form_of(build) == "ps3" else 6
    height = 0
    pending = [(0, 1)] if nodes else []
    while pending:
        index, depth = pending.pop()
        height = max(height, depth)
        node = nodes[index]
        if not node.is_leaf:
            pending.append((index + (node.data >> shift), depth + 1))
            pending.append((index + 1, depth + 1))
    return height


def leaf_runs(nodes: Sequence[TreeNode], build: str) -> dict[int, list[int]]:
    """For every node index, the table entries of its subtree (from the stored tree)."""
    ps3 = form_of(build) == "ps3"
    shift = 2 if ps3 else 6
    runs: dict[int, list[int]] = {}

    def visit(index: int) -> list[int]:
        node = nodes[index]
        if node.is_leaf:
            first = node.data >> shift
            count = node.a if ps3 else (node.data >> 2) & 15
            result = list(range(first, first + count))
        else:
            result = visit(index + 1) + visit(index + (node.data >> shift))
        runs[index] = result
        return result

    if nodes:
        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 4 * len(nodes) + 100))
        try:
            visit(0)
        finally:
            sys.setrecursionlimit(old)
    return runs


# --- W14-14 -------------------------------------------------------------------------------------

def switch_node_bounds(rows, tris: Iterable[Sequence[int]], scale) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """W14-14 switch form: (lower, upper) of a node from its triangles' stored vertices."""
    pad = f32(NODE_PAD)
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for tri in tris:
        for v in tri:
            for a in range(3):
                lo[a] = min(lo[a], rows[v][a])
                hi[a] = max(hi[a], rows[v][a])
    lower = tuple(_i16(math.floor(f32(f32(lo[a] - pad) / scale[a]))) for a in range(3))
    upper = tuple(_i16(math.ceil(f32(f32(hi[a] + pad) / scale[a]))) for a in range(3))
    return lower, upper


def ps3_split_values(rows, left: Iterable[Sequence[int]], right: Iterable[Sequence[int]],
                     axis: int, scale) -> tuple[int, int]:
    """W14-14 ps3: (a, b) of an internal node, as stored unsigned 16-bit words."""
    pad = f32(SPLIT_PAD)
    s = scale[axis]
    top = max(rows[v][axis] for tri in left for v in tri)
    bottom = min(rows[v][axis] for tri in right for v in tri)
    a = math.ceil(f32(f32(top + pad) / s))
    b = math.floor(f32(f32(bottom - pad) / s))
    return _i16(a) & 0xFFFF, _i16(b) & 0xFFFF


# --- W14-15 -------------------------------------------------------------------------------------

def triangle_neighbours(triangles: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    """W14-15: slot j = third vertex of the one other triangle with the reversed edge
    (v[j+1], v[j]); -1 when there is none, more than one, or another triangle has the same edge
    in the same direction."""
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for t, tri in enumerate(triangles):
        for j in range(3):
            edges.setdefault((tri[j], tri[(j + 1) % 3]), []).append((t, j))
    result = []
    for t, tri in enumerate(triangles):
        slots = []
        for j in range(3):
            a, b = tri[j], tri[(j + 1) % 3]
            candidates = [(u, k) for u, k in edges.get((b, a), ()) if u != t]
            same = [u for u, _ in edges.get((a, b), ()) if u != t]
            if len({u for u, _ in candidates}) != 1 or same:
                slots.append(-1)
                continue
            u, k = candidates[0]
            slots.append(triangles[u][(k + 2) % 3])
        result.append(tuple(slots))
    return result


def non_manifold_slots(triangles: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    """W14-15 / V16-3: the (triangle, slot) pairs whose edge is non-manifold: several candidate
    triangles with the reversed edge, or the edge also used in the same direction."""
    edges: dict[tuple[int, int], list[int]] = {}
    for t, tri in enumerate(triangles):
        for j in range(3):
            edges.setdefault((tri[j], tri[(j + 1) % 3]), []).append(t)
    found = set()
    for t, tri in enumerate(triangles):
        for j in range(3):
            a, b = tri[j], tri[(j + 1) % 3]
            if len({u for u in edges.get((b, a), ()) if u != t}) > 1 or any(u != t for u in edges[(a, b)]):
                found.add((t, j))
    return found


def _edge_users(triangles, keys) -> dict[tuple[int, int], tuple]:
    """Per undirected edge, the sorted keys of the triangles using it in either direction."""
    users: dict[tuple[int, int], list] = {}
    for tri, key in zip(triangles, keys):
        for j in range(3):
            a, b = tri[j], tri[(j + 1) % 3]
            users.setdefault((min(a, b), max(a, b)), []).append(key)
    return {edge: tuple(sorted(found)) for edge, found in users.items()}


def proven_neighbours(triangles, positions, source_triangles=None, source_positions=None,
                      source_neighbours=None) -> list[tuple[int, int, int]]:
    """V16-3: the neighbours a PROVEN-only derivation writes. Manifold slots follow W14-15. A
    non-manifold slot keeps the stored value of the source when the triangle and every triangle on
    that edge are unchanged (same vertices at the same positions, the same set of triangles on the
    edge); otherwise it is -1. Duplicated triangles (the same vertices) are matched by their
    occurrence order among the triangles with the same vertices and positions (V16-10). Without a source every non-manifold slot is -1 (as W14-15)."""
    triangles = [tuple(t) for t in triangles]
    result = [list(n) for n in triangle_neighbours(triangles)]
    if source_triangles is None:
        return [tuple(n) for n in result]

    def keys(tris, points):
        # V16-10: a triangle is its vertices with their positions plus its occurrence order among
        # the triangles with the same key (duplicated triangles may store different neighbours).
        seen: dict[tuple, int] = {}
        found = []
        for tri in tris:
            base = tuple((v, tuple(points[v][:3]) if v < len(points) else None) for v in tri)
            found.append((base, seen.get(base, 0)))
            seen[base] = seen.get(base, 0) + 1
        return found

    now_keys = keys(triangles, positions)
    old_tris = [tuple(t) for t in source_triangles]
    old_keys = keys(old_tris, source_positions)
    old_by_key = {}
    for k, n in zip(old_keys, source_neighbours):
        old_by_key.setdefault(k, n)
    now_users, old_users = _edge_users(triangles, now_keys), _edge_users(old_tris, old_keys)
    for t, j in non_manifold_slots(triangles):
        a, b = triangles[t][j], triangles[t][(j + 1) % 3]
        edge = (min(a, b), max(a, b))
        stored = old_by_key.get(now_keys[t])
        if stored is not None and old_users.get(edge) == now_users[edge]:
            result[t][j] = stored[j]
    return [tuple(n) for n in result]


# --- Mesh ---------------------------------------------------------------------------------------

def _per_triangle(value, count: int, what: str) -> list[int]:
    if isinstance(value, int):
        return [value] * count
    values = list(value)
    if len(values) != count:
        raise FormatError(f"{len(values)} {what} for {count} triangles")
    return values


def tree_nodes(nodes: list[_Node], tri_vertices, rows, scale, build: str) -> list[TreeNode]:
    """Pack built nodes (W14-13 packing, W14-14 values). ``tri_vertices`` by input index."""
    ps3 = form_of(build) == "ps3"
    result: list[TreeNode] = []
    for node in nodes:
        if node.leaf:
            count = len(node.tris)
            if ps3:
                result.append(TreeNode((node.first << 2) | LEAF, None, None, count, 0))
            else:
                lower, upper = switch_node_bounds(rows, (tri_vertices[t] for t in node.tris), scale)
                result.append(TreeNode((node.first << 6) | (count << 2) | LEAF, lower, upper, None, None))
            continue
        if ps3:
            left = nodes[node.index + 1].tris
            right = nodes[node.index + node.right].tris
            a, b = ps3_split_values(rows, [tri_vertices[t] for t in left],
                                    [tri_vertices[t] for t in right], node.axis, scale)
            result.append(TreeNode((node.right << 2) | node.axis, None, None, a, b))
        else:
            lower, upper = switch_node_bounds(rows, (tri_vertices[t] for t in node.tris), scale)
            result.append(TreeNode((node.right << 6) | node.axis, lower, upper, None, None))
    return result


def build_collision_mesh(positions, triangles, materials=0, flags=0, build: str = "switch", *,
                         mesh_flags: int = 0, mesh_material: int = 0) -> CollisionMesh:
    """W14-12 to W14-15: a collision mesh from positions and triangles (vertex index triples).
    ``materials`` and ``flags`` are per triangle (a sequence) or one value for all. The table
    order is the leaf order; ``mesh.triangles`` are the used entries in that order."""
    form = form_of(build)
    positions = [tuple(float(c) for c in p[:3]) for p in positions]
    tris = [tuple(int(v) for v in t) for t in triangles]
    count = len(tris)
    materials = _per_triangle(materials, count, "materials")
    flags = _per_triangle(flags, count, "flags")
    for t in tris:
        if len(t) != 3 or not all(0 <= v < len(positions) for v in t):
            raise FormatError(f"triangle {t} of {len(positions)} vertices")
    if form == "ps3" and len(positions) > 0xFFFF:
        raise FormatError(f"{len(positions)} vertices: ps3 indices are 16-bit (0xFFFF is 'none')")
    limit = 1 << (30 if form == "ps3" else 26)
    if count + 3 >= limit:
        raise FormatError(f"{count} triangles do not fit the node word")
    centre, rows, half, scale = mesh_bounds(positions, build)
    built, order = build_tree(rows, tris)
    neighbours = triangle_neighbours(tris)
    table = [Triangle(tris[t], neighbours[t], materials[t], flags[t]) for t in order]
    used = list(table)
    if form == "ps3" and table:
        table += [table[-1]] * 3
    nodes = tree_nodes(built, tris, rows, scale, build)
    height = max((n.depth for n in built), default=0)
    stored_positions = [(r[0] + centre[0], r[1] + centre[1], r[2] + centre[2]) for r in rows]
    return CollisionMesh(centre, half, scale, height, count, mesh_flags, mesh_material,
                         stored_positions, table, nodes, used, rows)


def new_collision_file(meshes: list[CollisionMesh], build: str, handle: int = 0, flags: int = 0,
                       revision: int = 0, source_hash: int = 0) -> CollisionFile:
    """A collision file model for ``write_collision`` (header version from the version table)."""
    version = expected_version("phm", build) or 0
    header = SnoHeader(build_order(build), SNO_SIGNATURE, version, revision, source_hash, handle, 0, 0,
                       given_build=build)
    return CollisionFile(header, build, flags, list(meshes), [])


# --- derive -------------------------------------------------------------------------------------

def _stored_rows(mesh: CollisionMesh) -> list[tuple[float, float, float, float]]:
    """The vertex rows to store: the parsed rows while ``positions`` still match them."""
    rows = mesh.vertices
    c = mesh.centre
    if rows is not None and len(rows) == len(mesh.positions) and all(
        (r[0] + c[0], r[1] + c[1], r[2] + c[2]) == tuple(p) for r, p in zip(rows, mesh.positions)
    ):
        return [tuple(r) for r in rows]
    return [(f32(p[0] - c[0]), f32(p[1] - c[1]), f32(p[2] - c[2]), 0.0) for p in mesh.positions]


def derive_mesh(mesh: CollisionMesh, build: str, derive, source: CollisionMesh | None = None) -> CollisionMesh:
    """Apply ``derive`` (False, "proven" or True) to one mesh; returns a new mesh. ``source`` is
    the mesh as parsed: with "proven", non-manifold neighbour slots of unchanged triangles keep
    their stored value (V16-3); True writes -1 there, as W14-15."""
    if derive is False or derive is None:
        return mesh
    form = form_of(build)
    if derive is True:
        return build_collision_mesh(
            mesh.positions, [t.vertices for t in mesh.triangles], [t.material for t in mesh.triangles],
            [t.flags for t in mesh.triangles], build, mesh_flags=mesh.flags, mesh_material=mesh.material,
        )
    if derive != "proven":
        raise ValueError(f"derive must be False, 'proven' or True, not {derive!r}")
    # PROVEN values on the stored tree and stored bounds.
    rows = _stored_rows(mesh)
    count = mesh.triangle_count
    used = list(mesh.table[:count])
    if source is None:
        neighbours = triangle_neighbours([t.vertices for t in used])
    else:
        old = source.table[:source.triangle_count]
        neighbours = proven_neighbours([t.vertices for t in used], mesh.positions,
                                       [t.vertices for t in old], source.positions,
                                       [t.neighbours for t in old])
    used = [Triangle(t.vertices, n, t.material, t.flags) for t, n in zip(used, neighbours)]
    table = list(used)
    if form == "ps3" and table:
        table += [table[-1]] * 3
    nodes = list(mesh.nodes)
    if form == "ps3" and nodes:
        runs = leaf_runs(nodes, build)
        for i, node in enumerate(nodes):
            if node.is_leaf:
                continue
            left = [used[e].vertices for e in runs[i + 1]]
            right = [used[e].vertices for e in runs[i + (node.data >> 2)]]
            a, b = ps3_split_values(rows, left, right, node.axis, mesh.box_scale)
            nodes[i] = TreeNode(node.data, None, None, a, b)
    return CollisionMesh(mesh.centre, mesh.half_extent, mesh.box_scale, tree_height(nodes, build), count,
                         mesh.flags, mesh.material, list(mesh.positions), table, nodes, used, rows)


# --- writer -------------------------------------------------------------------------------------

def _encode_triangles(table: Sequence[Triangle], form: str, order: str) -> bytes:
    out = bytearray()
    for t in table:
        if form == "ps3":
            n = tuple(0xFFFF if v == -1 else v for v in t.neighbours)
            out += struct.pack(order + "6HHH", *t.vertices, *n, t.material, t.flags)
        else:
            out += struct.pack(order + "6iHH", *t.vertices, *t.neighbours, t.material, t.flags)
    return bytes(out)


def _encode_nodes(nodes: Sequence[TreeNode], form: str, order: str) -> bytes:
    out = bytearray()
    for n in nodes:
        if form == "ps3":
            out += struct.pack(order + "IHH", n.data, n.a, n.b)
        else:
            out += struct.pack(order + "6hI", *n.lower, *n.upper, n.data)
    return bytes(out)


def _set_array(record: L.Record, slot: L.PairSlot, data: bytes, build: str) -> None:
    """Give a record-array pair new bytes; an unchanged block keeps its PairValue (sharing)."""
    old = record.pairs.get(slot.offset)
    if old is not None and old.block is not None and old.size is None and _block_bytes(old.block) == data:
        return
    if not data:
        record.pairs[slot.offset] = L.PairValue(None)
        return
    size = L.record_layout(build, "phm", slot.field.element).size
    if len(data) % size:
        raise FormatError(f"{slot.path}: {len(data)} bytes is not a whole number of {size}")
    elements = [L.Record(slot.field.element, bytearray(data[i:i + size])) for i in range(0, len(data), size)]
    record.pairs[slot.offset] = L.PairValue(L.Block(elements=elements))


def _block_bytes(block: L.Block) -> bytes:
    if block.elements is None:
        return bytes(block.data or b"")
    return b"".join(bytes(e.raw) for e in block.elements) + block.extra


def _template(collision: CollisionFile, build: str) -> L.SnoTree:
    if collision.source is not None:
        return L.parse_tree(collision.source, "phm", build)
    top = L.new_record(build, "phm", L.top_record(build, "phm"))
    return L.SnoTree(build, "phm", bytes(L.HEADER_SIZE), top)


def write_collision(collision: CollisionFile, derive=False) -> bytes:
    """Rebuild a ``.phm`` from its model with the W11 tree writer.

    ``derive``: False (default) stores every value of the model as it is (an unchanged canonical
    file is rebuilt byte for byte); "proven" recomputes counts, neighbours (keeping the stored
    value of a non-manifold slot of an unchanged triangle, V16-3), the ps3 table tail, ps3 split
    values and the tree height on the stored tree; True rebuilds every mesh with
    ``build_collision_mesh`` (W14-12 to W14-15, including the SUPPORTED bounds and splits)."""
    build = collision.build
    form = form_of(build)
    order = build_order(build)
    tree = _template(collision, build)
    h = collision.header
    tree.header = struct.pack(order + "4I", h.signature, h.version, h.revision, h.source_hash)
    top = tree.top
    struct.pack_into(order + "3I", top.raw, 0, h.handle, h.runtime_1, h.runtime_2)
    struct.pack_into(order + "I", top.raw, 0x0C, collision.flags)
    struct.pack_into(order + "i", top.raw, 0x10, len(collision.meshes))
    (mesh_slot,) = [s for s in L.pair_slots(build, "phm", top.name) if s.offset == 0x14]
    old_value = top.pairs.get(0x14)
    old_records = list(old_value.block.elements) if old_value is not None and old_value.block is not None else []
    mesh_name = mesh_slot.field.element
    slots = {s.offset: s for s in L.pair_slots(build, "phm", mesh_name)}
    f = _FIELDS[build]
    records = []
    sources = []
    if derive == "proven" and collision.source is not None:
        from .collision import parse_collision
        sources = parse_collision(collision.source, build).meshes
    for i, mesh in enumerate(collision.meshes):
        mesh = derive_mesh(mesh, build, derive, sources[i] if i < len(sources) else None)
        record = old_records[i] if i < len(old_records) else L.new_record(build, "phm", mesh_name)
        raw = record.raw
        rows = _stored_rows(mesh)
        struct.pack_into(order + "3f", raw, f["centre"], *mesh.centre)
        struct.pack_into(order + "3f", raw, f["half"], *mesh.half_extent)
        struct.pack_into(order + "3f", raw, f["scale"], *mesh.box_scale)
        struct.pack_into(order + "i", raw, f["nodes"], len(mesh.nodes))
        struct.pack_into(order + "i", raw, f["vertices"], len(rows))
        struct.pack_into(order + "i", raw, f["triangles"], mesh.triangle_count)
        if form == "ps3":
            struct.pack_into(order + "i", raw, f["entries"], len(mesh.table))
        struct.pack_into(order + "i", raw, f["height"], mesh.tree_height)
        struct.pack_into(order + "I", raw, f["flags"], mesh.flags)
        struct.pack_into(order + "I", raw, f["material"], mesh.material)
        _set_array(record, slots[f["vertex_pair"]], b"".join(struct.pack(order + "4f", *r) for r in rows), build)
        _set_array(record, slots[f["triangle_pair"]], _encode_triangles(mesh.table, form, order), build)
        _set_array(record, slots[f["node_pair"]], _encode_nodes(mesh.nodes, form, order), build)
        records.append(record)
    if records:
        if not (old_value is not None and old_value.block is not None
                and len(old_records) == len(records) and all(a is b for a, b in zip(old_records, records))):
            top.pairs[0x14] = L.PairValue(L.Block(elements=records))
    else:
        top.pairs[0x14] = L.PairValue(None)
    return L.write_tree(tree)
