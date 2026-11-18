"""The visual-mesh octree of an ``.app`` (``26_corrections15.md`` W13-3): layout, primitives,
counts, subdivision and the root cube of a changed mesh.

Layout (all builds): the six counts are the u32 at body 0xE0 (nMaxNodes, nMaxLeaves,
nMaxPrimitives, then the node, leaf and primitive counts); the pairs are nodes (body 0xF8, 48-byte
records: cube low corner, cube size, 8 children), leaves (body 0x108, 24 bytes: cube, first
primitive, primitive count) and primitives (body 0x118, 8 bytes: user data, index-list position).

The builder works from a parsed mesh (an ``AppearanceFile``). Exact leaf membership of a game file
is not derivable (the original was built from unquantised positions); a built tree is a safe
replacement in which every primitive is in every leaf whose cube it overlaps.

``27_corrections16.md``: child cubes are float32 (V15-1); a leaf's primitives are sorted by
(sub-object, position counted across the sub-object's draws end to end) (V15-2); a reader accepts
the second primitive form (user data = the sub-object alone, position counted end to end) (V15-3).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from . import sno_layout as L
from .binary import FormatError

FLAG_OCTREE = 0x1000          # a geo-set-0 sub-object with this flag gives primitives
LEAF_BIT = 0x80000000          # a child that is a leaf index
SPLIT_LIMIT = 256              # a child becomes a node above this many primitives ...
MIN_NODE_SIZE = 2.0            # ... and when its size is at least this
LOW_WIDEN = 3e-5               # the low side of a cube is widened by this x size
COUNTS_AT = 0x10 + 0xD0        # body 0xE0
PAIRS_AT = (COUNTS_AT + 0x18, COUNTS_AT + 0x28, COUNTS_AT + 0x38)  # body 0xF8, 0x108, 0x118
RECORD_NAMES = ("OctreeNode", "OctreeLeaf", "OctreePrimitive")
NODE_SIZE, LEAF_SIZE, PRIMITIVE_SIZE = 48, 24, 8


@dataclass
class Octree:
    """``nodes``: (low corner, size, 8 children); ``leaves``: (low corner, size, first, count);
    ``primitives``: (user data, index-list position). ``maxima``: nMaxNodes, nMaxLeaves,
    nMaxPrimitives. ``stored_counts``: the node, leaf and primitive counts as read from a file
    (None for a built tree, whose counts are its list lengths)."""

    nodes: list = field(default_factory=list)
    leaves: list = field(default_factory=list)
    primitives: list = field(default_factory=list)
    maxima: tuple = (0, 0, 0)
    stored_counts: tuple | None = None

    @property
    def counts(self) -> tuple[int, int, int, int, int, int]:
        return (*self.maxima, len(self.nodes), len(self.leaves), len(self.primitives))

    @property
    def root(self) -> tuple[tuple[float, float, float], float] | None:
        return (tuple(self.nodes[0][0]), self.nodes[0][1]) if self.nodes else None


def f32(value: float) -> float:
    """V15-1: ``value`` rounded to float32 (as stored)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_cube(low, size) -> tuple[tuple[float, float, float], float]:
    return tuple(f32(v) for v in low), f32(size)


def child_cube(low, size: float, k: int) -> tuple[tuple[float, float, float], float]:
    """V15-1: child ``k`` (bit 2: x, bit 1: y, bit 0: z) of the float32 cube (low, size):
    corner = f32(corner + offset x f32(size / 2)), size = f32(size / 2)."""
    half = f32(size / 2)
    offsets = (k >> 2 & 1, k >> 1 & 1, k & 1)
    return tuple(f32(low[a] + offsets[a] * half) for a in range(3)), half


# --- Draws and triangles ---------------------------------------------------------------------------

def octree_draws(sub) -> list:
    """W13-3: the draws of a sub-object in octree order: its ps3 draws, or on Switch the packed
    list and each skinning record in order; when it has neither, its fat list."""
    draws = [d for d in sub.draws if d.kind != "fat"]
    if draws:
        return draws
    if sub.fat_draw is not None:
        return [sub.fat_draw]
    return [d for d in sub.draws if d.kind == "fat"][:1]


def geo0_triangle_count(app) -> int:
    """nMaxNodes = nMaxLeaves: the geo-set-0 triangles of all draws, repeated indices included."""
    return sum(len(d.indices) // 3 for sub in app.sub_objects for d in octree_draws(sub))


def octree_triangles(app) -> list[tuple[int, int, tuple]]:
    """W13-3 primitives: (user data, index-list position, three corners) for every triangle of
    every draw of the flagged geo-set-0 sub-objects, except triangles that repeat an index.
    user data = sub-object index + 65536 x draw index; position = 3 x triangle number."""
    out = []
    for sub in app.sub_objects:
        if not sub.flags & FLAG_OCTREE:
            continue
        for k, draw in enumerate(octree_draws(sub)):
            idx = draw.indices
            for t in range(len(idx) // 3):
                a, b, c = idx[3 * t:3 * t + 3]
                if a == b or b == c or a == c:
                    continue
                corners = (tuple(draw.positions[a]), tuple(draw.positions[b]), tuple(draw.positions[c]))
                out.append((sub.index + 65536 * k, 3 * t, corners))
    return out


def draw_offsets(app) -> dict[int, list[int]]:
    """V15-2: per sub-object index, the start of each of its octree draws in the sub-object's
    index lists taken end to end, followed by their total."""
    out = {}
    for sub in app.sub_objects:
        starts, at = [], 0
        for draw in octree_draws(sub):
            starts.append(at)
            at += len(draw.indices)
        starts.append(at)
        out[sub.index] = starts
    return out


def primitive_rank(offsets, user: int, position: int) -> tuple[int, int]:
    """V15-2: the sort key of a primitive in either form: (sub-object, position counted across the
    sub-object's draws end to end). The draw number in the user data counts only through the
    start of its draw."""
    sub, draw = user & 0xFFFF, user >> 16
    starts = offsets.get(sub)
    if starts is None or draw >= len(starts) - 1:
        return sub, position
    return sub, starts[draw] + position


def canonical_primitive(offsets, user: int, position: int) -> tuple[int, int]:
    """V15-3: a primitive in the form of W13-3 (user data = sub-object + 65536 x draw, position
    within the draw). The second form (user data = the sub-object alone, position counted end to
    end, past the first draw) is mapped to it; anything else is returned unchanged."""
    sub, draw = user & 0xFFFF, user >> 16
    starts = offsets.get(sub)
    if draw or starts is None:
        return user, position
    for k in range(len(starts) - 2, 0, -1):
        if position >= starts[k]:
            return sub + 65536 * k, position - starts[k]
    return user, position


# --- Overlap -----------------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def triangle_overlaps_cube(corners, low, size: float) -> bool:
    """W13-3: separating-axis test of a triangle against the cube [low, low + size) per axis, the
    low side widened by 3e-5 x size (half-open on the high side)."""
    lo = [low[k] - LOW_WIDEN * size for k in range(3)]
    hi = [low[k] + size for k in range(3)]
    for k in range(3):
        values = [p[k] for p in corners]
        if max(values) < lo[k] or min(values) >= hi[k]:
            return False
    centre = [(lo[k] + hi[k]) / 2 for k in range(3)]
    half = [(hi[k] - lo[k]) / 2 for k in range(3)]
    v = [_sub(p, centre) for p in corners]
    edges = (_sub(v[1], v[0]), _sub(v[2], v[1]), _sub(v[0], v[2]))
    units = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for edge in edges:
        for unit in units:
            axis = _cross(unit, edge)
            p = [_dot(x, axis) for x in v]
            r = sum(half[k] * abs(axis[k]) for k in range(3))
            if min(p) > r or max(p) < -r:
                return False
    normal = _cross(edges[0], edges[1])
    r = sum(half[k] * abs(normal[k]) for k in range(3))
    return abs(_dot(normal, v[0])) <= r


# --- Root cube ---------------------------------------------------------------------------------

def triangles_box(triangles) -> tuple[list[float], list[float]] | None:
    points = [p for _, _, corners in triangles for p in corners]
    if not points:
        return None
    return [min(p[k] for p in points) for k in range(3)], [max(p[k] for p in points) for k in range(3)]


def power_of_two_at_least(value: float) -> float:
    """The smallest power of two (2^k, k may be negative) not below ``value``; 1 for 0."""
    if value <= 0:
        return 1.0
    size = 1.0
    while size < value:
        size *= 2
    while size / 2 >= value:
        size /= 2
    return size


def derived_root(triangles) -> tuple[tuple[float, float, float], float]:
    """W13-3 root for a changed mesh: size = the smallest power of two at least the largest extent
    of the primitives' box, centred on that box."""
    box = triangles_box(triangles)
    if box is None:
        return (0.0, 0.0, 0.0), 1.0
    low, high = box
    size = power_of_two_at_least(max(high[k] - low[k] for k in range(3)))
    return f32_cube(tuple((low[k] + high[k]) / 2 - size / 2 for k in range(3)), size)


def root_holds(root, triangles) -> bool:
    """The primitives' box lies inside the half-open root cube."""
    box = triangles_box(triangles)
    if box is None:
        return True
    (corner, size), (low, high) = root, box
    return all(corner[k] <= low[k] and high[k] < corner[k] + size for k in range(3))


# --- Builder -----------------------------------------------------------------------------------

def build_octree(app, root=None, triangles=None) -> Octree:
    """W13-3: the octree of a parsed mesh. ``root`` is (low corner, size); by default the derived
    root. The root is rounded to float32 and every child cube is float32 (V15-1). Nodes are
    numbered depth-first in pre-order (root 0), leaves in walk order; each leaf's primitives are
    consecutive and sorted by (sub-object, position across its draws) (V15-2)."""
    if triangles is None:
        triangles = octree_triangles(app)
    if root is None:
        root = derived_root(triangles)
    tree = Octree()
    by_key = {(user, position): corners for user, position, corners in triangles}
    offsets = draw_offsets(app)
    keys = sorted(by_key, key=lambda key: primitive_rank(offsets, *key))

    def node(low, size, candidates) -> int:
        index = len(tree.nodes)
        tree.nodes.append(None)
        children = []
        for k in range(8):
            child, half = child_cube(low, size, k)
            inside = [key for key in candidates if triangle_overlaps_cube(by_key[key], child, half)]
            if len(inside) > SPLIT_LIMIT and half >= MIN_NODE_SIZE:
                children.append(node(child, half, inside))
            else:
                children.append(LEAF_BIT | len(tree.leaves))
                tree.leaves.append((child, half, len(tree.primitives), len(inside)))
                tree.primitives.extend(inside)
        tree.nodes[index] = (tuple(low), size, tuple(children))
        return index

    node(*f32_cube(root[0], root[1]), keys)
    total = geo0_triangle_count(app)
    rule = (total, total, 2 * total if 2 * total >= len(tree.primitives) else 4 * total)
    tree.maxima = raised_maxima(rule, tree)
    return tree


def raised_maxima(maxima, tree: Octree) -> tuple[int, int, int]:
    """V15-5: ``maxima`` raised to the node, leaf and primitive counts where those are larger."""
    lengths = (len(tree.nodes), len(tree.leaves), len(tree.primitives))
    return tuple(max(int(m), n) for m, n in zip(maxima, lengths))


def _canonical(tree: Octree, app):
    if app is None:
        return [tuple(p) for p in tree.primitives]
    offsets = draw_offsets(app)
    return [canonical_primitive(offsets, *p) for p in tree.primitives]


def missing_primitives(tree: Octree, triangles, app=None) -> list[tuple[int, int]]:
    """Primitives that are in no leaf (a primitive outside the root cube). With ``app``, stored
    primitives of the second form are accepted (V15-3)."""
    held = set(_canonical(tree, app))
    return sorted({(u, p) for u, p, _ in triangles} - held)


def leaf_problems(tree: Octree, triangles, app=None) -> list[str]:
    """Every leaf's primitives overlap its cube as stored, in float32 (W13-3, V15-1). With
    ``app``, primitives of the second form are accepted (V15-3)."""
    by_key = {(u, p): c for u, p, c in triangles}
    primitives = _canonical(tree, app)
    found = []
    for n, (low, size, first, count) in enumerate(tree.leaves):
        low, size = f32_cube(low, size)
        for key in primitives[first:first + count]:
            if key not in by_key or not triangle_overlaps_cube(by_key[key], low, size):
                found.append(f"leaf {n}: primitive {key} does not overlap its cube")
    return found


# --- Encoding and reading --------------------------------------------------------------------------

def _check(value, code, what):
    from .app_mesh import check_float, check_word
    return check_float(value, what) if code == "f" else check_word(value, code, what)


def encode_octree(tree: Octree, order: str) -> tuple[bytes, list[bytes], list[bytes], list[bytes]]:
    """The six counts and the node, leaf and primitive records in the byte order ``order``."""
    for k, value in enumerate(tree.counts):
        _check(value, "I", f"octree count {k}")
    counts = struct.pack(order + "6I", *tree.counts)
    nodes, leaves, prims = [], [], []
    for low, size, children in tree.nodes:
        for v in (*low, size):
            _check(v, "f", "octree node cube")
        nodes.append(struct.pack(order + "4f8I", *low, size, *children))
    for low, size, first, count in tree.leaves:
        for v in (*low, size):
            _check(v, "f", "octree leaf cube")
        leaves.append(struct.pack(order + "4f2I", *low, size, first, count))
    for user, position in tree.primitives:
        _check(user, "I", "octree primitive user data")
        _check(position, "I", "octree primitive position")
        prims.append(struct.pack(order + "2I", user, position))
    return counts, nodes, leaves, prims


def set_octree(tree: L.SnoTree, octree: Octree | None) -> None:
    """Store ``octree`` in an ``.app`` structure tree as new blocks (W2-14); None stores the
    empty form (six zero counts, empty pairs)."""
    order = "<" if tree.build == "switch" else ">"
    top = tree.top
    if octree is None:
        top.raw[COUNTS_AT:COUNTS_AT + 24] = bytes(24)
        for at in PAIRS_AT:
            top.pairs[at] = L.PairValue(None)
        return
    counts, *lists = encode_octree(octree, order)
    top.raw[COUNTS_AT:COUNTS_AT + 24] = counts
    for at, name, records in zip(PAIRS_AT, RECORD_NAMES, lists):
        elements = []
        for raw in records:
            element = L.new_record(tree.build, "app", name)
            element.raw[:] = raw
            elements.append(element)
        top.pairs[at] = L.PairValue(L.Block(elements=elements) if elements else None)


def read_octree(tree: L.SnoTree) -> Octree | None:
    """The stored octree of an ``.app`` structure tree; None when it has no nodes."""
    order = "<" if tree.build == "switch" else ">"
    top = tree.top
    counts = struct.unpack_from(order + "6I", top.raw, COUNTS_AT)

    def records(at):
        value = top.pairs.get(at)
        if value is None or value.block is None:
            return []
        block = value.block
        if block.elements is not None:
            return [bytes(e.raw) for e in block.elements]
        return []

    nodes_raw, leaves_raw, prims_raw = (records(at) for at in PAIRS_AT)
    if not nodes_raw:
        return None
    out = Octree(maxima=counts[:3], stored_counts=counts[3:])
    for raw in nodes_raw:
        values = struct.unpack(order + "4f8I", raw)
        out.nodes.append((values[:3], values[3], values[4:]))
    for raw in leaves_raw:
        values = struct.unpack(order + "4f2I", raw)
        out.leaves.append((values[:3], values[3], values[4], values[5]))
    for raw in prims_raw:
        out.primitives.append(struct.unpack(order + "2I", raw))
    return out


def octree_problems(app, tree: Octree | None) -> list[str]:
    """W13-3 checks of a (stored or built) octree against a parsed mesh: flag 0x1000 and the
    presence of primitives, the node numbering and the leaf order of V15-2 (either primitive
    form, V15-3). V15-5: each maximum must be at least its count and stored counts must equal the
    list lengths; a maximum that differs from the W13-3 rule is not a problem."""
    found = []
    offsets = draw_offsets(app)
    flagged = any(s.flags & FLAG_OCTREE for s in app.sub_objects)
    if tree is None:
        if flagged and octree_triangles(app):
            found.append("a flagged geo-set-0 sub-object but no octree (W13-3)")
        return found
    if bool(tree.primitives) != (flagged and bool(octree_triangles(app))):
        found.append("primitives present exactly when a geo-set-0 sub-object has flag 0x1000 (W13-3)")
    lengths = (len(tree.nodes), len(tree.leaves), len(tree.primitives))
    for name, maximum, length in zip(("nMaxNodes", "nMaxLeaves", "nMaxPrimitives"), tree.maxima, lengths):
        if maximum < length:
            found.append(f"{name} {maximum} is below the count {length} (V15-5)")
    if tree.stored_counts is not None and tuple(tree.stored_counts) != lengths:
        found.append(f"counts {tuple(tree.stored_counts)} are not the list lengths {lengths} (V15-5)")
    for n, (_, _, children) in enumerate(tree.nodes):
        for child in children:
            if child & LEAF_BIT:
                if child & ~LEAF_BIT >= len(tree.leaves):
                    found.append(f"node {n}: leaf {child & ~LEAF_BIT} past {len(tree.leaves)}")
            elif not n < child < len(tree.nodes):
                found.append(f"node {n}: child node {child} is not after it (pre-order)")
    for n, (_, _, first, count) in enumerate(tree.leaves):
        if first + count > len(tree.primitives):
            found.append(f"leaf {n}: primitives past the list")
        ranks = [primitive_rank(offsets, *key) for key in tree.primitives[first:first + count]]
        if any(a >= b for a, b in zip(ranks, ranks[1:])):
            found.append(f"leaf {n}: primitives not sorted by (sub-object, position across its draws)")
    return found


__all__ = [
    "FLAG_OCTREE", "LEAF_BIT", "Octree", "build_octree", "canonical_primitive", "child_cube",
    "derived_root", "draw_offsets", "encode_octree", "f32", "f32_cube", "primitive_rank",
    "raised_maxima",
    "geo0_triangle_count", "leaf_problems", "missing_primitives", "octree_draws",
    "octree_problems", "octree_triangles", "power_of_two_at_least", "read_octree", "root_holds",
    "set_octree", "triangle_overlaps_cube", "triangles_box",
]
