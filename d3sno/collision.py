"""``.phm`` collision meshes."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair
from .sno import SnoHeader, parse_header
from .transform import Vector

LEAF = 3
_MESH_SIZE = {"switch": 112, "ps3": 136}
_NODE_SIZE = {"switch": 16, "ps3": 8}
_TRIANGLE_SIZE = {"switch": 28, "ps3": 16}
_VERTEX_SIZE = 16
# Field positions in a mesh record.
_FIELDS = {
    "switch": {
        "centre": 0x18, "half": 0x24, "scale": 0x30, "nodes": 0x3C, "vertices": 0x40,
        "triangles": 0x44, "height": 0x48, "vertex_pair": 0x50, "triangle_pair": 0x58,
        "node_pair": 0x60, "flags": 0x68, "material": 0x6C,
    },
    "ps3": {
        "centre": 0x20, "half": 0x2C, "scale": 0x38, "nodes": 0x44, "vertices": 0x48,
        "triangles": 0x4C, "entries": 0x54, "height": 0x5C, "vertex_pair": 0x60,
        "triangle_pair": 0x68, "node_pair": 0x78, "flags": 0x80, "material": 0x84,
    },
}


@dataclass(frozen=True)
class Triangle:
    vertices: tuple[int, int, int]
    neighbours: tuple[int, int, int]  # -1 for none
    material: int
    flags: int


@dataclass(frozen=True)
class TreeNode:
    data: int
    lower: tuple[int, int, int] | None  # switch
    upper: tuple[int, int, int] | None  # switch
    a: int | None  # ps3
    b: int | None  # ps3

    @property
    def axis(self) -> int:
        return self.data & 3

    @property
    def is_leaf(self) -> bool:
        return self.data & 3 == LEAF


@dataclass
class CollisionMesh:
    centre: Vector
    half_extent: Vector
    box_scale: Vector
    tree_height: int
    triangle_count: int
    flags: int
    material: int
    positions: list[Vector]
    table: list[Triangle]
    nodes: list[TreeNode]
    triangles: list[Triangle]  # in use; on ps3 these come from the tree


@dataclass
class CollisionFile:
    header: SnoHeader
    build: str
    flags: int
    meshes: list[CollisionMesh]
    warnings: list[str] = field(default_factory=list)


def _read_triangles(raw: bytes, build: str, order: str) -> list[Triangle]:
    result = []
    if build == "switch":
        for row in struct.iter_unpack(order + "6iHH", raw):
            result.append(Triangle(row[:3], row[3:6], row[6], row[7]))
    else:
        for row in struct.iter_unpack(order + "6HHH", raw):
            neighbours = tuple(-1 if value == 0xFFFF else value for value in row[3:6])
            result.append(Triangle(row[:3], neighbours, row[6], row[7]))
    return result


def _read_nodes(raw: bytes, build: str, order: str) -> list[TreeNode]:
    if build == "switch":
        return [
            TreeNode(row[6], row[:3], row[3:6], None, None)
            for row in struct.iter_unpack(order + "6hI", raw)
        ]
    return [
        TreeNode(row[0], None, None, row[1], row[2])
        for row in struct.iter_unpack(order + "IHH", raw)
    ]


def _walk_tree(nodes: list[TreeNode], build: str, position: int) -> list[int]:
    """Visit the tree and return the triangle index of every leaf slot, in leaf order."""
    shift = 6 if build == "switch" else 2
    leaves: list[int] = []
    seen: set[int] = set()
    pending = [0] if nodes else []
    while pending:
        index = pending.pop()
        if index >= len(nodes):
            raise FormatError(f"tree node {index} of {len(nodes)}", position)
        if index in seen:
            raise FormatError(f"tree node {index} is reached twice", position)
        seen.add(index)
        node = nodes[index]
        if node.is_leaf:
            first = node.data >> shift
            count = (node.data >> 2) & 15 if build == "switch" else node.a
            leaves.extend(range(first, first + count))
        else:
            right = node.data >> shift
            if right < 2:
                raise FormatError(f"tree node {index} has right child offset {right}", position)
            pending.append(index + right)
            pending.append(index + 1)
    return leaves


def _sized(view: ByteView, record: ByteView, at: int, base: int, size: int, what: str) -> bytes:
    pair = record.pair(at, what)
    if pair.size != size:
        raise FormatError(f"{what}: pair size {pair.size}, expected {size}", base + at)
    return view.pair_data(pair, base + at, what)


def _read_mesh(view: ByteView, raw: bytes, build: str, base: int, warnings: list[str], index: int) -> CollisionMesh:
    f = _FIELDS[build]
    r = ByteView(raw, view.order)
    node_count = r.i32(f["nodes"])
    vertex_count = r.i32(f["vertices"])
    triangle_count = r.i32(f["triangles"])
    entries = r.i32(f["entries"]) if build == "ps3" else triangle_count
    if min(node_count, vertex_count, triangle_count, entries) < 0:
        raise FormatError("negative count in a collision mesh", base)
    if entries < triangle_count:
        raise FormatError(f"{entries} table entries for {triangle_count} triangles", base)
    centre = r.f32s(f["centre"], 3)
    vertices = _sized(view, r, f["vertex_pair"], base, vertex_count * _VERTEX_SIZE, "vertices")
    positions = [
        (row[0] + centre[0], row[1] + centre[1], row[2] + centre[2])
        for row in struct.iter_unpack(view.order + "4f", vertices)
    ]
    table_raw = _sized(view, r, f["triangle_pair"], base, entries * _TRIANGLE_SIZE[build], "triangles")
    table = _read_triangles(table_raw, build, view.order)
    nodes_raw = _sized(view, r, f["node_pair"], base, node_count * _NODE_SIZE[build], "nodes")
    nodes = _read_nodes(nodes_raw, build, view.order)

    leaves = _walk_tree(nodes, build, base)
    if any(index >= entries for index in leaves):
        raise FormatError(f"leaf triangle past {entries} table entries", base)
    if len(set(leaves)) != len(leaves):
        warnings.append(f"mesh {index}: a triangle lies in more than one leaf")
    if len(set(leaves)) != triangle_count:
        warnings.append(f"mesh {index}: leaves hold {len(set(leaves))} of {triangle_count} triangles")
    triangles = table if build == "switch" else [table[i] for i in dict.fromkeys(leaves)]
    for triangle in triangles:
        if not all(0 <= v < vertex_count for v in triangle.vertices):
            raise FormatError(f"triangle vertex {triangle.vertices} of {vertex_count}", base)
    return CollisionMesh(
        centre, r.f32s(f["half"], 3), r.f32s(f["scale"], 3), r.i32(f["height"]), triangle_count,
        r.u32(f["flags"]), r.u32(f["material"]), positions, table, nodes, triangles,
    )


def parse_collision(data: bytes, build: str | None = None, strict: bool = False) -> CollisionFile:
    """Read a ``.phm``. A triangle not in exactly one leaf is a warning, unless ``strict``."""
    header = parse_header(data, build)
    resolved = header.build
    view = ByteView(data, header.order)
    count = view.i32(0x20, "mesh count")
    pair = view.pair(0x24, "meshes pair")
    size = _MESH_SIZE[resolved]
    if count < 0 or pair.size != count * size:
        raise FormatError(f"meshes of {pair.size} bytes for {count}", 0x24)
    raw = view.pair_data(pair, 0x24, "meshes")
    warnings: list[str] = []
    meshes = [
        _read_mesh(view, raw[i * size:(i + 1) * size], resolved,
                   pair.file_offset + i * size, warnings, i)
        for i in range(count)
    ]
    if strict and warnings:
        raise FormatError(warnings[0], 0x24)
    return CollisionFile(header, resolved, view.u32(0x1C), meshes, warnings)


def to_obj(collision: CollisionFile) -> str:
    lines = ["# d3sno collision export"]
    base = 1
    for number, mesh in enumerate(collision.meshes):
        lines.append(f"o mesh_{number}")
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh.positions)
        for triangle in mesh.triangles:
            a, b, c = (v + base for v in triangle.vertices)
            lines.append(f"f {a} {b} {c}")
        base += len(mesh.positions)
    return "\n".join(lines) + "\n"
