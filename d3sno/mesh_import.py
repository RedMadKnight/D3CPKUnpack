"""Mesh import from glTF 2.0 and Wavefront OBJ onto a template ``.app`` (``34_modding_tools.md``
W16-6 to W16-11).

The import starts from the template file. Every value that equals what the exporter
(``appearance.to_gltf`` / ``to_obj``) writes for the template's value keeps the template's
stored value, so an export followed by an import gives the template byte for byte, and an edit
changes only what was edited. A changed mesh is written with
``app_mesh.write_app_mesh(template, mesh, derive=True, keep_skin_bytes=True)``.

Corrections 21 (``35_corrections21.md``): V21-1 sub-objects with cloth or fat vertices keep every
value; V21-2 skin edits keep the stored bytes of unchanged vertices; V21-3 emptying every
skinning record of a switch/x360 sub-object is refused; V21-4 ``extras.d3sno.no_indices``;
V21-5 ``<name>__sub<i>`` object names.

Stage 17 (``37_mesh_parts.md``): W17-12 new meshes and primitives become new sub-objects and new
geometry (editor copies included), W17-13 ``missing`` = keep | empty | remove (replaces the
V21-3 / Q699 refusals: emptying every part of a sub-object removes it), W17-14 the report lists
every added and removed part (``ImportReport.added`` / ``removed``).

Corrections 23 (``38_corrections23.md``): V23-3 a changed ps3 draw over a limit is refused (import
the geometry as a new mesh, which is cut); V23-6 an editor copy of a cloth or fat sub-object is
refused; V23-7 the import creates no material entry, and an ``extras.d3sno.material`` hash not in
the table gives a warning; V23-8 the weights of a new part (editor copies included) come from the
glTF floats quantised by W16-9 to a sum of 255, so on x360, where 12 % of stored skinned vertices
sum to 252-254, a copy's weights may differ from its source by 1 or 2 per vertex (unchanged parts
keep their stored bytes, so round trips are not affected); V23-9 a writer refusal names its draw
once.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import math
import os
import re
import struct
import urllib.parse
from dataclasses import dataclass, field
from fractions import Fraction

from . import app_mesh as M
from .app_parts import free_name, name_hash, ps3_name_field
from .appearance import (
    GEOSET_LOWPOLY, AppearanceFile, Draw, SubObject, _bone_node, _object_name, _unit, export_object_names,
    export_sub_objects,
    parse_appearance,
)
from .binary import FormatError

MAX_VERTICES = 65536  # W16-11 rule 5: indices are u16
QUANTIZATION = "KHR_mesh_quantization"
GLB_MAGIC = b"glTF"
GLB_JSON = 0x4E4F534A
GLB_BIN = 0x004E4942

_COMPONENTS = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_TYPES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_NORMALISED = {5120: 127, 5121: 255, 5122: 32767, 5123: 65535}
_INDEX_TYPES = (5121, 5123, 5125)
_IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
_DRAW_SUFFIX = re.compile(r"_draw_(\d+)$")
_COPY_SUFFIX = re.compile(r"^(.*)\.\d{3}$")
_BONE_NAME = re.compile(r"bone_(\d+)_([0-9A-Fa-f]{8})")
_REPEATED = re.compile(r"(.+)__sub(\d+)")  # V21-5
_TEXTURE_MATERIAL = re.compile(r"texture_(-?\d+)_draw_\d+")  # W16-5 material names (W17-12 rule 3)
LOWPOLY_PREFIX = "lowpoly_"  # G1-1 object names of geo set 1


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --- glTF reading (W16-6) ----------------------------------------------------------------------

def _element_layout(kind: str, size: int) -> tuple[list[int], int]:
    """Byte offsets of the components inside one element, and the element size. Matrix columns
    of 1- and 2-byte components start on 4-byte boundaries (glTF 2.0, data alignment)."""
    n = _TYPES[kind]
    if kind in ("MAT2", "MAT3") and size < 4 and not (kind == "MAT2" and size == 2):
        rows = 2 if kind == "MAT2" else 3
        column = (rows * size + 3) // 4 * 4
        offsets = [c * column + r * size for c in range(rows) for r in range(rows)]
        return offsets, column * rows
    return [k * size for k in range(n)], n * size


@dataclass
class GltfDocument:
    """A read glTF 2.0 document: its JSON and the bytes of each buffer."""
    json: dict
    buffers: list[bytes]

    def _get(self, key: str, index, what: str) -> dict:
        items = self.json.get(key)
        if not _is_int(index) or not isinstance(items, list) or not 0 <= index < len(items) \
                or not isinstance(items[index], dict):
            raise FormatError(f"glTF: {what} {index!r} does not exist")
        return items[index]

    def accessor_info(self, index) -> tuple[int, str, int, bool]:
        """(componentType, type, count, normalized) of an accessor."""
        acc = self._get("accessors", index, "accessor")
        return acc["componentType"], acc["type"], acc["count"], bool(acc.get("normalized", False))

    def accessor(self, index) -> list[tuple]:
        """The accessor's elements as tuples: ints, or floats for float and normalised
        integer accessors. Sparse substitution is applied."""
        acc = self._get("accessors", index, "accessor")
        comp, kind, count = acc["componentType"], acc["type"], acc["count"]
        code, size = _COMPONENTS[comp]
        offsets, element = _element_layout(kind, size)
        if "bufferView" in acc:
            raw, base, stride = self._view(acc["bufferView"], acc.get("byteOffset", 0), element)
            values = [tuple(struct.unpack_from("<" + code, raw, base + i * stride + o)[0] for o in offsets)
                      for i in range(count)]
        else:
            values = [tuple(0.0 if code == "f" else 0 for _ in offsets) for _ in range(count)]
        sparse = acc.get("sparse")
        if sparse is not None:
            n = sparse["count"]
            ind = sparse["indices"]
            icode, isize = _COMPONENTS[ind["componentType"]]
            raw, base, _ = self._view(ind["bufferView"], ind.get("byteOffset", 0), isize, stride_allowed=False)
            where = [struct.unpack_from("<" + icode, raw, base + k * isize)[0] for k in range(n)]
            val = sparse["values"]
            raw, base, _ = self._view(val["bufferView"], val.get("byteOffset", 0), element, stride_allowed=False)
            for k, at in enumerate(where):
                values[at] = tuple(struct.unpack_from("<" + code, raw, base + k * element + o)[0] for o in offsets)
        if acc.get("normalized"):
            divisor = _NORMALISED[comp]
            values = [tuple(max(c / divisor, -1.0) for c in row) for row in values]
        return values

    def _view(self, index, offset: int, element: int, stride_allowed: bool = True):
        view = self._get("bufferViews", index, "buffer view")
        stride = view.get("byteStride") if stride_allowed else None
        return self.buffers[view["buffer"]], view.get("byteOffset", 0) + offset, stride or element


def _uri_bytes(uri: str, base_path, what: str) -> bytes:
    if uri.startswith("data:"):
        header, comma, payload = uri[5:].partition(",")
        if not comma:
            raise FormatError(f"glTF: {what}: malformed data URI")
        if header.endswith(";base64"):
            try:
                return base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise FormatError(f"glTF: {what}: bad base64 data ({exc})") from None
        return urllib.parse.unquote_to_bytes(payload)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", uri) or uri.startswith(("/", "\\")):
        raise FormatError(f"glTF: {what}: URI {uri!r} is not relative")
    if base_path is None:
        raise FormatError(f"glTF: {what}: URI {uri!r} needs a base path")
    root = os.path.realpath(os.fspath(base_path))
    if os.path.isfile(root):
        root = os.path.dirname(root)
    target = os.path.realpath(os.path.join(root, urllib.parse.unquote(uri)))
    try:
        inside = os.path.commonpath([root, target]) == root
    except ValueError:
        inside = False
    if not inside or target == root:
        raise FormatError(f"glTF: {what}: URI {uri!r} leaves the folder {root}")
    try:
        with open(target, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise FormatError(f"glTF: {what}: cannot read {uri!r} ({exc.strerror})") from None


def _read_glb(data: bytes) -> tuple[bytes, bytes | None]:
    if len(data) < 12:
        raise FormatError("GLB: shorter than its 12-byte header", 0)
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC:
        raise FormatError("GLB: bad magic", 0)
    if version != 2:
        raise FormatError(f"GLB: version {version}, only 2 is read", 4)
    if length != len(data):
        raise FormatError(f"GLB: header length {length} for {len(data)} bytes", 8)
    chunks = []
    pos = 12
    while pos < length:
        if pos + 8 > length:
            raise FormatError("GLB: truncated chunk header", pos)
        size, kind = struct.unpack_from("<II", data, pos)
        if pos + 8 + size > length:
            raise FormatError(f"GLB: chunk of {size} bytes past the end", pos)
        chunks.append((kind, data[pos + 8:pos + 8 + size], pos))
        pos += 8 + size
    if not chunks or chunks[0][0] != GLB_JSON:
        raise FormatError("GLB: the first chunk is not JSON", 12)
    binary = chunks[1][1] if len(chunks) > 1 and chunks[1][0] == GLB_BIN else None
    for kind, _, at in chunks[1:]:
        if kind == GLB_JSON or (kind == GLB_BIN and at != chunks[1][2]):
            raise FormatError("GLB: a JSON or BIN chunk out of place", at)
    return chunks[0][1], binary


def _reject_constant(name):
    raise ValueError(f"{name} is not JSON")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise FormatError(f"glTF: {message}")


def _check_int(value, what: str, low: int = 0) -> int:
    _need(_is_int(value) and value >= low, f"{what} {value!r} is not an integer >= {low}")
    return value


def read_gltf(data, *, base_path=None) -> GltfDocument:
    """W16-6: a ``.gltf`` (JSON with data-URI buffers or buffers in files beside it, resolved
    against ``base_path``; a URI leaving that folder is refused) or a ``.glb``. Buffers, views,
    accessors (bounds, sparse) and primitives (mode 4, index range) are checked here;
    ``extensionsRequired`` may only name ``KHR_mesh_quantization``."""
    data = bytes(data) if not isinstance(data, str) else data.encode("utf-8")
    binary = None
    if data[:4] == GLB_MAGIC:
        text, binary = _read_glb(data)
    else:
        text = data
    try:
        document = json.loads(text.decode("utf-8-sig"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise FormatError(f"glTF: the JSON does not parse ({exc})") from None
    _need(isinstance(document, dict), "the JSON is not an object")
    asset = document.get("asset")
    _need(isinstance(asset, dict) and isinstance(asset.get("version"), str)
          and asset["version"].split(".")[0] == "2", "asset.version is not 2.x")
    required = document.get("extensionsRequired", [])
    _need(isinstance(required, list), "extensionsRequired is not a list")
    for name in required:
        _need(name == QUANTIZATION, f"required extension {name!r} is not supported")
    for key in ("buffers", "bufferViews", "accessors", "meshes", "nodes", "skins", "materials"):
        _need(isinstance(document.get(key, []), list), f"{key} is not a list")
        _need(all(isinstance(x, dict) for x in document.get(key, [])), f"an entry of {key} is not an object")
    buffers = _read_buffers(document, binary, base_path)
    doc = GltfDocument(document, buffers)
    _check_views(doc)
    for index in range(len(document.get("accessors", []))):
        _check_accessor(doc, index)
    _check_meshes(doc)
    return doc


def _read_buffers(document: dict, binary, base_path) -> list[bytes]:
    buffers = []
    for i, buffer in enumerate(document.get("buffers", [])):
        length = _check_int(buffer.get("byteLength"), f"buffer {i} byteLength")
        if "uri" in buffer:
            _need(isinstance(buffer["uri"], str), f"buffer {i}: uri is not a string")
            raw = _uri_bytes(buffer["uri"], base_path, f"buffer {i}")
        else:
            _need(i == 0 and binary is not None, f"buffer {i} has no uri and no GLB BIN chunk")
            raw = binary
            _need(len(raw) - length < 4, f"buffer 0: BIN chunk of {len(raw)} bytes for byteLength {length}")
        _need(len(raw) >= length, f"buffer {i}: {len(raw)} bytes for byteLength {length}")
        buffers.append(bytes(raw[:length]))
    return buffers


def _check_views(doc: GltfDocument) -> None:
    for i, view in enumerate(doc.json.get("bufferViews", [])):
        buffer = _check_int(view.get("buffer"), f"buffer view {i} buffer")
        _need(buffer < len(doc.buffers), f"buffer view {i}: buffer {buffer} does not exist")
        offset = _check_int(view.get("byteOffset", 0), f"buffer view {i} byteOffset")
        length = _check_int(view.get("byteLength"), f"buffer view {i} byteLength")
        _need(offset + length <= len(doc.buffers[buffer]),
              f"buffer view {i}: bytes {offset}..{offset + length} past buffer {buffer} of {len(doc.buffers[buffer])}")
        if "byteStride" in view:
            stride = _check_int(view["byteStride"], f"buffer view {i} byteStride", 4)
            _need(stride <= 252 and stride % 4 == 0, f"buffer view {i}: byteStride {stride}")


def _view_reach(doc: GltfDocument, index, offset, element: int, count: int, size: int, what: str,
                stride_allowed: bool = True) -> None:
    views = doc.json.get("bufferViews", [])
    _need(_is_int(index) and 0 <= index < len(views), f"{what}: buffer view {index!r} does not exist")
    _check_int(offset, f"{what} byteOffset")
    _need(offset % size == 0, f"{what}: byteOffset {offset} is not a multiple of {size}")
    view = views[index]
    stride = (view.get("byteStride") if stride_allowed else None) or element
    _need(stride >= element, f"{what}: byteStride {stride} below the element size {element}")
    if count:
        end = offset + stride * (count - 1) + element
        _need(end <= view["byteLength"], f"{what}: bytes up to {end} past buffer view {index} of {view['byteLength']}")


def _check_accessor(doc: GltfDocument, i: int) -> None:
    acc = doc.json["accessors"][i]
    what = f"accessor {i}"
    comp = acc.get("componentType")
    _need(comp in _COMPONENTS, f"{what}: componentType {comp!r}")
    kind = acc.get("type")
    _need(kind in _TYPES, f"{what}: type {kind!r}")
    count = _check_int(acc.get("count"), f"{what} count", 1)
    normalized = acc.get("normalized", False)
    _need(isinstance(normalized, bool), f"{what}: normalized is not a boolean")
    _need(not normalized or comp in _NORMALISED, f"{what}: componentType {comp} cannot be normalised")
    size = _COMPONENTS[comp][1]
    _, element = _element_layout(kind, size)
    if "bufferView" in acc:
        _view_reach(doc, acc["bufferView"], acc.get("byteOffset", 0), element, count, size, what)
    else:
        _need("byteOffset" not in acc, f"{what}: byteOffset without a buffer view")
    sparse = acc.get("sparse")
    if sparse is None:
        return
    _need(isinstance(sparse, dict), f"{what}: sparse is not an object")
    n = _check_int(sparse.get("count"), f"{what} sparse count", 1)
    _need(n <= count, f"{what}: sparse count {n} above count {count}")
    ind, val = sparse.get("indices"), sparse.get("values")
    _need(isinstance(ind, dict) and isinstance(val, dict), f"{what}: sparse indices or values missing")
    _need(ind.get("componentType") in _INDEX_TYPES, f"{what}: sparse index type {ind.get('componentType')!r}")
    isize = _COMPONENTS[ind["componentType"]][1]
    _view_reach(doc, ind.get("bufferView"), ind.get("byteOffset", 0), isize, n, isize, f"{what} sparse indices", False)
    _view_reach(doc, val.get("bufferView"), val.get("byteOffset", 0), element, n, size, f"{what} sparse values", False)
    icode = _COMPONENTS[ind["componentType"]][0]
    raw, base, _ = doc._view(ind["bufferView"], ind.get("byteOffset", 0), isize, stride_allowed=False)
    where = [struct.unpack_from("<" + icode, raw, base + k * isize)[0] for k in range(n)]
    _need(all(a < b for a, b in zip(where, where[1:])), f"{what}: sparse indices do not strictly increase")
    _need(where[-1] < count, f"{what}: sparse index {where[-1]} past count {count}")


def _attribute(doc: GltfDocument, primitive: dict, name: str, kinds: tuple, what: str):
    index = primitive["attributes"].get(name)
    if index is None:
        return None
    comp, kind, _, normalized = doc.accessor_info(index)
    _need(kind in kinds, f"{what}: {name} has type {kind}")
    return index


def _no_indices_mark(primitive: dict) -> bool:
    """V21-4: ``extras.d3sno.no_indices`` is true (the JSON value ``true`` only)."""
    extras = primitive.get("extras")
    info = extras.get("d3sno") if isinstance(extras, dict) else None
    return isinstance(info, dict) and info.get("no_indices") is True


def _check_meshes(doc: GltfDocument) -> None:
    accessors = doc.json.get("accessors", [])
    for m, mesh in enumerate(doc.json.get("meshes", [])):
        primitives = mesh.get("primitives")
        _need(isinstance(primitives, list) and primitives, f"mesh {m} has no primitives")
        for p, primitive in enumerate(primitives):
            what = f"mesh {m} primitive {p}"
            _need(isinstance(primitive, dict) and isinstance(primitive.get("attributes"), dict),
                  f"{what}: no attributes")
            mode = primitive.get("mode", 4)
            _need(mode == 4, f"{what}: mode {mode!r}, only 4 (triangles) is read")
            attributes = primitive["attributes"]
            for name, index in attributes.items():
                _need(_is_int(index) and 0 <= index < len(accessors), f"{what}: {name} accessor {index!r}")
            _need("POSITION" in attributes, f"{what}: no POSITION")
            count = accessors[attributes["POSITION"]]["count"]
            for name, index in attributes.items():
                _need(accessors[index]["count"] == count,
                      f"{what}: {name} has {accessors[index]['count']} elements, POSITION {count}")
            marked = _no_indices_mark(primitive)
            _need(not (marked and "indices" in primitive), f"{what}: extras.d3sno.no_indices with an indices accessor")
            if marked:
                pass  # V21-4: an empty index list
            elif "indices" in primitive:
                index = primitive["indices"]
                _need(_is_int(index) and 0 <= index < len(accessors), f"{what}: indices accessor {index!r}")
                comp, kind, _, normalized = doc.accessor_info(index)
                _need(kind == "SCALAR" and comp in _INDEX_TYPES and not normalized,
                      f"{what}: indices accessor of type {kind} / {comp}")
                values = [row[0] for row in doc.accessor(index)]
                _need(len(values) % 3 == 0, f"{what}: {len(values)} indices are not whole triangles")
                _need(not values or max(values) < count, f"{what}: index {max(values or [0])} past {count} vertices")
            else:
                _need(count % 3 == 0, f"{what}: {count} vertices without indices are not whole triangles")


def gltf_to_glb(document: dict) -> bytes:
    """A document of ``appearance.to_gltf`` (one data-URI buffer) as a ``.glb``: the buffer moves
    to the BIN chunk. Used by the tests and by scripts that want a binary file."""
    doc = json.loads(json.dumps(document))
    blob = b""
    if doc.get("buffers"):
        blob = _uri_bytes(doc["buffers"][0]["uri"], None, "buffer 0")
        doc["buffers"][0] = {"byteLength": len(blob)}
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode("utf-8")
    text += b" " * (-len(text) % 4)
    out = struct.pack("<II", len(text), GLB_JSON) + text
    if blob:
        blob += bytes(-len(blob) % 4)
        out += struct.pack("<II", len(blob), GLB_BIN) + blob
    return GLB_MAGIC + struct.pack("<II", 2, 12 + len(out)) + out


# --- OBJ reading (W16-10) ----------------------------------------------------------------------

@dataclass
class ObjGroup:
    name: str | None
    faces: list = field(default_factory=list)  # triangles of (v, vt, vn) corners, 0-based, None absent
    vertex_lines: list = field(default_factory=list)  # the v lines declared inside the group


@dataclass
class ObjObject:
    name: str | None
    groups: list = field(default_factory=list)


@dataclass
class ObjDocument:
    positions: list
    texcoords: list
    normals: list
    objects: list


def _floats(tokens, low: int, line: int, what: str) -> tuple:
    if len(tokens) < low:
        raise FormatError(f"OBJ line {line}: {what} needs {low} numbers")
    try:
        values = tuple(float(t) for t in tokens)
    except ValueError:
        raise FormatError(f"OBJ line {line}: {what} has a value that is not a number") from None
    if not all(math.isfinite(v) for v in values):
        raise FormatError(f"OBJ line {line}: {what} is not finite")
    return values


def read_obj(text: str) -> ObjDocument:
    """W16-10: ``o``, ``g``, ``v``, ``vt``, ``vn`` and ``f`` (``a``, ``a/b``, ``a//c``, ``a/b/c``,
    negative indices; faces of more than three corners as a fan from the first). ``s``,
    ``usemtl``, ``mtllib`` and comments are ignored; any other statement is refused."""
    doc = ObjDocument([], [], [], [])
    current: ObjObject | None = None
    group: ObjGroup | None = None
    pending = []  # (line, kind, index) of positive indices checked at the end

    def where() -> ObjGroup:
        nonlocal current, group
        if current is None:
            current = ObjObject(None)
            doc.objects.append(current)
        if group is None:
            group = ObjGroup(None)
            current.groups.append(group)
        return group

    lists = {"v": doc.positions, "vt": doc.texcoords, "vn": doc.normals}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        keyword, _, rest = line.partition(" ")
        tokens = rest.split()
        if keyword == "v":
            where().vertex_lines.append(len(doc.positions))
            doc.positions.append(_floats(tokens, 3, number, "v")[:3])
        elif keyword == "vt":
            values = _floats(tokens, 1, number, "vt")
            doc.texcoords.append((values[0], values[1] if len(values) > 1 else 0.0))
        elif keyword == "vn":
            doc.normals.append(_floats(tokens, 3, number, "vn")[:3])
        elif keyword == "f":
            if len(tokens) < 3:
                raise FormatError(f"OBJ line {number}: a face needs three corners")
            corners = []
            for token in tokens:
                parts = token.split("/")
                if len(parts) > 3 or not parts[0]:
                    raise FormatError(f"OBJ line {number}: corner {token!r}")
                corner = []
                for kind, part in zip(("v", "vt", "vn"), parts + [""] * (3 - len(parts))):
                    if not part:
                        corner.append(None)
                        continue
                    try:
                        value = int(part)
                    except ValueError:
                        raise FormatError(f"OBJ line {number}: corner {token!r}") from None
                    if value == 0:
                        raise FormatError(f"OBJ line {number}: index 0 in {token!r}")
                    if value < 0:
                        value += len(lists[kind])
                        if value < 0:
                            raise FormatError(f"OBJ line {number}: relative index {part} before the first {kind}")
                    else:
                        value -= 1
                        pending.append((number, kind, value))
                    corner.append(value)
                corners.append(tuple(corner))
            target = where()
            for k in range(1, len(corners) - 1):
                target.faces.append((corners[0], corners[k], corners[k + 1]))
        elif keyword == "o":
            current = ObjObject(rest.strip() or None)
            doc.objects.append(current)
            group = None
        elif keyword == "g":
            if current is None:
                current = ObjObject(None)
                doc.objects.append(current)
            group = ObjGroup(rest.strip() or None)
            current.groups.append(group)
        elif keyword in ("s", "usemtl", "mtllib"):
            continue
        else:
            raise FormatError(f"OBJ line {number}: statement {keyword!r} is not read")
    for number, kind, value in pending:
        if value >= len(lists[kind]):
            raise FormatError(f"OBJ line {number}: {kind} index {value + 1} past {len(lists[kind])}")
    return doc


# --- The import (W16-7 to W16-11) --------------------------------------------------------------

@dataclass
class ImportReport:
    """What ``import_mesh_report`` did: draw labels changed and kept, and warnings (W16-9).
    W17-14: ``added`` and ``removed`` hold the label of every added and removed part (sub-object,
    ps3 draw, skinning record, rigid list, material entry), as the writer's ``PartsReport`` gives
    them (a removed part by its template place, an added one by its output place)."""
    changed: list = field(default_factory=list)
    kept: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    added: list = field(default_factory=list)
    removed: list = field(default_factory=list)


@dataclass
class _Source:
    """One imported primitive (glTF) or draw (OBJ) in the template's terms."""
    form: str  # "gltf" or "obj"
    positions: list
    indices: list
    normals: list | None = None  # per vertex, an entry may be None (OBJ)
    uv1: list | None = None
    uv2_raw: list | None = None
    uv2_float: list | None = None
    colours_1: list | None = None
    colours_2: list | None = None
    influences: list | None = None  # per vertex [(bone or None, weight)] over every JOINTS_n set
    has_joints: bool = False
    exporter_order: bool = True  # OBJ: vertices numbered as the exporter writes them


@dataclass
class _Options:
    flip_v: bool
    lightmap_v_scale: float | None
    missing: str
    skin: str
    look: int = 0


@dataclass
class _Match:
    """A mesh (object) matched to a template sub-object: its primitives (groups) by draw index,
    and the ones that match no draw, which are new geometry of the sub-object (W17-12)."""
    sources: dict
    joint_map: object
    extra: list = field(default_factory=list)


@dataclass
class _NewPart:
    """A mesh (glTF) or object (OBJ) that becomes a new sub-object (W17-12). ``template`` is the
    matched sub-object of an editor copy (its template part), else None."""
    what: str
    name: str | None
    geo_set: int
    template: SubObject | None
    info: dict | None  # the mesh's extras.d3sno
    textures: list  # texture handles of primitive materials ``texture_<handle>_draw_<k>``, in order
    sources: list
    joint_map: object = None


def _label(sub: SubObject, k: int) -> str:
    return f"{sub.label}, draw {k}"


def _strip_copy(name: str) -> str | None:
    match = _COPY_SUFFIX.match(name)
    return match.group(1) if match else None


def _named(name: str, candidates: list[SubObject]) -> list[SubObject]:
    """W16-7 step 2 for one name: the sub-objects whose exporter object name equals it, and
    (V21-5) for ``<name>__sub<i>`` sub-object i of its geo set when its object name is
    ``<name>``."""
    found = [sub for sub in candidates if _object_name(sub) == name]
    match = _REPEATED.fullmatch(name)
    if match:
        base, index = match.group(1), int(match.group(2))
        found += [sub for sub in candidates
                  if sub.index == index and _object_name(sub) == base and sub not in found]
    return found


def _match_sub(extras, names, candidates: list[SubObject]) -> SubObject | None:
    """W16-7: extras.d3sno, then a name equal to the exporter's object name (or V21-5
    ``<name>__sub<i>``), then that name without a trailing ``.NNN``. A name fitting several
    sub-objects is refused."""
    info = extras.get("d3sno") if isinstance(extras, dict) else None
    if isinstance(info, dict):
        for sub in candidates:
            if _is_int(info.get("geo_set")) and _is_int(info.get("sub_object")) \
                    and (sub.geo_set, sub.index) == (info["geo_set"], info["sub_object"]):
                return sub
    for step in (lambda n: n, _strip_copy):
        for name in names:
            name = step(name) if isinstance(name, str) else None
            if not name:
                continue
            found = _named(name, candidates)
            if len(found) > 1:
                raise FormatError(f"name {name!r} fits several sub-objects: "
                                  + "; ".join(sub.label for sub in found))
            if found:
                return found[0]
    return None


def _match_draws(sub: SubObject, keys: list, what: str) -> list:
    """W16-7 draws: ``keys`` holds, per primitive, the draw named by extras or material (or
    None). The rest are matched by order when the counts allow it. W17-12: when they do not and
    some primitive names a draw, a primitive naming none matches no draw and is new geometry of the
    sub-object (None in the result); when no primitive names a draw, the counts must agree."""
    positioned = [k for k, d in enumerate(sub.draws) if d.positions]
    by_order = len(keys) == len(positioned)
    out = []
    for p, k in enumerate(keys):
        if k is None:
            if not by_order and all(key is None for key in keys):
                raise FormatError(f"{what}: primitive {p} names no draw and {len(keys)} primitives "
                                  f"do not match the {len(positioned)} draws of {sub.label} by order")
            if not by_order:
                out.append(None)  # W17-12: new geometry
                continue
            k = positioned[p]
        if not 0 <= k < len(sub.draws):
            raise FormatError(f"{what}: draw {k} of {sub.label}, which has {len(sub.draws)} draws")
        if k in out:
            raise FormatError(f"{what}: two primitives for {_label(sub, k)}")
        out.append(k)
    return out


# --- node transforms -----------------------------------------------------------------------------

def _mat_mul(a, b):
    return tuple(sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4)) for c in range(4) for r in range(4))


def _local_matrix(node: dict) -> tuple:
    if "matrix" in node:
        matrix = node["matrix"]
        _need(isinstance(matrix, list) and len(matrix) == 16 and all(isinstance(v, (int, float)) for v in matrix),
              "node matrix is not 16 numbers")
        return tuple(float(v) for v in matrix)
    values = {}
    for key, default in (("translation", [0.0] * 3), ("rotation", [0.0, 0.0, 0.0, 1.0]), ("scale", [1.0] * 3)):
        value = node.get(key, default)
        _need(isinstance(value, list) and len(value) == len(default)
              and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value),
              f"node {key} is not {len(default)} numbers")
        values[key] = [float(v) for v in value]
    t, (x, y, z, w), s = values["translation"], values["rotation"], values["scale"]
    rotation = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
                2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
                2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y))
    matrix = tuple(rotation[c * 3 + r] * s[c] for c in range(3) for r in range(3))
    return (matrix[0:3] + (0.0,) + matrix[3:6] + (0.0,) + matrix[6:9] + (0.0,)
            + (t[0], t[1], t[2], 1.0))


def _world_matrices(doc: GltfDocument) -> list[tuple]:
    nodes = doc.json.get("nodes", [])
    parent: dict[int, int] = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            _need(_is_int(child) and 0 <= child < len(nodes), f"node {i}: child {child!r}")
            _need(child not in parent, f"node {child} has two parents")
            parent[child] = i
    world: dict[int, tuple] = {}

    def resolve(i: int, seen: tuple) -> tuple:
        if i in world:
            return world[i]
        _need(i not in seen, f"node {i} is its own ancestor")
        local = _local_matrix(nodes[i])
        if i in parent:
            above = resolve(parent[i], seen + (i,))
            local = local if above == _IDENTITY else (above if local == _IDENTITY else _mat_mul(above, local))
        world[i] = local
        return local

    return [resolve(i, ()) for i in range(len(nodes))]


def _transform(matrix, positions, normals):
    """Positions through the matrix; normals through the inverse transpose of its 3x3, then
    normalised (W16-7)."""
    m = matrix
    out_p = [tuple(m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r] for r in range(3)) for x, y, z in positions]
    if normals is None:
        return out_p, None
    a = [[m[c * 4 + r] for c in range(3)] for r in range(3)]  # a[row][col]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    _need(det != 0, "a node matrix cannot be inverted for the normals")
    # inverse transpose = cofactor matrix / det
    cof = [[(a[(r + 1) % 3][(c + 1) % 3] * a[(r + 2) % 3][(c + 2) % 3]
             - a[(r + 1) % 3][(c + 2) % 3] * a[(r + 2) % 3][(c + 1) % 3]) / det for c in range(3)] for r in range(3)]
    out_n = []
    for n in normals:
        v = tuple(sum(cof[r][c] * n[c] for c in range(3)) for r in range(3))
        length = math.sqrt(sum(c * c for c in v))
        out_n.append(tuple(c / length for c in v) if length else v)
    return out_p, out_n


# --- joints and weights (W16-9) -------------------------------------------------------------------

def weight_bytes(pairs) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """W16-9: (bone, weight) pairs of one vertex to four joints and four weight bytes summing to
    255. Equal bones are added, zero weights dropped, the four largest kept (ties by first
    occurrence), normalised, floored to units of 1/255 with the remaining units given to the
    largest fractional parts (ties by lower position), sorted by descending weight (ties by first
    occurrence); unused slots are (0, 0). A vertex with no weight left raises FormatError."""
    merged: dict[int, Fraction] = {}
    for bone, weight in pairs:
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise FormatError(f"weight {weight!r} is not a finite number >= 0")
        if weight == 0:
            continue
        merged[bone] = merged.get(bone, Fraction(0)) + Fraction(weight)
    items = [(bone, w) for bone, w in merged.items() if w > 0]
    if not items:
        raise FormatError("a vertex of a skinned draw has no weight left (W16-9)")
    kept = sorted(sorted(range(len(items)), key=lambda p: (-items[p][1], p))[:4])
    items = [items[p] for p in kept]
    total = sum(w for _, w in items)
    scaled = [w / total * 255 for _, w in items]
    units = [math.floor(s) for s in scaled]
    for p in sorted(range(len(items)), key=lambda p: (-(scaled[p] - units[p]), p))[:255 - sum(units)]:
        units[p] += 1
    order = sorted(range(len(items)), key=lambda p: (-units[p], p))
    joints = [items[p][0] if units[p] else 0 for p in order] + [0] * (4 - len(order))
    weights = [units[p] for p in order] + [0] * (4 - len(order))
    return tuple(joints), tuple(weights)


def export_weights(draw: Draw, i: int) -> list[tuple[int, float]]:
    """W16-9 export form of a stored vertex: (skeleton joint, weight / sum as float32)."""
    row = draw.weights[i]
    total = sum(row)
    weights = [_f32(w / total) for w in row] if total else [1.0, 0.0, 0.0, 0.0]
    return list(zip(draw.joints[i], weights))


class _JointMap:
    """W16-9: glTF joint j of a skin to a skeleton bone, by the exporter's node name (hash
    checked), else by node index when the skin's extras give the skeleton's bone count."""

    def __init__(self, doc: GltfDocument, skin_index, app: AppearanceFile, warnings: list[str]):
        skin = doc._get("skins", skin_index, "skin")
        joints = skin.get("joints")
        _need(isinstance(joints, list) and joints, f"skin {skin_index} has no joints")
        nodes = doc.json.get("nodes", [])
        extras = skin.get("extras")
        info = extras.get("d3sno") if isinstance(extras, dict) else None
        by_index = isinstance(info, dict) and info.get("bones") == len(app.bones)
        self.bones: list[int | None] = []
        self.problems: dict[int, str] = {}
        for j, node_index in enumerate(joints):
            _need(_is_int(node_index) and 0 <= node_index < len(nodes), f"skin {skin_index}: joint node {node_index!r}")
            node = nodes[node_index]
            name = node.get("name")
            match = _BONE_NAME.fullmatch(name) if isinstance(name, str) else None
            bone = None
            if match:
                i = int(match.group(1))
                if i < len(app.bones) and app.bones[i].name_hash == int(match.group(2), 16):
                    bone = i
            if bone is None and by_index and node_index < len(app.bones):
                bone = node_index
            self.bones.append(bone)
            if bone is None:
                self.problems[j] = (f"skin {skin_index} joint {j} (node {node_index} {name!r}) names no "
                                    "skeleton bone (W16-9)")
            elif not _same_bone_transform(node, app.bones[bone]):
                warnings.append(f"bone {bone}: the glTF node transform differs from the skeleton; "
                                "ignored, the skeleton is not changed (W16-9)")

    def bone(self, joint: int) -> int | None:
        return self.bones[joint] if _is_int(joint) and 0 <= joint < len(self.bones) else None

    def need(self, joint: int, what: str) -> int:
        bone = self.bone(joint)
        if bone is None:
            problem = self.problems.get(joint, f"joint {joint} past the skin's {len(self.bones)} joints")
            raise FormatError(f"{what}: {problem}")
        return bone


def _same_bone_transform(node: dict, bone) -> bool:
    try:
        want = _local_matrix(_bone_node(bone))
        got = _local_matrix(node)
    except FormatError:
        return False
    return all(abs(a - b) <= 1e-5 * (1 + abs(b)) for a, b in zip(got, want))


# --- glTF sources --------------------------------------------------------------------------------

def _gltf_source(doc: GltfDocument, primitive: dict, matrix, joint_map, what: str) -> _Source:
    attributes = primitive["attributes"]
    floats = lambda rows: [tuple(float(c) for c in row) for row in rows]  # noqa: E731
    positions = floats(doc.accessor(_attribute(doc, primitive, "POSITION", ("VEC3",), what)))
    count = len(positions)
    if _no_indices_mark(primitive):
        indices = []  # V21-4
    elif "indices" in primitive:
        indices = [row[0] for row in doc.accessor(primitive["indices"])]
    else:
        indices = list(range(count))  # W16-6
    source = _Source("gltf", positions, indices)
    index = _attribute(doc, primitive, "NORMAL", ("VEC3",), what)
    source.normals = floats(doc.accessor(index)) if index is not None else None
    index = _attribute(doc, primitive, "TEXCOORD_0", ("VEC2",), what)
    source.uv1 = floats(doc.accessor(index)) if index is not None else None
    index = _attribute(doc, primitive, "TEXCOORD_1", ("VEC2",), what)
    source.uv2_float = floats(doc.accessor(index)) if index is not None else None
    index = _attribute(doc, primitive, "_D3_UV2", ("VEC2",), what)
    if index is not None:
        source.uv2_raw = [tuple(_to_int(c, 0xFFFF, f"{what}: _D3_UV2") for c in row) for row in doc.accessor(index)]
    for name, slot in (("_D3_COLOR1", "colours_1"), ("_D3_COLOR2", "colours_2")):
        index = _attribute(doc, primitive, name, ("VEC4",), what)
        if index is not None:
            setattr(source, slot, [bytes(_to_int(c, 0xFF, f"{what}: {name}") for c in row)
                                   for row in doc.accessor(index)])
    sets = []
    n = 0
    while f"JOINTS_{n}" in attributes or f"WEIGHTS_{n}" in attributes:
        _need(f"JOINTS_{n}" in attributes and f"WEIGHTS_{n}" in attributes,
              f"{what}: JOINTS_{n} and WEIGHTS_{n} do not come in a pair")
        joints_index = _attribute(doc, primitive, f"JOINTS_{n}", ("VEC4",), what)
        comp, _, _, normalized = doc.accessor_info(joints_index)
        _need(comp in (5121, 5123) and not normalized, f"{what}: JOINTS_{n} is not unsigned byte or short")
        sets.append((doc.accessor(joints_index),
                     doc.accessor(_attribute(doc, primitive, f"WEIGHTS_{n}", ("VEC4",), what))))
        n += 1
    if sets:
        source.has_joints = True
        _need(joint_map is not None, f"{what}: joints without a skin on the node")
        source.influences = []
        for i in range(count):
            pairs = []
            for joints, weights in sets:
                for joint, weight in zip(joints[i], weights[i]):
                    pairs.append((joint, float(weight)))
            source.influences.append(pairs)
    elif matrix is not None and matrix != _IDENTITY:
        source.positions, source.normals = _transform(matrix, source.positions, source.normals)
    return source


def _to_int(value, high: int, what: str) -> int:
    if isinstance(value, float):
        value = round(value * high)  # a normalised or float attribute
    if not 0 <= value <= high:
        raise FormatError(f"{what}: value {value} outside 0..{high}")
    return value


def _d3sno_info(extras) -> dict | None:
    info = extras.get("d3sno") if isinstance(extras, dict) else None
    return info if isinstance(info, dict) else None


def _has_joints(primitive: dict) -> bool:
    return any(name.startswith(("JOINTS_", "WEIGHTS_")) for name in primitive["attributes"])


def _new_geo_set(info: dict | None, name: str | None) -> int:
    """W17-12: ``extras.d3sno.geo_set`` when present (0 or 1), else 1 for a ``lowpoly_`` name,
    else 0."""
    if info is not None and _is_int(info.get("geo_set")) and info["geo_set"] in (0, 1):
        return info["geo_set"]
    return 1 if isinstance(name, str) and name.startswith(LOWPOLY_PREFIX) else 0


def _choose_matches(found, names) -> dict:
    """W17-12 editor copies: of the meshes matching one sub-object, the one whose name equals the
    exporter's name exactly, else the first in node order, is the match. ``found`` holds (names of
    the mesh, sub-object or None); returns {(geo set, index): position in ``found``}."""
    chosen: dict = {}
    for p, (given, sub) in enumerate(found):
        if sub is None:
            continue
        key = (sub.geo_set, sub.index)
        exact = names.get(key) in given
        if key not in chosen or (exact and not chosen[key][1]):
            chosen[key] = (p, exact)
    return {key: p for key, (p, _) in chosen.items()}


def _refuse_bones(app: AppearanceFile, primitive: dict, what: str) -> None:
    """W17-12: in a file without bones, a primitive with joints is refused."""
    if not app.bones and _has_joints(primitive):
        raise FormatError(f"{what}: a primitive with joints in a file without bones (W17-12)")


def _gltf_matches(doc: GltfDocument, app: AppearanceFile, candidates, skipped, report):
    """W16-7, W17-12: (matches by (geo set, index), new parts in node order)."""
    nodes = doc.json.get("nodes", [])
    meshes = doc.json.get("meshes", [])
    worlds = _world_matrices(doc)
    found = []  # (node index, node, mesh, what, sub-object or None)
    for n, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        m = node["mesh"]
        _need(_is_int(m) and 0 <= m < len(meshes), f"node {n}: mesh {m!r}")
        mesh = meshes[m]
        what = f"glTF mesh {m} ({mesh.get('name')!r})"
        given = (mesh.get("name"), node.get("name"))
        try:
            sub = _match_sub(mesh.get("extras"), given, candidates)
            if sub is None and _match_sub(mesh.get("extras"), given, skipped):
                report.warnings.append(f"{what} is a low-poly sub-object and is not imported")
                continue
        except FormatError as exc:
            raise FormatError(f"{what}: {exc.message}") from None
        found.append((n, node, mesh, what, sub))
    chosen = _choose_matches([((mesh.get("name"), node.get("name")), sub) for _, node, mesh, _, sub in found],
                             export_object_names(app))
    joint_maps: dict = {}

    def joint_map_of(node):
        if "skin" not in node:
            return None
        if node["skin"] not in joint_maps:
            joint_maps[node["skin"]] = _JointMap(doc, node["skin"], app, report.warnings)
        return joint_maps[node["skin"]]

    matches: dict = {}
    new_parts: list[_NewPart] = []
    for p, (n, node, mesh, what, sub) in enumerate(found):
        if sub is not None and chosen[(sub.geo_set, sub.index)] == p:
            joint_map = joint_map_of(node)
            keys = []
            for primitive in mesh["primitives"]:
                k = None
                info = _d3sno_info(primitive.get("extras"))
                if info is not None and _is_int(info.get("draw")):
                    k = info["draw"]
                elif "material" in primitive:
                    material = doc._get("materials", primitive["material"], "material")
                    match = _DRAW_SUFFIX.search(material.get("name") or "")
                    k = int(match.group(1)) if match else None
                keys.append(k)
            draws = _match_draws(sub, keys, what)
            sources, extra = {}, []
            for q, (k, primitive) in enumerate(zip(draws, mesh["primitives"])):
                pwhat = f"{what} primitive {q}"
                if k is None:
                    _refuse_bones(app, primitive, pwhat)
                    extra.append(_gltf_source(doc, primitive, worlds[n], joint_map, pwhat))
                else:
                    sources[k] = _gltf_source(doc, primitive, worlds[n], joint_map, pwhat)
            matches[(sub.geo_set, sub.index)] = _Match(sources, joint_map, extra)
            continue
        # W17-12: a mesh that matches no sub-object, or an editor copy of a matched one
        name = mesh.get("name") if isinstance(mesh.get("name"), str) and mesh.get("name") else node.get("name")
        if sub is None and not (isinstance(name, str) and name):
            raise FormatError(f"{what} matches no sub-object and has no name for a new one (W17-12)")
        info = _d3sno_info(mesh.get("extras"))
        textures = []
        for q, primitive in enumerate(mesh["primitives"]):
            _refuse_bones(app, primitive, f"{what} primitive {q}")
            if "material" in primitive:
                material = doc._get("materials", primitive["material"], "material")
                match = _TEXTURE_MATERIAL.fullmatch(material.get("name") or "")
                if match:
                    textures.append(int(match.group(1)))
        joint_map = joint_map_of(node)
        sources = [_gltf_source(doc, primitive, worlds[n], joint_map, f"{what} primitive {q}")
                   for q, primitive in enumerate(mesh["primitives"])]
        geo = _new_geo_set(info, name)
        new_parts.append(_NewPart(what, name, geo, sub, info, textures, sources, joint_map))
    return matches, new_parts


# --- OBJ sources -------------------------------------------------------------------------------

def _obj_matches(doc: ObjDocument, app: AppearanceFile, candidates, skipped, report):
    """W16-10, W17-12: (matches by (geo set, index), new parts in file order)."""
    found = []  # (object, groups, what, sub-object or None)
    for obj in doc.objects:
        groups = [g for g in obj.groups if g.faces]
        if not groups:
            continue
        what = f"OBJ object {obj.name!r}"
        try:
            sub = _match_sub(None, (obj.name,), candidates)
            if sub is None and _match_sub(None, (obj.name,), skipped):
                report.warnings.append(f"{what} is a low-poly sub-object and is not imported")
                continue
        except FormatError as exc:
            raise FormatError(f"{what}: {exc.message}") from None
        if sub is None and not obj.name:
            raise FormatError(f"{what} matches no sub-object and has no name for a new one (W17-12)")
        found.append((obj, groups, what, sub))
    chosen = _choose_matches([((obj.name,), sub) for obj, _, _, sub in found], export_object_names(app))
    matches: dict = {}
    new_parts: list[_NewPart] = []
    for p, (obj, groups, what, sub) in enumerate(found):
        if sub is None or chosen[(sub.geo_set, sub.index)] != p:
            # W17-12: an ``o`` name that matches no sub-object, or an editor copy (no extras in OBJ)
            geo = _new_geo_set(None, obj.name)
            new_parts.append(_NewPart(what, obj.name, geo, sub, None, [], [_obj_source(doc, groups, what)]))
            continue
        named = {}
        plain = []
        for g in groups:
            match = re.fullmatch(r"draw_(\d+)", g.name or "")
            if match:
                named.setdefault(int(match.group(1)), []).append(g)
            else:
                plain.append(g)
        positioned = [k for k, d in enumerate(sub.draws) if d.positions]
        extra = []
        if plain and named:
            extra = [_obj_source(doc, plain, f"{what} new geometry")]  # W17-12: a ``g`` matching no draw
        elif plain:
            if len(positioned) != 1:
                raise FormatError(f"{what}: no 'g draw_<k>' groups and {len(positioned)} draws with positions "
                                  f"in {sub.label}")
            named = {positioned[0]: plain}
        out = {}
        for k, parts in named.items():
            if not 0 <= k < len(sub.draws):
                raise FormatError(f"{what}: draw {k} of {sub.label}, which has {len(sub.draws)} draws")
            out[k] = _obj_source(doc, parts, f"{what} draw {k}")
        matches[(sub.geo_set, sub.index)] = _Match(out, None, extra)
    return matches, new_parts


def _obj_source(doc: ObjDocument, groups, what: str) -> _Source:
    faces = [face for g in groups for face in g.faces]
    declared = [v for g in groups for v in g.vertex_lines]
    corners = [c for face in faces for c in face]
    has_t = {c[1] is not None for c in corners}
    has_n = {c[2] is not None for c in corners}
    exporter = len(has_t) == 1 and len(has_n) == 1 and \
        all(c[1] in (None, c[0]) and c[2] in (None, c[0]) for c in corners)
    if exporter:
        refs = {c[0] for c in corners}
        order = declared if refs <= set(declared) and len(set(declared)) == len(declared) else sorted(refs)
        where = {v: i for i, v in enumerate(order)}
        indices = [where[c[0]] for c in corners]
        keys = [(v, v if True in has_t else None, v if True in has_n else None) for v in order]
        for v, t, n in keys:
            if t is not None and t >= len(doc.texcoords) or n is not None and n >= len(doc.normals):
                raise FormatError(f"{what}: v = vt = vn corner {v + 1} has no vt or vn line")
    else:
        where = {}
        keys = []
        indices = []
        for c in corners:
            if c not in where:
                where[c] = len(keys)
                keys.append(c)
            indices.append(where[c])
    source = _Source("obj", [doc.positions[v] for v, _, _ in keys], indices, exporter_order=exporter)
    source.uv1 = [doc.texcoords[t] if t is not None else None for _, t, _ in keys]
    source.normals = [doc.normals[n] if n is not None else None for _, _, n in keys]
    return source


# --- one draw -------------------------------------------------------------------------------------

def _fixed(draw: Draw) -> bool:
    return draw.kind in ("packed", "ps3")


def _raw(position, sub: SubObject):
    try:
        return M.raw_coordinates(position, sub.bias, sub.scale)
    except FormatError:
        return None


def _position_kept(form: str, sub: SubObject, draw: Draw, i: int, value) -> bool:
    """A fixed-point position whose raw coordinates equal the stored ones, or a fat position
    equal to its export form (float32 in glTF, ``%.6f`` in OBJ)."""
    stored = draw.positions[i]
    if _fixed(draw):
        raw = _raw(value, sub)
        return raw is not None and raw == _raw(stored, sub)
    if form == "gltf":
        return tuple(_f32(c) for c in stored) == tuple(value)
    return tuple(float(f"{c:.6f}") for c in stored) == tuple(value)


def _positions_kept(form: str, sub: SubObject, draw: Draw, values) -> list[bool]:
    """``_position_kept`` for every vertex of a same-count draw, with the stored raw coordinates
    converted once per draw (Corrections 21, speed note)."""
    if not _fixed(draw):
        return [_position_kept(form, sub, draw, i, value) for i, value in enumerate(values)]
    stored = [_raw(p, sub) for p in draw.positions]
    out = []
    for i, value in enumerate(values):
        raw = _raw(value, sub)
        out.append(raw is not None and raw == stored[i])
    return out


def _normal_kept(form: str, stored, value) -> bool:
    if form == "gltf":
        return tuple(_f32(c) for c in _unit(stored)) == tuple(value)
    return tuple(float(f"{c:.6f}") for c in stored) == tuple(value)


def _uv_words(uv):
    try:
        return M.uv1_words(uv)
    except FormatError:
        return None


def _make_draw(app: AppearanceFile, sub: SubObject, k: int, src: _Source, opts: _Options,
               joint_map, warnings: list[str]) -> Draw:
    stored = sub.draws[k]
    label = _label(sub, k)
    n = len(src.positions)
    if n > MAX_VERTICES:
        raise FormatError(f"{label}: {n} vertices, indices are u16 (at most {MAX_VERTICES}, W16-11)")
    if keeps_every_value(sub) and n != len(stored.positions):
        raise _kept_refusal(sub, k, f"vertex count ({len(stored.positions)} -> {n})")
    if src.form == "gltf" and src.has_joints != stored.skinned:
        kind = "with joints for a rigid" if src.has_joints else "without joints for a skinned"
        raise FormatError(f"{label}: a primitive {kind} draw; flag bits are not changed (W16-8, Q376)")
    same = n == len(stored.positions)
    kept_at = _positions_kept(src.form, sub, stored, src.positions) if same else []
    if same and not src.exporter_order:  # OBJ numbered by first use (W16-10)
        same = all(kept_at)

    positions = [stored.positions[i] if same and kept_at[i] else tuple(p) for i, p in enumerate(src.positions)]

    def per_vertex(values, keep, convert, what):
        out = []
        for i in range(n):
            value = values[i] if values is not None else None
            if value is None:
                if not same:
                    raise FormatError(f"{label}: {what} missing for a new vertex list ({n} vertices "
                                      f"for {len(stored.positions)})")
                out.append(keep(i, None))
            else:
                out.append(keep(i, value) if same else convert(value))
        return out

    def normal_unit(value):
        length = math.sqrt(sum(c * c for c in value))
        return tuple(c / length for c in value) if length else (0.0, 0.0, 0.0)

    normals = per_vertex(
        src.normals,
        lambda i, v: stored.normals[i] if v is None or _normal_kept(src.form, stored.normals[i], v) else normal_unit(v),
        normal_unit, "normals")

    def uv_back(value):
        u, v = value
        return (float(u), 1.0 - v if opts.flip_v else float(v))

    uv1 = per_vertex(
        src.uv1,
        lambda i, v: stored.uv1[i] if v is None or (_uv_words(uv_back(v)) is not None
                                                    and _uv_words(uv_back(v)) == _uv_words(stored.uv1[i])) else uv_back(v),
        uv_back, "UV1 (texture coordinates)")

    if src.uv2_raw is not None:
        uv2 = [tuple(row) for row in src.uv2_raw]
    elif src.uv2_float is not None:
        factor = opts.lightmap_v_scale or 1.0
        uv2 = [(round(u * 8192 + 0x8000), round(v / factor * 8192 + 0x8000)) for u, v in src.uv2_float]
    elif same:
        uv2 = list(stored.uv2_raw)
    else:
        uv2 = [(0x8000, 0x8000)] * n
    for i, (u, v) in enumerate(uv2):
        if not (0 <= u <= 0xFFFF and 0 <= v <= 0xFFFF):
            raise FormatError(f"{label}: vertex {i} UV2 {(u, v)} outside the u16 range")
    if same and uv2 == stored.uv2_raw:
        uv2 = stored.uv2_raw

    colours = []
    for given, kept, name in ((src.colours_1, stored.colours_1, "_D3_COLOR1"), (src.colours_2, stored.colours_2, "_D3_COLOR2")):
        if given is not None:
            colours.append(kept if same and given == kept else list(given))
        elif same:
            colours.append(kept)
        elif kept:
            raise FormatError(f"{label}: {name} missing for a new vertex list and the draw has colours (W16-8)")
        else:
            warnings.append(f"{label}: no colours given or stored; new vertices get colour 0")
            colours.append([bytes(4)] * n)

    joints = weights = None
    if stored.skinned:
        joints, weights = [], []
        for i in range(n):
            if src.influences is not None:
                pairs = src.influences[i]
                if same and _export_form_equal(pairs, joint_map, stored, i):
                    joints.append(stored.joints[i])
                    weights.append(stored.weights[i])
                    continue
                mapped = []
                for joint, weight in pairs:
                    if weight != 0:
                        mapped.append((joint_map.need(joint, f"{label}: vertex {i}"), weight))
                try:
                    j, w = weight_bytes(mapped)
                except FormatError as exc:
                    raise FormatError(f"{label}: vertex {i}: {exc.message}") from None
            elif same:
                j, w = stored.joints[i], stored.weights[i]
            elif opts.skin == "nearest":
                if not stored.positions:
                    raise FormatError(f"{label}: no stored vertex to take the skin from")
                p = positions[i]
                best = min(range(len(stored.positions)),
                           key=lambda q: (sum((p[c] - stored.positions[q][c]) ** 2 for c in range(3)), q))
                j, w = stored.joints[best], stored.weights[best]
            else:
                raise FormatError(f"{label}: OBJ has no skin and the vertex list is new; use skin='nearest' (W16-10)")
            joints.append(tuple(j))
            weights.append(tuple(w))
        if joints == stored.joints:
            joints = stored.joints
        if weights == stored.weights:
            weights = stored.weights

    indices = list(src.indices)
    draw = dataclasses.replace(
        stored, positions=positions, normals=normals, colours_1=colours[0], colours_2=colours[1], uv1=uv1,
        uv2_raw=uv2, indices=indices, joints=joints, weights=weights)
    if _equal(draw, stored):
        return stored
    if keeps_every_value(sub):
        name = next(name for name in _FIELDS if getattr(draw, name) != getattr(stored, name))
        raise _kept_refusal(sub, k, _FIELD_NAMES[name])
    _validate(app, draw, label)
    return draw


def keeps_every_value(sub: SubObject) -> bool:
    """V21-1 (replaces W16-11 rule 4, Q716): a sub-object with a cloth record (G1-4; a non-empty
    cloth pair or a read record) or with fat vertices (+0x04 non-zero or a fat draw) keeps every
    stored value; any change to one of its draws is refused."""
    return bool(sub.has_cloth or sub.cloth is not None or sub.fat_draw is not None or sub.cloth_vertex_count)


def _kept_refusal(sub: SubObject, k: int, what: str) -> FormatError:
    return FormatError(f"{_label(sub, k)}: the import changes the {what} of a sub-object with cloth data or fat "
                       "vertices, which keeps every stored value (V21-1; how cloth follows its draw is OPEN)")


def _export_form_equal(pairs, joint_map, stored: Draw, i: int) -> bool:
    want = export_weights(stored, i)
    if len(pairs) < len(want):
        return False
    want = want + [(0, 0.0)] * (len(pairs) - len(want))
    for (joint, weight), (bone, value) in zip(pairs, want):
        if weight != value or joint_map.bone(joint) != bone:
            return False
    return True


_FIELDS = ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw", "indices", "joints", "weights")
_FIELD_NAMES = {"positions": "positions", "normals": "normals", "colours_1": "colours (_D3_COLOR1)",
                "colours_2": "colours (_D3_COLOR2)", "uv1": "UV1", "uv2_raw": "UV2", "indices": "indices",
                "joints": "joints", "weights": "weights"}


def _equal(a: Draw, b: Draw) -> bool:
    return all(getattr(a, name) == getattr(b, name) for name in _FIELDS)


def _validate(app: AppearanceFile, draw: Draw, label: str) -> None:
    """Value checks of a changed draw before writing, so a refusal names its draw (W16-11)."""
    try:
        for i, p in enumerate(draw.positions):
            for c in p:
                M.check_float(c, f"vertex {i} position")
            M.normal_bytes(draw.normals[i])
            M.uv1_words(draw.uv1[i])
            if draw.skinned:
                for joint, weight in zip(draw.joints[i], draw.weights[i]):
                    M.check_word(weight, "B", f"vertex {i} weight")
                    if weight and not 0 <= joint < len(app.bones):
                        raise FormatError(f"vertex {i}: joint {joint} is not one of the {len(app.bones)} bones")
    except FormatError as exc:
        raise FormatError(f"{label}: {exc.message}") from None


def _empty_draw(sub: SubObject, k: int, build: str) -> Draw:
    """W16-7 ``missing="empty"`` (W17-13: kept for compatibility): the draw emptied in place. The
    writer (``derive=True``) drops an emptied ps3 draw or skinning record (W13-7) and removes an
    emptied rigid list and a sub-object left without geometry (W17-14). A sub-object with cloth
    data or fat vertices keeps every value (V21-1)."""
    stored = sub.draws[k]
    if keeps_every_value(sub):
        raise _kept_refusal(sub, k, "vertex count (emptying the draw)")
    return dataclasses.replace(stored, positions=[], normals=[], colours_1=[], colours_2=[], uv1=[], uv2_raw=[],
                               indices=[], joints=[] if stored.skinned else None,
                               weights=[] if stored.skinned else None, local_joints=None)


# --- new parts (W17-12) ---------------------------------------------------------------------------

def _normal_unit(value) -> tuple:
    length = math.sqrt(sum(c * c for c in value))
    return tuple(c / length for c in value) if length else (0.0, 0.0, 0.0)


def _uv_back(value, flip_v: bool) -> tuple:
    u, v = value
    return (float(u), 1.0 - v if flip_v else float(v))


def _nearest_slots(template: SubObject, positions, label: str) -> tuple[list, list]:
    """W17-12 ``skin="nearest"``: each vertex takes the joints and weights of the nearest vertex of
    the template part (ties: the first in draw order)."""
    pool = [(p, j, w) for d in template.draws if d.joints is not None and d.weights is not None
            for p, j, w in zip(d.positions, d.joints, d.weights)]
    if not pool:
        raise FormatError(f"{label}: the template part {template.label} has no skinned vertex to take the "
                          "skin from (W17-12)")
    joints, weights = [], []
    for p in positions:
        best = min(range(len(pool)), key=lambda q: (sum((p[c] - pool[q][0][c]) ** 2 for c in range(3)), q))
        joints.append(tuple(pool[best][1]))
        weights.append(tuple(pool[best][2]))
    return joints, weights


def _new_draw(app: AppearanceFile, src: _Source, opts: _Options, joint_map, template: SubObject, label: str,
              warnings: list[str]) -> Draw:
    """W17-12: the geometry of a primitive (OBJ object or group) that matches no draw, as a new
    draw of the file's kind (ps3 draw, or a switch / x360 packed list). Normals and UV1 are needed;
    colours not given are 0 (a warning), UV2 not given is (0x8000, 0x8000). In a file with bones
    the part needs joints and weights (W16-9), from OBJ only with ``skin="nearest"``; in a file
    without bones joints are refused."""
    n = len(src.positions)
    if src.has_joints and not app.bones:
        raise FormatError(f"{label}: a primitive with joints in a file without bones (W17-12)")
    for values, what in ((src.normals, "normals"), (src.uv1, "UV1 (texture coordinates)")):
        if values is None or any(v is None for v in values):
            raise FormatError(f"{label}: {what} missing for a new part (W17-12)")
    positions = [tuple(float(c) for c in p) for p in src.positions]
    normals = [_normal_unit(v) for v in src.normals]
    uv1 = [_uv_back(v, opts.flip_v) for v in src.uv1]
    if src.uv2_raw is not None:
        uv2 = [tuple(row) for row in src.uv2_raw]
    elif src.uv2_float is not None:
        factor = opts.lightmap_v_scale or 1.0
        uv2 = [(round(u * 8192 + 0x8000), round(v / factor * 8192 + 0x8000)) for u, v in src.uv2_float]
    else:
        uv2 = [(0x8000, 0x8000)] * n
    for i, (u, v) in enumerate(uv2):
        if not (0 <= u <= 0xFFFF and 0 <= v <= 0xFFFF):
            raise FormatError(f"{label}: vertex {i} UV2 {(u, v)} outside the u16 range")
    colours = []
    for given in (src.colours_1, src.colours_2):
        if given is None:
            colours.append([bytes(4)] * n)
        else:
            colours.append(list(given))
    if src.colours_1 is None or src.colours_2 is None:
        warnings.append(f"{label}: no colours given for a new part; its vertices get colour 0")
    joints = weights = None
    skinned = bool(app.bones)
    if skinned:
        if src.influences is not None:
            joints, weights = [], []
            for i, pairs in enumerate(src.influences):
                mapped = [(joint_map.need(joint, f"{label}: vertex {i}"), weight) for joint, weight in pairs if weight != 0]
                try:
                    j, w = weight_bytes(mapped)
                except FormatError as exc:
                    raise FormatError(f"{label}: vertex {i}: {exc.message}") from None
                joints.append(tuple(j))
                weights.append(tuple(w))
        elif src.form == "obj" and opts.skin == "nearest":
            joints, weights = _nearest_slots(template, positions, label)
        elif src.form == "obj":
            raise FormatError(f"{label}: a new skinned part from OBJ (the file has bones) is refused unless "
                              "skin='nearest' (W17-12)")
        else:
            raise FormatError(f"{label}: a new part in a file with bones needs joints and weights (W17-12, W16-9)")
    draw = Draw(skinned, positions, normals, colours[0], colours[1], uv1, uv2, list(src.indices), joints, weights,
                kind="ps3" if app.build == "ps3" else "packed")
    _validate(app, draw, label)
    return draw


def _appended(base: Draw | None, parts: list[Draw]) -> Draw:
    """W17-12 switch / x360 rigid: the vertices and triangles of ``parts`` appended to the single
    rigid list ``base`` (or, without one, joined as a new list)."""
    draws = ([base] if base is not None else []) + list(parts)
    values = {name: [v for d in draws for v in getattr(d, name)]
              for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw")}
    indices, offset = [], 0
    for d in draws:
        indices += [i + offset for i in d.indices]
        offset += len(d.positions)
    return dataclasses.replace(draws[0], indices=indices, **values)


def _all_subs(app: AppearanceFile) -> list[SubObject]:
    return list(app.sub_objects) + list(app.extra_sub_objects)


def _named_sub(app: AppearanceFile, info: dict | None, geo: int) -> SubObject | None:
    """The template sub-object named by ``extras.d3sno`` (``sub_object``, in ``geo_set`` when
    given, else in the new part's geo set)."""
    if info is None or not _is_int(info.get("sub_object")):
        return None
    g = info["geo_set"] if _is_int(info.get("geo_set")) else geo
    subs = (app.sub_objects, app.extra_sub_objects)[g] if g in (0, 1) else []
    return subs[info["sub_object"]] if 0 <= info["sub_object"] < len(subs) else None


def new_part_material(app: AppearanceFile, info: dict | None, textures, geo: int, template: SubObject | None,
                      look: int = 0, warnings: list | None = None, what: str = "new part") -> tuple[int | None, SubObject]:
    """W17-12: (material hash, template part) of a new sub-object. Material, first that applies:
    1. ``extras.d3sno.material`` when it is in the material table;
    2. the material of the sub-object named by ``extras.d3sno``;
    3. a primitive material ``texture_<handle>_draw_<k>``: the material of the first sub-object
       (geo set 0, then 1) whose export (``look``) uses that texture handle;
    4. the material of the template part (W17-10).
    Template part: ``template`` (the matched sub-object of an editor copy), else the sub-object
    named by ``extras.d3sno``, else W17-10: the first sub-object with the chosen material, else the
    file's first sub-object.

    ``38`` V23-7: the import creates no material entry. When ``extras.d3sno.material`` is a hash
    that is not in the table, rule 1 does not apply; a warning naming the part (``what``), the hash
    and the rule used instead is added to ``warnings``."""
    from .material import sub_object_texture

    everything = _all_subs(app)
    if not everything:
        raise FormatError("the template has no sub-object to copy values from (W17-10)")
    named = _named_sub(app, info, geo)
    material = None
    rule = None
    unknown = info.get("material") if info is not None and _is_int(info.get("material")) else None
    if unknown is not None and app.material_data.find(unknown) is not None:
        material, rule, unknown = unknown, 1, None  # rule 1
    elif named is not None and named.material_hash is not None:
        material, rule = named.material_hash, 2  # rule 2
    else:
        for handle in textures:  # rule 3
            user = next((s for s in everything if sub_object_texture(app, s, look) == handle), None)
            if user is not None and user.material_hash is not None:
                material, rule = user.material_hash, 3
                break
    part = template if template is not None else named
    if part is None:
        part = next((s for s in everything if material is not None and s.material_hash == material), everything[0])
    if material is None:
        material, rule = part.material_hash, 4  # rule 4
    if unknown is not None and warnings is not None:  # V23-7
        shown = "none" if material is None else f"0x{material:08X}"
        warnings.append(f"{what}: extras.d3sno.material 0x{unknown & 0xFFFFFFFF:08X} is not in the material table "
                        f"and the import creates no material entry (V23-7); material rule {rule} gave {shown}")
    return material, part


def new_part_name(name: str, build: str, taken: set) -> str:
    """W17-12: the mesh name without the ``lowpoly_`` prefix, ``.`` replaced by ``_``; when its
    W17-2 hash is taken (``taken``), ``_2``, ``_3``, ...: the smallest free number. ps3 names are
    ASCII of at most 127 bytes; a longer or non-ASCII name is refused (any build: W17-2 hashes ASCII
    bytes)."""
    base = name[len(LOWPOLY_PREFIX):] if name.startswith(LOWPOLY_PREFIX) else name
    base = base.replace(".", "_")
    if not base:
        raise FormatError(f"name {name!r} leaves no sub-object name (W17-12)")
    out = free_name(base, taken)
    if build == "ps3":
        ps3_name_field(out)
    return out


def _taken_hashes(subs, build: str) -> set:
    taken = set()
    for s in subs:
        if s.sub_hash is not None:
            taken.add(s.sub_hash)
        if build == "ps3" and s.name:
            try:
                taken.add(name_hash(s.name))
            except FormatError:
                pass
    return taken


def _new_sub(app: AppearanceFile, part: _NewPart, opts: _Options, index: int, taken: set,
             warnings: list[str]) -> SubObject:
    """W17-12: a new sub-object (origin None) from a mesh or OBJ object. ``38`` V23-6: an editor
    copy of a V21-1 sub-object (cloth record or fat vertices) is refused: cloth data cannot be
    copied by import."""
    if part.template is not None and keeps_every_value(part.template):
        raise FormatError(f"{part.what}: an editor copy of {part.template.label}, a sub-object with cloth data or "
                          "fat vertices; cloth data cannot be copied by import (V21-1, V23-6)")
    try:
        name = new_part_name(part.name, app.build, taken)
    except FormatError as exc:
        raise FormatError(f"{part.what}: {exc.message}") from None
    material, template = new_part_material(app, part.info, part.textures, part.geo_set, part.template, opts.look,
                                           warnings, part.what)
    draws = [_new_draw(app, src, opts, part.joint_map, template,
                       f"{part.what} primitive {q}" if src.form == "gltf" else part.what, warnings)
             for q, src in enumerate(part.sources)]
    if app.build != "ps3" and not app.bones and len(draws) > 1:
        draws = [_appended(None, draws)]  # one rigid list (W17-7, W17-12)
    taken.add(name_hash(name))
    return SubObject(index, name, None, (0.0,) * 3, (0.0,) * 3, None, None, draws, material_hash=material,
                     geo_set=part.geo_set, origin=None, template_part=(template.geo_set, template.index))


def _dropped_label(sub: SubObject, draw: Draw) -> str | None:
    """The writer's label of a template part emptied in place, which it drops (W13-7) or removes
    (W17-14): a ps3 draw or skinning record without vertices or triangles, a rigid list without
    vertices."""
    if draw.origin is None:
        return None
    if draw.kind == "ps3" and (not draw.positions or not draw.indices):
        return f"{sub.label}, draw {draw.origin}"
    if draw.kind == "packed" and draw.skinned and (not draw.positions or not draw.indices):
        return f"{sub.label}, skinning record {draw.origin}"
    if draw.kind == "packed" and not draw.skinned and not draw.positions:
        return f"{sub.label}, rigid list"
    return None


# --- the API --------------------------------------------------------------------------------------

MISSING = ("keep", "empty", "remove")  # W17-13


def _options(flip_v, lightmap_v_scale, missing, skin, look=0) -> _Options:
    if missing not in MISSING:
        raise ValueError(f"missing must be 'keep', 'empty' or 'remove', not {missing!r}")
    if skin not in ("refuse", "nearest"):
        raise ValueError(f"skin must be 'refuse' or 'nearest', not {skin!r}")
    if lightmap_v_scale is not None and (not isinstance(lightmap_v_scale, (int, float))
                                         or not math.isfinite(lightmap_v_scale)):
        raise ValueError(f"lightmap_v_scale {lightmap_v_scale!r} is not a finite number")
    if not _is_int(look) or look < 0:
        raise ValueError(f"look {look!r} is not an integer >= 0")
    return _Options(bool(flip_v), lightmap_v_scale, missing, skin, look)


def _document(document):
    if isinstance(document, (GltfDocument, ObjDocument)):
        return document
    if isinstance(document, str):
        return read_obj(document)
    if isinstance(document, dict):
        return read_gltf(json.dumps(document).encode("utf-8"))
    if isinstance(document, (bytes, bytearray, memoryview)):
        return read_gltf(bytes(document))
    raise ValueError(f"document of type {type(document).__name__}: give read_gltf / read_obj output")


def import_mesh_report(template_app, document, *, flip_v=False, lightmap_v_scale=None, missing="keep",
                       skin="refuse", low_poly=True, look=0) -> tuple[bytes, ImportReport]:
    """``import_mesh`` that also returns an ``ImportReport`` (changed and kept draw labels, added
    and removed parts, warnings)."""
    opts = _options(flip_v, lightmap_v_scale, missing, skin, look)
    app = template_app if isinstance(template_app, AppearanceFile) else parse_appearance(bytes(template_app))
    document = _document(document)
    report = ImportReport()
    candidates = export_sub_objects(app, low_poly)
    skipped = [] if low_poly else list(app.extra_sub_objects)
    if isinstance(document, GltfDocument):
        matches, new_parts = _gltf_matches(document, app, candidates, skipped, report)
    else:
        matches, new_parts = _obj_matches(document, app, candidates, skipped, report)

    changed: list[tuple[SubObject, int, Draw]] = []  # template draws changed in place
    plans: dict = {}  # id(sub-object) -> (draws, {template k: draw}) or None when it goes (W17-13)
    structural = False  # a part is added or removed
    for sub in candidates:
        match = matches.get((sub.geo_set, sub.index))
        if match is None and opts.missing == "remove" and any(d.positions for d in sub.draws):
            plans[id(sub)] = None  # W17-13: removed whole (W17-10; cloth too, V21-1)
            structural = True
            continue
        found = match.sources if match is not None else {}
        draws, by_k = [], {}
        for k, stored in enumerate(sub.draws):
            label = _label(sub, k)
            if k in found:
                draw = _make_draw(app, sub, k, found[k], opts, match.joint_map, report.warnings)
            elif not stored.positions:
                draw = stored
            elif opts.missing == "empty":
                draw = _empty_draw(sub, k, app.build)
            elif opts.missing == "remove":
                if keeps_every_value(sub):
                    raise _kept_refusal(sub, k, "draw list (removing the draw)")
                structural = True  # W17-13: the draw, skinning record or rigid list goes
                continue
            else:
                draw = stored
            if draw is stored:
                report.kept.append(label)
            else:
                report.changed.append(label)
                changed.append((sub, k, draw))
            draws.append(draw)
            by_k[k] = draw
        extra = match.extra if match is not None else []
        if extra:
            if keeps_every_value(sub):
                raise FormatError(f"{sub.label}: new geometry for a sub-object with cloth data or fat vertices, "
                                  "which is only removed whole or kept unchanged (V21-1, W17-10)")
            new = [_new_draw(app, src, opts, match.joint_map, sub, f"{sub.label}, new geometry {q}",
                             report.warnings) for q, src in enumerate(extra)]
            structural = True
            if app.build != "ps3" and not app.bones:  # W17-12: appended to the single rigid list
                at = next((p for p, d in enumerate(draws) if d.kind == "packed" and not d.skinned), None)
                if at is None:
                    draws.append(_appended(None, new))
                else:
                    k = next(k for k, d in by_k.items() if d is draws[at])
                    draws[at] = by_k[k] = _appended(draws[at], new)
                    label = _label(sub, k)
                    if label in report.kept:
                        report.kept.remove(label)
                        report.changed.append(label)
                    changed[:] = [c for c in changed if not (c[0] is sub and c[1] == k)] + [(sub, k, draws[at])]
            else:
                draws.extend(new)  # ps3: new draws cut by W17-5a; skinned: records rebuilt by W17-7
        plans[id(sub)] = (draws, by_k)
    for sub in skipped:
        report.kept.extend(_label(sub, k) for k in range(len(sub.draws)))

    lists = []
    for subs in (app.sub_objects, app.extra_sub_objects):
        out = []
        for sub in subs:
            plan = plans.get(id(sub), ())
            if plan is None:
                continue
            if plan == ():
                out.append(sub)
                continue
            draws, by_k = plan
            fat = sub.fat_draw
            for k, d in enumerate(sub.draws):
                if d is sub.fat_draw and k in by_k:
                    fat = by_k[k]
            out.append(dataclasses.replace(sub, draws=draws, fat_draw=fat))
        lists.append(out)
    taken = [_taken_hashes(lists[0], app.build), _taken_hashes(lists[1], app.build)]
    for part in new_parts:
        if part.geo_set == 1 and not low_poly:
            report.warnings.append(f"{part.what} is a new low-poly sub-object and is not imported (low_poly=False)")
            continue
        lists[part.geo_set].append(_new_sub(app, part, opts, len(lists[part.geo_set]), taken[part.geo_set],
                                            report.warnings))
        structural = True

    if not changed and not structural:
        return bytes(app.source), report  # W16-11 rule 1
    mesh = dataclasses.replace(app, sub_objects=lists[0], extra_sub_objects=lists[1])
    try:
        out, parts = M.write_app_mesh_report(app, mesh, derive=True, keep_skin_bytes=True)  # V21-2
    except FormatError as exc:
        if structural:
            raise
        for one in changed:  # W16-11 rule 3: name the draw the writer refuses
            try:
                _write(app, [one])
            except FormatError as single:
                if single.message.startswith(f"{one[0].label},"):  # ``38`` V23-9: the label once
                    raise FormatError(single.message) from None
                raise FormatError(f"{_label(one[0], one[1])}: {single.message}") from None
        raise FormatError(f"{'; '.join(_label(s, k) for s, k, _ in changed)}: {exc.message}") from None
    removed = [label for sub, _, draw in changed if (label := _dropped_label(sub, draw)) is not None]
    report.removed = removed + [label for label in parts.removed if label not in removed]
    report.added = list(parts.added)
    report.warnings.extend(f"{label}: a new part without geometry is not written" for label in parts.skipped)
    return out, report


def _write(app: AppearanceFile, changed) -> bytes:
    by_sub: dict = {}
    for sub, k, draw in changed:
        by_sub.setdefault(id(sub), (sub, {}))[1][k] = draw
    replaced = {}
    for sub, draws in by_sub.values():
        new_draws = [draws.get(k, d) for k, d in enumerate(sub.draws)]
        fat = sub.fat_draw
        for k, d in enumerate(sub.draws):
            if d is sub.fat_draw and k in draws:
                fat = draws[k]
        replaced[id(sub)] = dataclasses.replace(sub, draws=new_draws, fat_draw=fat)
    mesh = dataclasses.replace(
        app, sub_objects=[replaced.get(id(s), s) for s in app.sub_objects],
        extra_sub_objects=[replaced.get(id(s), s) for s in app.extra_sub_objects])
    return M.write_app_mesh(app, mesh, derive=True, keep_skin_bytes=True)  # V21-2


def import_mesh(template_app, document, *, flip_v=False, lightmap_v_scale=None, missing="keep",
                skin="refuse", low_poly=True, look=0) -> bytes:
    """W16-11, W17-12 to W17-14: the template ``.app`` (parsed, or its bytes) with the mesh of
    ``document`` (a ``read_gltf`` or ``read_obj`` result, or OBJ text). Unchanged: the template
    bytes as they are; else ``write_app_mesh(template, mesh, derive=True)``.

    W17-12: a mesh (OBJ ``o``) that matches no sub-object, and every editor copy of a matched one,
    becomes a new sub-object at the end of its geo set; a primitive (OBJ ``g``) that matches no draw
    is new geometry of its sub-object. W17-13 ``missing``: "keep" (default) keeps template parts
    the document does not hold, "empty" empties their draws (an emptied part, and a sub-object left
    without geometry, is removed), "remove" removes them. ``look`` is the look the export used for
    ``texture_<handle>_draw_<k>`` material names (W17-12 material rule 3)."""
    return import_mesh_report(template_app, document, flip_v=flip_v, lightmap_v_scale=lightmap_v_scale,
                              missing=missing, skin=skin, low_poly=low_poly, look=look)[0]


__all__ = [
    "GltfDocument", "ImportReport", "MISSING", "ObjDocument", "export_weights", "gltf_to_glb", "import_mesh",
    "import_mesh_report", "keeps_every_value", "new_part_material", "new_part_name", "read_gltf", "read_obj",
    "weight_bytes",
]
