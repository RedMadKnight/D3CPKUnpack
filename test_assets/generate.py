"""Original test assets ("programmer art") and their console builds.

Everything here is made from scratch by this script: a checker texture, a box mesh, a box
collision mesh, a material, a short text table and a short animation. Nothing is read or
derived from game files. The console files are built with the d3sno writers.

    python -I -B test_assets/generate.py sources          # rewrite test_assets/sources/
    python -I -B test_assets/generate.py check            # sources on disk equal the generated ones
    python -I -B test_assets/generate.py build OUT_DIR    # write the console files per build
    python -I -B test_assets/generate.py OUT_DIR          # the same

``build`` writes ``OUT_DIR/<build>/<archive path>``, ``OUT_DIR/<build>/CoreTOC.dat`` and
``OUT_DIR/<build>/test_assets.cpk``. Keep OUT_DIR outside the repository (or under
``test_assets/out/``, which is ignored).
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SOURCES = os.path.join(HERE, "sources")
if REPO not in sys.path:
    sys.path.insert(0, REPO)  # python -I adds neither the current folder nor PYTHONPATH

from d3sno import anim_write  # noqa: E402
from d3sno import animation  # noqa: E402
from d3sno import archive  # noqa: E402
from d3sno import collision_write  # noqa: E402
from d3sno import index_write  # noqa: E402
from d3sno import material  # noqa: E402
from d3sno import material_write  # noqa: E402
from d3sno import sno_layout  # noqa: E402
from d3sno import texture  # noqa: E402
from d3sno import texture_write  # noqa: E402
from d3sno.png import read_png  # noqa: E402
from d3sno.sno import group_by_extension, hash33  # noqa: E402

BUILDS = ("ps3", "switch", "x360")

# Archive form per build: (version, byte order).
ARCHIVE_FORM = {"ps3": (6, ">"), "switch": (7, "<"), "x360": (7, ">")}

# Our own SNO handles, fixed so every build is reproducible. Chosen from a high range that a
# real file list is unlikely to use; they are not taken from any game index.
HANDLES = {
    "tex": 0x7A550001,
    "mat": 0x7A550002,
    "phm": 0x7A550003,
    "ani": 0x7A550004,
}
NAMES = {
    "tex": "TA_Checker64",
    "mat": "TA_CheckerMaterial",
    "phm": "TA_BoxCollision",
    "ani": "TA_BoxBob",
}

# DXT1 (format code 9). The two checker colours are exact in RGB565, so the top level decodes
# back to the source pixels.
TEXTURE_FORMAT = 9

SOURCE_FILES = (
    "checker_64.png",
    "box.obj",
    "box_collision.obj",
    "material.json",
    "strings.tsv",
    "anim_box_bob.json",
)


# --- sources -----------------------------------------------------------------------------------

def checker_pixels(size: int = 64, cell: int = 8) -> texture.Image:
    """A magenta / dark grey checker with a white grid band, 4 pixels wide, every 4 cells.

    Every 4 x 4 block holds one colour, and each colour is exact in RGB565, so DXT1 stores the
    top level without loss."""
    a = (0xFF, 0x00, 0xFF, 0xFF)
    b = (0x21, 0x20, 0x21, 0xFF)
    line = (0xFF, 0xFF, 0xFF, 0xFF)
    out = bytearray()
    for y in range(size):
        for x in range(size):
            if x % (cell * 4) < 4 or y % (cell * 4) < 4:
                out += bytes(line)
            else:
                out += bytes(a if ((x // cell) + (y // cell)) % 2 == 0 else b)
    return texture.Image(size, size, bytes(out))


def box_geometry(half: float = 1.0):
    """A cube centred on the origin, z up: 8 corners, 12 triangles wound outward (CCW)."""
    h = half
    corners = [(x, y, z) for z in (-h, h) for y in (-h, h) for x in (-h, h)]
    # quads as corner indices, counter-clockwise seen from outside
    quads = [
        (0, 2, 3, 1),  # -z (bottom)
        (4, 5, 7, 6),  # +z (top)
        (0, 1, 5, 4),  # -y
        (2, 6, 7, 3),  # +y
        (0, 4, 6, 2),  # -x
        (1, 3, 7, 5),  # +x
    ]
    normals = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
    return corners, quads, normals


def _num(v: float) -> str:
    text = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def box_obj() -> bytes:
    corners, quads, normals = box_geometry()
    lines = ["# test_assets: original box mesh, made by generate.py (2 x 2 x 2, z up)",
             "o TA_Box"]
    lines += [f"v {_num(x)} {_num(y)} {_num(z)}" for x, y, z in corners]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    lines += [f"vn {_num(x)} {_num(y)} {_num(z)}" for x, y, z in normals]
    lines.append("usemtl TA_CheckerMaterial")
    for face, quad in enumerate(quads):
        a, b, c, d = (i + 1 for i in quad)
        n = face + 1
        lines.append(f"f {a}/1/{n} {b}/2/{n} {c}/3/{n}")
        lines.append(f"f {a}/1/{n} {c}/3/{n} {d}/4/{n}")
    return ("\n".join(lines) + "\n").encode("ascii")


def box_collision_obj() -> bytes:
    corners, quads, _ = box_geometry()
    lines = ["# test_assets: original box collision mesh, made by generate.py (positions and triangles)",
             "o TA_BoxCollision"]
    lines += [f"v {_num(x)} {_num(y)} {_num(z)}" for x, y, z in corners]
    for a, b, c, d in quads:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
        lines.append(f"f {a + 1} {c + 1} {d + 1}")
    return ("\n".join(lines) + "\n").encode("ascii")


def material_json() -> bytes:
    doc = {
        "name": NAMES["mat"],
        "shader_map": -1,
        "colours": {
            "ambient": [0.2, 0.2, 0.2, 1.0],
            "diffuse": [1.0, 1.0, 1.0, 1.0],
            "specular": [0.5, 0.5, 0.5, 1.0],
            "emissive": [0.0, 0.0, 0.0, 1.0],
        },
        "power": 8.0,
        "sort": 0,
        "textures": [{"slot": "diffuse", "texture": NAMES["tex"]}],
    }
    return (json.dumps(doc, indent=2) + "\n").encode("ascii")


STRINGS = (
    ("TA_Hello", "Hello from the test assets."),
    ("TA_BoxName", "Test Box"),
    ("TA_BoxDesc", "A plain 2 x 2 x 2 box with a checker texture."),
    ("TA_Empty", ""),
    ("TA_Accents", "Zażółć gęślą jaźń"),
)


def strings_tsv() -> bytes:
    lines = ["label\ttext"] + [f"{label}\t{text}" for label, text in STRINGS]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _yaw(degrees: float):
    half = math.radians(degrees) / 2
    return [0.0, 0.0, round(math.sin(half), 6), round(math.cos(half), 6)]


def anim_json() -> bytes:
    doc = {
        "name": NAMES["ani"],
        "fps": 30,
        "keyframes": 31,
        "tracks": [
            {
                "bone": "TA_Root",
                "translation": [[0, [0.0, 0.0, 0.0]], [15, [0.0, 0.0, 0.5]], [30, [0.0, 0.0, 0.0]]],
                "rotation": [[0, _yaw(0)], [15, _yaw(45)], [30, _yaw(90)]],
                "scale": [[0, [1.0]]],
            },
        ],
    }
    return (json.dumps(doc, indent=2) + "\n").encode("ascii")


def make_sources() -> dict[str, bytes]:
    return {
        "checker_64.png": texture.encode_png(checker_pixels()),
        "box.obj": box_obj(),
        "box_collision.obj": box_collision_obj(),
        "material.json": material_json(),
        "strings.tsv": strings_tsv(),
        "anim_box_bob.json": anim_json(),
    }


def read_sources(folder: str = SOURCES) -> dict[str, bytes]:
    out = {}
    for name in SOURCE_FILES:
        with open(os.path.join(folder, name), "rb") as f:
            out[name] = f.read()
    return out


def same_source(name: str, a: bytes, b: bytes) -> bool:
    """PNG files are compared by pixels (zlib output may differ between builds of zlib)."""
    if name.endswith(".png"):
        pa, pb = read_png(a), read_png(b)
        return (pa.width, pa.height, pa.rgba) == (pb.width, pb.height, pb.rgba)
    return a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


# --- source readers ----------------------------------------------------------------------------

def parse_obj(data: bytes):
    """Positions and triangles of an OBJ (``v`` and ``f`` lines; polygons fanned)."""
    positions, triangles = [], []
    for raw in data.decode("ascii").splitlines():
        parts = raw.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v":
            positions.append(tuple(float(p) for p in parts[1:4]))
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            for i in range(1, len(idx) - 1):
                triangles.append((idx[0], idx[i], idx[i + 1]))
    return positions, triangles


def parse_strings(data: bytes) -> list[tuple[str, str]]:
    lines = data.decode("utf-8").replace("\r\n", "\n").split("\n")
    rows = []
    for line in lines[1:]:
        if line:
            label, text = line.split("\t", 1)
            rows.append((label, text))
    return rows


# --- console builds ----------------------------------------------------------------------------

def identity(ext: str):
    return index_write.sno_identity(group_by_extension(ext).id, NAMES[ext], HANDLES[ext])


def build_tex(sources, build: str) -> bytes:
    image = read_png(sources["checker_64.png"])
    data = texture_write.encode_texture(image, build, TEXTURE_FORMAT)
    return identity("tex").stamp(data)


def _colour(values, build: str):
    if build == "ps3":  # ((r, g, b, a) bytes, float scale)
        return (tuple(max(0, min(255, round(v * 255))) for v in values), 1.0)
    return tuple(float(v) for v in values)


def build_mat(sources, build: str) -> bytes:
    doc = json.loads(sources["material.json"])
    slots = {name: number for number, name in material.SLOT_NAMES.items()}
    handles = {NAMES["tex"]: HANDLES["tex"]}
    colours = [_colour(doc["colours"][k], build) for k in ("ambient", "diffuse", "specular", "emissive")]
    entries = [(slots[t["slot"]], handles[t["texture"]]) for t in doc["textures"]]
    tree = material_write.new_material(build, doc["shader_map"], colours, doc["power"], doc["sort"],
                                       entries, handle=HANDLES["mat"])
    return sno_layout.write_tree(tree)


def build_phm(sources, build: str) -> bytes:
    positions, triangles = parse_obj(sources["box_collision.obj"])
    mesh = collision_write.build_collision_mesh(positions, triangles, 0, 0, build)
    return collision_write.write_collision(
        collision_write.new_collision_file([mesh], build, handle=HANDLES["phm"]))


def build_ani(sources, build: str) -> bytes | None:
    """Switch only: a new ps3 / x360 ``.ani`` needs a parent record table and a ``.can`` child,
    and d3sno has no from-scratch writer for those (see README)."""
    if build != "switch":
        return None
    doc = json.loads(sources["anim_box_bob.json"])
    tracks = doc["tracks"]
    perm = anim_write.new_permutation(build, doc["name"], doc["keyframes"], doc["fps"] / 60,
                                      [hash33(t["bone"]) for t in tracks])

    def curve(keys):
        return animation.Curve([k[0] for k in keys], [tuple(float(c) for c in k[1]) for k in keys])

    curves = animation.DecodedPermutation(
        [curve(t["translation"]) for t in tracks],
        [curve(t["rotation"]) for t in tracks],
        [curve(t["scale"]) for t in tracks],
    )
    anim_write.encode_animation([(perm, curves)], build)
    return identity("ani").stamp(anim_write.write_ani([perm], build))


BUILDERS = {"tex": build_tex, "mat": build_mat, "phm": build_phm, "ani": build_ani}


def build_files(sources, build: str) -> dict[str, bytes]:
    """Archive path -> SNO file bytes, for every type the library can create on ``build``."""
    out = {}
    for ext, make in BUILDERS.items():
        data = make(sources, build)
        if data is not None:
            out[identity(ext).path] = data
    return out


def toc_entries(files: dict[str, bytes]):
    by_path = {identity(ext).path: identity(ext).entry for ext in BUILDERS}
    return [by_path[p] for p in files]


def build_all(sources=None) -> dict[str, dict[str, bytes]]:
    """{build: {name: bytes}}: the SNO files, ``CoreTOC.dat`` and ``test_assets.cpk``."""
    sources = read_sources() if sources is None else sources
    result = {}
    for build in BUILDS:
        files = build_files(sources, build)
        toc = index_write.write_core_toc(toc_entries(files), build)
        version, order = ARCHIVE_FORM[build]
        cpk = archive.write_archive(files, version=version, order=order)
        result[build] = dict(files, **{"CoreTOC.dat": toc, "test_assets.cpk": cpk})
    return result


# --- command line ------------------------------------------------------------------------------

def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 1 and args[0] not in ("sources", "check", "build"):
        args = ["build", args[0]]  # a lone folder: build into it
    if not args or args[0] not in ("sources", "check", "build") or (args[0] == "build") != (len(args) == 2):
        print(__doc__)
        return 2
    if args[0] == "sources":
        for name, data in make_sources().items():
            _write(os.path.join(SOURCES, name), data)
        print(f"wrote {len(SOURCE_FILES)} files to {SOURCES}")
        return 0
    if args[0] == "check":
        on_disk = read_sources()
        bad = [n for n, d in make_sources().items() if not same_source(n, d, on_disk[n])]
        for n in bad:
            print(f"differs: {n}")
        return 1 if bad else 0
    for build, files in build_all().items():
        for name, data in files.items():
            _write(os.path.join(args[1], build, *name.split("\\")), data)
            print(f"{build}: {name} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
