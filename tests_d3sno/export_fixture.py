"""Synthetic game folders for the stage-18 export tests (``40_export.md``). Nothing here comes from
game files: every input is built by the test builders, in a temporary folder outside any git tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import tempfile
from pathlib import Path

import builders as B
from d3sno import archive as A
from d3sno import index_write as IW
from d3sno import texture_write as TW
from d3sno.texture import FORMAT_NORMAL, Image
from d3sno.index import TocEntry
from d3sno.sno import group_by_extension
from test_stage12 import BONES
from test_stage17_writer import laid_out, switch_skin, with_materials

MIP, NORMAL = 502, 503
TEX, CUBE, APP, PLAIN_APP, ANI, LONE_ANI, ANS, ACR, MON, MAT, SCN, MRK, WRL, PHM, STL, LVL, BROKEN = (
    500, 501, 600, 601, 700, 701, 800, 100, 110, 1100, 1200, 1300, 1400, 1000, 1500, 1600, 1700)
MISSING_TEX = 999


def with_handle(data: bytes, handle: int, order: str) -> bytes:
    out = bytearray(data)
    struct.pack_into(order + "I", out, 0x10, handle)
    return bytes(out)


def name_of(ext: str, stem: str, locale: str | None = None) -> str:
    group = group_by_extension(ext)
    return f"{group.name}\\{stem}.{ext}" + (f"|{locale}" if locale else "")


def textured_mesh(texture_handle=TEX):
    """A skinned switch mesh whose sub-object material names ``texture_handle`` as diffuse."""
    data, build = switch_skin()
    return with_materials(data, build, looks=(0x61,), entries=[(0xAAA, [[(1, texture_handle)]])], assign=(0xAAA,))


def three_cell_scene(build="switch"):
    order = B.order_of(build)
    inner = 3 * 3
    sections = B.nav_sections(order, inner, cell_flags=(0x1001, 0x0002, 0x0004))
    scn, nav = B.build_scene(build, sections=sections)
    img = B.Image(0, order)
    img.b = bytearray(scn)
    img.pair(0x10 + 0x120, struct.pack(order + "I", MRK))  # marker sets (body 0x120)
    if build == "switch":
        img.put(0x10 + 0x20C, "ii", APP, PHM)
    else:
        img.put(0x10 + 0x210, "ii", APP, PHM)
    return img.done(), nav


def collision_mesh(build="switch"):
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    return B.build_phm(build, [{
        "centre": (0.0, 0.0, 0.0), "vertices": vertices,
        "triangles": [B.triangle_row(build, (0, 1, 2), material=4, flags=5), B.triangle_row(build, (1, 3, 2))],
        "nodes": [B.tree_internal(build, 2), B.tree_leaf(build, 0, 1), B.tree_leaf(build, 1, 1)],
    }])


def switch_files() -> dict[str, dict[str, bytes]]:
    """Two archives (``Common``, ``Act1``) and a cutscene archive, switch build."""
    o = "<"
    mesh = with_handle(textured_mesh(), APP, o)
    plain = with_handle(textured_mesh(MISSING_TEX), PLAIN_APP, o)
    tex = with_handle(B.build_tex("switch", 0, 4, 4, bytes(range(64))), TEX, o)
    ani = with_handle(B.build_ani("switch", [B.simple_permutation("switch"),
                                             B.simple_permutation("switch", name="Run")]), ANI, o)
    lone = with_handle(B.build_ani("switch", [B.simple_permutation("switch")]), LONE_ANI, o)
    ans = with_handle(B.build_ans("switch", default=[(2, 0x10, ANI)]), ANS, o)
    acr = with_handle(B.build_record("switch", "actor", {
        "self_handle": ACR, "appearance_ref": APP, "anim_set_ref": ANS, "physics_mesh_ref": PHM,
        "monster_ref": -1, "anim_tree_ref": -1, "actor_kind": 1, "bound_sphere_radius": 0.5}), ACR, o)
    mon = with_handle(B.build_record("switch", "monster", {"self_handle": MON, "actor_ref": ACR}), MON, o)
    mat = with_handle(B.build_mat("switch", textures=[(1, TEX), (3, MISSING_TEX)]), MAT, o)
    scn, _ = three_cell_scene()
    scn = with_handle(scn, SCN, o)
    mrk = with_handle(B.build_mrk("switch", markers=[
        {"type": 0, "group": 1, "handle": ACR, "position": (1.0, 2.0, 0.0)},
        {"type": 0, "group": 1, "handle": ACR, "position": (4.0, 5.0, 0.0), "rotation": (0.0, 0.0, 0.6, 0.8)},
    ]), MRK, o)
    wrl = with_handle(B.build_wrl("switch", chunks=[[B.scene_entry("switch", SCN, position=(100.0, 200.0, 0.0))]]),
                      WRL, o)
    phm = with_handle(collision_mesh(), PHM, o)
    stl = with_handle(B.build_stl("switch", [(0x1234, "Hello".encode()), (0x99, "Zdanie ż".encode())]), STL, o)
    lvl = with_handle(B.sno_head(1, o) + bytes(16), LVL, o)
    cube = with_handle(TW.encode_texture([image(4, 4, k) for k in range(6)], "switch", 0, levels=1, faces=6), CUBE, o)
    mip = with_handle(TW.encode_texture(image(8, 8, 9), "switch", 0, levels=3), MIP, o)
    normal = with_handle(TW.encode_texture(image(4, 4, 3), "switch", FORMAT_NORMAL, levels=1), NORMAL, o)
    broken = with_handle(B.sno_head(1, o) + bytes(16), BROKEN, o)  # a texture far too short
    toc_entries = [TocEntry(group_by_extension(ext).id, handle, stem) for ext, handle, stem in (
        ("tex", TEX, "Stone"), ("tex", CUBE, "Sky"), ("app", APP, "Hero"), ("app", PLAIN_APP, "Rock"),
        ("ani", ANI, "Hero_walk"), ("ani", LONE_ANI, "Lone"), ("ans", ANS, "Hero"), ("acr", ACR, "Hero"),
        ("mon", MON, "Hero"), ("mat", MAT, "Stone"), ("scn", SCN, "Room"), ("mrk", MRK, "Room"),
        ("wrl", WRL, "Town"), ("phm", PHM, "Room"), ("stl", STL, "Words"), ("lvl", LVL, "Town"),
        ("tex", BROKEN, "Broken"), ("tex", MIP, "Mip"), ("tex", NORMAL, "Bumps"))]
    toc = IW.write_core_toc(toc_entries, "switch")
    common = {
        name_of("tex", "Stone"): tex, name_of("app", "Hero"): mesh, name_of("app", "Rock"): plain,
        name_of("ans", "Hero"): ans, name_of("acr", "Hero"): acr, name_of("mat", "Stone"): mat,
        name_of("stl", "Words", "enUS"): stl, "CoreTOC.dat": toc, name_of("tex", "Broken"): broken,
        name_of("tex", "Sky"): cube, name_of("tex", "Mip"): mip, name_of("tex", "Bumps"): normal,
    }
    act1 = {
        name_of("tex", "Stone"): tex,  # the same file in a second archive: written once
        name_of("ani", "Hero_walk"): ani, name_of("ani", "Lone"): lone, name_of("mon", "Hero"): mon,
        name_of("scn", "Room"): scn, name_of("mrk", "Room"): mrk, name_of("wrl", "Town"): wrl,
        name_of("phm", "Room"): phm, name_of("lvl", "Town"): lvl,
    }
    cutscene = {name_of("tex", "Movie"): tex}
    return {"Common": common, "Act1": act1, "enUS_Cutscene": cutscene}


def image(w, h, seed):
    return Image(w, h, bytes((seed * 37 + i * 11) % 256 for i in range(w * h * 4)))


def ps3_files() -> dict[str, dict[str, bytes]]:
    """A ps3 set: a scene with its ``.nav`` and one without, an animation with its ``.can``, the
    mesh and actor that link it."""
    from test_stage12 import ps3
    from test_stage17_writer import canon

    o = ">"
    perms = [B.simple_permutation("ps3")]
    mesh = with_handle(canon(ps3(skinned=True, bones=BONES), "ps3"), APP, o)
    ani = with_handle(B.build_ani("ps3", perms), ANI, o)
    ans = with_handle(B.build_ans("ps3", default=[(2, 0x10, ANI)]), ANS, o)
    acr = with_handle(B.build_record("ps3", "actor", {"self_handle": ACR, "appearance_ref": APP,
                                                      "anim_set_ref": ANS}), ACR, o)
    scn, nav = three_cell_scene("ps3")
    lone_scn, _ = three_cell_scene("ps3")
    toc = IW.write_core_toc([TocEntry(group_by_extension(e).id, h, n) for e, h, n in (
        ("app", APP, "Hero"), ("ani", ANI, "Hero_walk"), ("ans", ANS, "Hero"), ("acr", ACR, "Hero"),
        ("scn", SCN, "Room"), ("scn", SCN + 1, "Hall"))], "ps3")
    return {"Common": {
        name_of("app", "Hero"): mesh, name_of("ani", "Hero_walk"): ani, "Anim\\Hero_walk\\0000.can": B.build_can(perms),
        name_of("ans", "Hero"): ans, name_of("acr", "Hero"): acr, name_of("scn", "Room"): with_handle(scn, SCN, o),
        "Scene\\Room\\0000.nav": nav, name_of("scn", "Hall"): with_handle(lone_scn, SCN + 1, o), "CoreTOC.dat": toc,
    }}


def write_game(root: Path, files: dict[str, dict[str, bytes]], version=7, order="<", **options) -> Path:
    """``options`` go to ``archive.write_archive`` (``47`` tests: ``level`` 1 for large archives)."""
    root.mkdir(parents=True, exist_ok=True)
    for stem, content in files.items():
        (root / f"{stem}.cpk").write_bytes(A.write_archive(content, version=version, order=order, **options))
    return root


def temp_root(case) -> Path:
    """A temporary folder, removed after the test; refused if it lies in a git working tree."""
    from d3sno.export import in_git_tree

    holder = tempfile.TemporaryDirectory(prefix="d3sno_export_")
    case.addCleanup(holder.cleanup)
    root = Path(holder.name)
    if in_git_tree(root) is not None:
        case.skipTest("the temporary folder lies inside a git working tree")
    return root
