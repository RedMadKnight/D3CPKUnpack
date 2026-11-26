"""Corrections 30 (``50_corrections30.md``): AA1 to AA4. Synthetic games of the test builders only; every
export and output goes to a temporary folder outside any git working tree."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import json
import unittest
from unittest import mock

import builders as B
import export_fixture as F
import test_stage21 as S
import d3sno
from d3sno import apply as AP
from d3sno import archive as A
from d3sno import export as E
from d3sno import json_import as J
from d3sno import sno_layout as L
from d3sno import world

BUILD = "switch"


def linked_markers() -> bytes:
    """A canonical marker set of three markers with links (two, one and two; the last of another type)."""
    data = B.build_mrk(BUILD, markers=[
        {"hash": 0x101, "type": 0, "group": 1, "handle": F.ACR, "position": (1.0, 2.0, 0.0), "links": [(1, 2), (3, 4)]},
        {"hash": 0x102, "type": 0, "group": 1, "handle": F.ACR, "position": (4.0, 5.0, 0.0), "links": [(5, 6)]},
        {"hash": 0x103, "type": 1, "group": 23, "handle": 5, "position": (7.0, 7.0, 0.0), "links": [(11, 12), (13, 14)]},
    ])
    return L.rebuild_sno(data, "mrk", BUILD)


def imported(data, edit):
    document = J._json_round(J.template_json("mrk", data, BUILD))
    edit(document)
    return J.import_json("mrk", data, BUILD, document)


class ListElements(unittest.TestCase):
    """AA2: changes inside the elements of a list that also changes length."""

    def setUp(self):
        self.data = linked_markers()

    def test_added_marker_and_changed_link(self):
        def edit(d):
            new = copy.deepcopy(d["markers"][1])  # a copy of the nearest earlier marker of its type (W22-4)
            new["position"] = [9.0, 9.0, 9.0]
            d["markers"].append(new)
            d["markers"][1]["links"][0][1] = 77

        result = imported(self.data, edit)
        markers = world.parse_marker_set(result.data, BUILD).markers
        self.assertEqual([m.position for m in markers], [(1.0, 2.0, 0.0), (4.0, 5.0, 0.0), (7.0, 7.0, 0.0), (9.0, 9.0, 9.0)])
        self.assertEqual([m.links for m in markers], [[(1, 2), (3, 4)], [(5, 77)], [(11, 12), (13, 14)], [(5, 6)]])
        self.assertEqual({row["path"] for row in result.changes}, {"markers"})

    def test_inserted_before_a_changed_marker(self):
        def edit(d):
            new = copy.deepcopy(d["markers"][1])
            new["position"] = [0.5, 0.5, 0.5]
            d["markers"].insert(0, new)
            d["markers"][2]["links"][0][0] = 50

        markers = world.parse_marker_set(imported(self.data, edit).data, BUILD).markers
        self.assertEqual([m.position for m in markers], [(0.5, 0.5, 0.5), (1.0, 2.0, 0.0), (4.0, 5.0, 0.0), (7.0, 7.0, 0.0)])
        self.assertEqual(markers[2].links, [(50, 6)])
        self.assertEqual(markers[0].links, [(5, 6)])

    def test_new_element_leaves(self):
        """A new marker's own leaves (its link value) are set on the copy."""
        def edit(d):
            new = copy.deepcopy(d["markers"][1])
            new["links"][0] = [8, 9]
            d["markers"].append(new)

        markers = world.parse_marker_set(imported(self.data, edit).data, BUILD).markers
        self.assertEqual([m.links for m in markers], [[(1, 2), (3, 4)], [(5, 6)], [(11, 12), (13, 14)], [(8, 9)]])

    def test_new_element_second_link(self):
        """A copy of the marker of type 1 (its nearest earlier marker of that type) with its second link changed."""
        def edit(d):
            new = copy.deepcopy(d["markers"][2])
            new["links"][1] = [21, 22]
            d["markers"].append(new)

        markers = world.parse_marker_set(imported(self.data, edit).data, BUILD).markers
        self.assertEqual(markers[3].links, [(11, 12), (21, 22)])
        self.assertEqual(markers[2].links, [(11, 12), (13, 14)])

    def test_list_fault_names_the_path(self):
        with mock.patch.object(J, "_match_elements", side_effect=TypeError("bad")):
            with self.assertRaises(J.Refused) as caught:
                imported(self.data, lambda d: d["markers"].pop())
        self.assertEqual((caught.exception.reason, J.path_text(caught.exception.path)), ("import_failed", "markers"))

    def test_nested_list_length_fixed(self):
        def edit(d):
            d["markers"].append(copy.deepcopy(d["markers"][0]))
            d["markers"][1]["links"].append([7, 8])

        with self.assertRaises(J.Refused) as caught:
            imported(self.data, edit)
        self.assertEqual(caught.exception.reason, "list_length_fixed")
        self.assertEqual(J.path_text(caught.exception.path), "markers[1].links")
        with self.assertRaises(J.Refused) as caught:
            imported(self.data, lambda d: d["markers"][1]["links"].append([7, 8]))
        self.assertEqual((caught.exception.reason, J.path_text(caught.exception.path)),
                         ("list_length_fixed", "markers[1].links"))

    def test_derived_leaf_in_an_element(self):
        def edit(d):
            d["markers"].append(copy.deepcopy(d["markers"][1]))
            d["markers"][1]["scale"] = 3.0

        with self.assertRaises(J.Refused) as caught:
            imported(self.data, edit)
        self.assertEqual((caught.exception.reason, J.path_text(caught.exception.path)), ("derived_field", "markers[1].scale"))


class Faults(unittest.TestCase):
    """AA1: an unexpected error refuses one entry; ``apply`` goes on."""

    def test_leaf_error_names_the_path(self):
        data = linked_markers()
        with mock.patch.object(J, "_write_change", side_effect=ZeroDivisionError("boom")):
            with self.assertRaises(J.Refused) as caught:
                imported(data, lambda d: d["markers"][0]["position"].__setitem__(0, 3.0))
        self.assertEqual(caught.exception.reason, "import_failed")
        self.assertEqual(J.path_text(caught.exception.path), "markers[0].position[0]")
        self.assertIn("ZeroDivisionError", caught.exception.message)

    def test_apply_goes_on(self):
        setup = S.Setup(self)
        for rel, change in ((f"data/acr/{F.ACR}.json", lambda d: d["fields"].__setitem__("bound_sphere_radius", 2.0)),
                            (f"data/mrk/{F.MRK}.json", lambda d: d["markers"][0]["position"].__setitem__(2, 7.0))):
            path = setup.edit / rel
            document = json.loads(path.read_text(encoding="utf-8"))
            change(document)
            path.write_text(json.dumps(document), encoding="utf-8")
        real = J.import_json

        def failing(ext, *args, **kw):
            if ext == "acr":
                raise RuntimeError("an importer fault")
            return real(ext, *args, **kw)

        with mock.patch.object(J, "import_json", failing):
            result, out = setup.apply()
        self.assertEqual(result.exit_code, 0, result.message)
        rows = {r["path"]: r for r in result.report["changed"]}
        acr = rows[f"data/acr/{F.ACR}.json"]
        self.assertEqual((acr["result"], acr["reason"]), ("refused", "import_failed"))
        self.assertIn("RuntimeError", acr["message"])
        self.assertEqual(rows[f"data/mrk/{F.MRK}.json"]["result"], "applied")
        self.assertEqual(result.report["archives_written"], ["Act1.cpk"])
        self.assertTrue((out / "Act1.cpk").is_file())

    def test_error_outside_the_importer(self):
        setup = S.Setup(self)
        S.move_vertex(setup.edit / S.MESH_PATH)
        with mock.patch.object(AP._Apply, "targets", side_effect=RuntimeError("targets")):
            result, _ = setup.apply()
        self.assertEqual(result.exit_code, 0)
        (row,) = result.report["changed"]
        self.assertEqual((row["result"], row["reason"]), ("refused", "import_failed"))

    def test_any_importer_error(self):
        setup = S.Setup(self)
        S.move_vertex(setup.edit / S.MESH_PATH)
        with mock.patch.object(AP._Apply, "import_mesh", side_effect=AttributeError("no such attribute")):
            result, _ = setup.apply()
        self.assertEqual(result.exit_code, 0)
        (row,) = result.report["changed"]
        self.assertEqual((row["result"], row["reason"]), ("refused", "import_failed"))
        self.assertIn("AttributeError", row["message"])


def repeated_game(root, variant: bytes):
    """The switch game with the marker set stored twice in ``Act1``: the template and ``variant``."""
    files = F.switch_files()
    game = F.write_game(root, files)
    act1 = files["Act1"]
    name = F.name_of("mrk", "Room")
    inputs = [A.InputFile(n, [d, variant] if n == name else [d]) for n, d in act1.items()]
    (game / "Act1.cpk").write_bytes(A.write_archive(inputs))
    return game


class RepeatedCopies(unittest.TestCase):
    """AA3, AA4: repeated copies of one archive entry are one file."""

    def setUp(self):
        self.mrk = F.switch_files()["Act1"][F.name_of("mrk", "Room")]
        variant = bytearray(self.mrk)
        variant[-1] ^= 0xFF  # a copy that differs from the first
        self.root = F.temp_root(self)
        self.game = repeated_game(self.root / "game", bytes(variant))
        export = self.root / "export"
        result = E.export_game(self.game, export, created="X")
        self.assertEqual(result.exit_code, 0, result.message)
        self.export = export

    def edit(self):
        path = self.export / f"data/mrk/{F.MRK}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["markers"][0]["position"][2] = 7.0
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_all_copies_replaced(self):
        self.edit()
        result = AP.apply_export(self.game, self.export, self.root / "out")
        self.assertEqual(result.exit_code, 0, result.message)
        (row,) = [r for r in result.report["changed"] if r["group"] == "mrk"]
        self.assertEqual(row["result"], "applied")
        self.assertIn("repeated_copies_differed: archive Act1", row["warnings"])
        cpk = A.read_archive((self.root / "out" / "Act1.cpk").read_bytes())
        entry = cpk.find(F.name_of("mrk", "Room"))
        copies = [cpk.read_file(entry, i) for i in range(entry.count)]
        self.assertEqual(len(copies), 2)
        self.assertEqual(copies[0], copies[1])
        self.assertEqual(world.parse_marker_set(copies[0], BUILD).markers[0].position, (1.0, 2.0, 7.0))

    def test_no_copy_matches(self):
        self.edit()
        with mock.patch.object(AP._Apply, "targets", lambda job, item, entry, template: (
                [], ["copy_differs_not_changed: archive Act1"])):
            result = AP.apply_export(self.game, self.export, self.root / "out")
        (row,) = [r for r in result.report["changed"] if r["group"] == "mrk"]
        self.assertEqual((row["result"], row["reason"]), ("refused", "no_copy_matches"))
        self.assertEqual(result.report["archives_written"], [])


class NoEffect(unittest.TestCase):
    """AA4: ``no_effect`` only when the import gives the template's bytes."""

    def test_reformatted_json(self):
        setup = S.Setup(self)
        path = setup.edit / f"data/acr/{F.ACR}.json"
        path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=4), encoding="utf-8")
        report, _ = setup.ok()
        (row,) = report["changed"]
        self.assertEqual((row["result"], row["fields_changed"], row["archives"]), ("no_effect", 0, []))
        self.assertEqual(report["archives_written"], [])


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 22, 1))  # later ones too
        self.assertEqual(d3sno.STAGE, 22)


if __name__ == "__main__":
    unittest.main()
