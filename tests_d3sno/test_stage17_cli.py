"""Stage 17 command line (``37_mesh_parts.md`` section 4): ``mesh-import --missing keep|empty|remove``
with the added and removed parts printed (W17-14), and the new ``mesh-transplant`` command (W17-11).
Synthetic files only."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import builders as B
from d3sno import app_mesh as M
from d3sno import appearance as P
from d3sno import cli
from d3sno import sno_layout as L
from test_stage12 import laid_out, ps3
from test_stage16_cli import run
from test_stage17_import import TRIANGLE, add_mesh, drop_mesh, primitive
from test_stage17_writer import canon, djb, ps3_file, switch_skin, switch_two, triangles


def donor_switch():
    data = canon(laid_out(B.build_app_switch(pool_spare=0, sub_objects=[
        {"rigid": [(0, 0, 0), (30, 0, 0), (0, 30, 2)], "index_count": 3},
        {"rigid": [(1, 1, 0), (5, 1, 0), (1, 5, 0)], "index_count": 3}])))
    tree = L.parse_tree(data, "app")
    struct.pack_into("<I", tree.top.pairs[0x64].block.elements[1].raw, 0x68, 0x4242)
    return L.write_tree(tree)


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return path


class MeshTransplant(Files):
    def test_round_trip(self):
        target_data, _ = switch_two()
        donor_data = donor_switch()
        target, source, out = self.write("t.app", target_data), self.write("s.app", donor_data), self.dir / "o.app"
        code, text, err = run("mesh-transplant", target, source, "--sub-object", "0:1", "-o", out)
        self.assertEqual(code, 0, err)
        data = out.read_bytes()
        # the command is W17-11: transplant + write_app_mesh(derive=True)
        app, donor = P.parse_appearance(target_data), P.parse_appearance(donor_data)
        self.assertEqual(data, M.write_app_mesh(app, M.transplant(app, donor, 0, 1), derive=True))
        again = P.parse_appearance(data, strict=True)
        self.assertEqual(M.write_app_mesh(again, derive=True), data)
        self.assertEqual(len(again.sub_objects), 4)
        self.assertEqual(triangles(again.sub_objects[3]), triangles(donor.sub_objects[1]))
        self.assertEqual(again.sub_objects[3].sub_hash, 0x4242)
        self.assertIn("added high-poly set, sub-object 3", text)
        self.assertIn(f"wrote {out}: {len(data)} bytes", text)

    def test_name(self):
        target, source, out = self.write("t.app", switch_two()[0]), self.write("s.app", donor_switch()), self.dir / "o.app"
        code, _, err = run("mesh-transplant", target, source, "--sub-object", "0:0", "-o", out, "--name", "Prop")
        self.assertEqual(code, 0, err)
        self.assertEqual(P.parse_appearance(out.read_bytes()).sub_objects[3].sub_hash, djb("Prop"))
        data, build = ps3_file()
        donor = canon(ps3(vertices=((0, 0, 0), (20, 0, 0), (0, 20, 0))), "ps3")
        target, source = self.write("t3.app", data), self.write("s3.app", donor)
        code, _, err = run("mesh-transplant", target, source, "--sub-object", "0:0", "-o", out, "--build", "ps3")
        self.assertEqual(code, 0, err)
        self.assertEqual([s.name for s in P.parse_appearance(out.read_bytes(), "ps3").sub_objects], ["Body", "Body_2"])
        code, _, err = run("mesh-transplant", target, source, "--sub-object", "0:0", "-o", out, "--build", "ps3",
                           "--name", "Arm")
        self.assertEqual(code, 0, err)
        self.assertEqual([s.name for s in P.parse_appearance(out.read_bytes(), "ps3").sub_objects], ["Body", "Arm"])
        # a duplicate given name is refused by the writer (W17-10)
        code, _, err = run("mesh-transplant", target, source, "--sub-object", "0:0", "-o", self.dir / "x.app",
                           "--build", "ps3", "--name", "body")
        self.assertEqual(code, 2)
        self.assertIn("duplicate name", err)
        self.assertFalse((self.dir / "x.app").exists())

    def test_output_may_not_be_an_input(self):
        target_data, donor_data = switch_two()[0], donor_switch()
        target, source = self.write("t.app", target_data), self.write("s.app", donor_data)
        for out in (target, source, self.dir / "." / "t.app"):
            code, text, err = run("mesh-transplant", target, source, "--sub-object", "0:0", "-o", out)
            self.assertEqual(code, 2)
            self.assertIn("never overwrite", err)
            self.assertEqual(text, "")
        self.assertEqual((target.read_bytes(), source.read_bytes()), (target_data, donor_data))

    def test_bad_sub_object(self):
        target, source, out = self.write("t.app", switch_two()[0]), self.write("s.app", donor_switch()), self.dir / "o.app"
        for place, message in (("x", "give G:I"), ("2:0", "give G:I"), ("0:", "give G:I"), ("0:-1", "give G:I"),
                               ("0:9", "no sub-object 9 in geo set 0"), ("1:0", "no sub-object 0 in geo set 1")):
            code, _, err = run("mesh-transplant", target, source, "--sub-object", place, "-o", out)
            self.assertEqual(code, 2, place)
            self.assertIn(message, err, place)
            self.assertFalse(out.exists())
        self.assertEqual(cli.sub_object_place(" 1:12 "), (1, 12))

    def test_refusals_of_transplant(self):
        out = self.dir / "o.app"
        rigid, skinned = self.write("r.app", switch_two()[0]), self.write("k.app", switch_skin()[0])
        code, _, err = run("mesh-transplant", rigid, skinned, "--sub-object", "0:0", "-o", out)
        self.assertEqual(code, 2)
        self.assertIn("different skeletons", err)
        x360 = self.write("x.app", switch_two(">")[0])
        code, _, err = run("mesh-transplant", x360, rigid, "--sub-object", "0:0", "-o", out, "--build", "x360")
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())

    def test_help(self):
        parser = cli.make_parser()
        choices = parser._subparsers._group_actions[0].choices
        text = choices["mesh-transplant"].format_help()
        self.assertIn("--sub-object G:I", text)
        self.assertIn("--name", text)
        self.assertIn("W17-11", text)
        self.assertIn("mesh-transplant", cli._WRITE_COMMANDS)
        self.assertIn("{keep,empty,remove}", choices["mesh-import"].format_help())
        self.assertIn("--missing remove", cli.MESH_IMPORT_NOTE)


class MeshImport(Files):
    def gltf(self, doc, name="m.gltf"):
        return self.write(name, json.dumps(doc))

    def test_missing_remove(self):
        data, _ = switch_two()
        app = P.parse_appearance(data)
        doc = P.to_gltf(app)
        drop_mesh(doc, 1)
        template, model, out = self.write("t.app", data), self.gltf(doc), self.dir / "o.app"
        code, text, err = run("mesh-import", template, model, "-o", out)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.read_bytes(), data)  # keep (the default)
        self.assertNotIn("removed", text)
        code, text, err = run("mesh-import", template, model, "-o", out, "--missing", "remove")
        self.assertEqual(code, 0, err)
        again = P.parse_appearance(out.read_bytes(), strict=True)
        self.assertEqual([triangles(s) for s in again.sub_objects],
                         [triangles(app.sub_objects[0]), triangles(app.sub_objects[2])])
        self.assertIn("removed high-poly set, sub-object 1\n", text)
        self.assertIn(": 0 changed, 2 kept, 0 added, 1 removed", text)
        code, text, err = run("mesh-import", template, model, "-o", out, "--missing", "empty")
        self.assertEqual(code, 0, err)
        self.assertIn("changed high-poly set, sub-object 1, draw 0", text)
        self.assertIn("removed high-poly set, sub-object 1, rigid list", text)

    def test_added_parts_printed(self):
        data, _ = switch_two()
        app = P.parse_appearance(data)
        doc = P.to_gltf(app)
        add_mesh(doc, "Extra", [primitive(doc, TRIANGLE, [0, 1, 2])])
        template, model, out = self.write("t.app", data), self.gltf(doc), self.dir / "o.app"
        code, text, err = run("mesh-import", template, model, "-o", out)
        self.assertEqual(code, 0, err)
        self.assertIn("added high-poly set, sub-object 3\n", text)
        self.assertIn(": 0 changed, 3 kept, 1 added, 0 removed", text)
        self.assertEqual(P.parse_appearance(out.read_bytes()).sub_objects[3].sub_hash, djb("Extra"))
        obj = self.write("m.obj", P.to_obj(app) + "o Crate\nv 20 0 0\nv 26 0 0\nv 20 6 0\nvt 0 0\nvn 0 0 1\n"
                                                     "f 10/4/4 11/4/4 12/4/4\n")
        code, text, err = run("mesh-import", template, obj, "-o", out)
        self.assertEqual(code, 0, err)
        self.assertIn("added high-poly set, sub-object 3", text)
        self.assertIn("warning: OBJ object 'Crate': no colours given", text)

    def test_refusal_exit_2(self):
        data, _ = switch_skin()
        app = P.parse_appearance(data)
        template, out = self.write("t.app", data), self.dir / "o.app"
        obj = self.write("m.obj", P.to_obj(app) + "o Arm\nv 0 0 1\nv 9 0 1\nv 0 1 1\nvt 0 0\nvn 0 0 1\n"
                                                     "f 4/2/2 5/2/2 6/2/2\n")
        code, _, err = run("mesh-import", template, obj, "-o", out)
        self.assertEqual(code, 2)
        self.assertIn("skin='nearest'", err)
        self.assertFalse(out.exists())
        code, text, err = run("mesh-import", template, obj, "-o", out, "--skin", "nearest")
        self.assertEqual(code, 0, err)
        self.assertIn("added high-poly set, sub-object 1", text)


if __name__ == "__main__":
    unittest.main()
