"""Stage 12.1: range checks in the mesh-level ``.app`` writer.

Every integer word is checked against its width before it is packed, and a changed ps3 draw has
its work buffer (0xC000) checked before its 16-bit index-pair pieces, so a mesh that does not fit
the format raises FormatError naming the field, never ``struct.error``. Synthetic data only.
"""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import d3sno
from d3sno import FormatError
from d3sno import app_mesh as M
from d3sno import appearance as P
from test_stage12 import BONES, ps3, switch_rigid, switch_skinned


def grow(draw, vertices: int = 0, triangles: int = 0) -> None:
    """Append ``vertices`` copies of vertex 0 (each a new vertex with its own triangle) and
    ``triangles`` more triangles over vertices 0, 1, 2."""
    for _ in range(vertices):
        n = len(draw.positions)
        for name in ("positions", "normals", "colours_1", "colours_2", "uv1", "uv2_raw"):
            getattr(draw, name).append(getattr(draw, name)[0])
        if draw.skinned:
            draw.joints.append(draw.joints[0])
            draw.weights.append(draw.weights[0])
        draw.indices += [0, 1, n]
    draw.indices += [0, 1, 2] * triangles


def ps3_rigid():
    return P.parse_appearance(ps3(), "ps3")


def ps3_skinned():
    return P.parse_appearance(ps3(skinned=True, bones=BONES), "ps3")


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 12, 1))
        self.assertGreaterEqual(d3sno.STAGE, 12)


class Words(unittest.TestCase):
    def test_check_word(self):
        self.assertEqual(M.check_word(0xFFFF, "H", "x"), 0xFFFF)
        for value, code in ((0x10000, "H"), (-1, "H"), (256, "B"), (0x8000, "h"), (-0x8001, "h"),
                            (1 << 32, "I"), (1 << 31, "i"), (1.5, "H"), (True, "B")):
            with self.assertRaisesRegex(FormatError, f"field: .* {code} word"):
                M.check_word(value, code, "field")

    def test_check_float(self):
        for bad in (float("nan"), float("inf"), 1e39):
            with self.assertRaises(FormatError):
                M.check_float(bad, "x")
        self.assertEqual(M.check_float(1.5, "x"), 1.5)


class OversizedSkinnedPs3Draw(unittest.TestCase):
    def test_work_buffer_before_the_words(self):
        app = ps3_skinned()
        grow(app.sub_objects[0].draws[0], vertices=3000)  # streams A, B and the work all too big
        # ``38`` V23-3: a changed draw over any limit is refused, the budget first
        with self.assertRaisesRegex(FormatError, r"draw 0: budget value .* \(W17-5a\); a changed draw is not cut"):
            M.write_app_mesh(app)

    def test_index_pieces_after_the_work_check(self):
        # 45,000 indices: the index pair (90,000 bytes) overflows its second 16-bit piece, but the
        # skinned work buffer is checked first and is the reported problem
        app = ps3_skinned()
        grow(app.sub_objects[0].draws[0], triangles=15000)
        with self.assertRaisesRegex(FormatError, "0xC000"):
            M.write_app_mesh(app)

    def test_descriptor_count(self):
        app = ps3_skinned()
        grow(app.sub_objects[0].draws[0], vertices=0x10000)
        with self.assertRaisesRegex(FormatError, r"vertex count \(descriptor bytes 8..9\).*H word"):
            M.write_app_mesh(app)

    def test_byte_fields(self):
        app = ps3_skinned()
        app.sub_objects[0].draws[0].weights[0] = (300, 0, 0, 0)
        with self.assertRaisesRegex(FormatError, r"vertex 0 weights.*B word"):
            M.write_app_mesh(app)


class OversizedRigidPs3Draw(unittest.TestCase):
    def test_stream_a(self):
        app = ps3_rigid()
        grow(app.sub_objects[0].draws[0], vertices=3000)  # 3003 x 6 bytes
        with self.assertRaisesRegex(FormatError, r"draw 0: budget value 72066 is above 55998 .*V23-3"):  # ``38`` V23-3
            M.write_app_mesh(app)

    def test_index_pieces(self):
        app = ps3_rigid()
        grow(app.sub_objects[0].draws[0], triangles=15000)  # 45,003 indices, 90,016 bytes
        # ``38`` V23-3: the work buffer of a changed 0x81 draw is refused before the index-pair words
        with self.assertRaisesRegex(FormatError, r"draw 0: work buffer 91264 is above 0xC000 .*V23-3"):
            M.write_app_mesh(app)

    def test_descriptor_index_count(self):
        app = ps3_rigid()
        grow(app.sub_objects[0].draws[0], triangles=22000)  # 66,003 indices
        with self.assertRaisesRegex(FormatError, r"index count \(descriptor bytes 10..11\)"):
            M.write_app_mesh(app)

    def test_uv2(self):
        app = ps3_rigid()
        app.sub_objects[0].draws[0].uv2_raw[1] = (0x10000, 0)
        with self.assertRaisesRegex(FormatError, r"vertex 1 UV2.*H word"):
            M.write_app_mesh(app)

    def test_nothing_written_after_a_refusal(self):
        app = ps3_rigid()
        source = bytes(app.source)
        grow(app.sub_objects[0].draws[0], vertices=3000)
        with self.assertRaises(FormatError):
            M.write_app_mesh(app)
        self.assertEqual(bytes(app.source), source)


class SwitchWords(unittest.TestCase):
    def test_packed_vertex_fields(self):
        app = P.parse_appearance(switch_rigid())
        app.sub_objects[0].draws[0].uv2_raw[0] = (-1, 0)
        with self.assertRaisesRegex(FormatError, r"vertex 0 UV2"):
            M.write_app_mesh(app)
        app = P.parse_appearance(switch_rigid())
        app.sub_objects[0].draws[0].normals[0] = (float("nan"), 0.0, 0.0)
        with self.assertRaisesRegex(FormatError, "normal"):
            M.write_app_mesh(app)
        app = P.parse_appearance(switch_rigid())
        app.sub_objects[0].draws[0].indices[0] = 0x10000
        with self.assertRaisesRegex(FormatError, "index"):
            M.write_app_mesh(app)

    def test_skinned_weights(self):
        app = P.parse_appearance(switch_skinned())
        app.sub_objects[0].draws[0].weights[0] = (256, 0, 0, 0)
        with self.assertRaisesRegex(FormatError, r"weights.*B word"):
            M.write_app_mesh(app)

    def test_pack_names_the_field(self):
        raw = bytearray(8)
        with self.assertRaisesRegex(FormatError, r"nIndexCount: .* i word"):
            M._pack(raw, "<", "i", 0, 1 << 31, field="nIndexCount")
        with self.assertRaisesRegex(FormatError, r"word \+0x4"):
            M._pack(raw, "<", "H", 4, -2)
        self.assertEqual(raw, bytearray(8))
        M._pack(raw, ">", "HH", 0, 1, 2)
        self.assertEqual(struct.unpack_from(">HH", raw), (1, 2))


if __name__ == "__main__":
    unittest.main()
