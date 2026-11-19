"""Stage 15.1, animation (31_corrections18.md V18-3): shared per-permutation blocks in a switch
``.ani``. Synthetic data only."""

import copy
import math
import unittest

from d3sno import anim_write as W
from d3sno import animation as A
from d3sno import sno_layout as L

FIELD_PATHS = {
    "track_hashes": "ptBoneNames", "blend_weights": "ptBoneBlendWeights",
    "translation": "ptTranslationCurves", "rotation": "ptRotationCurves",
    "scale": "ptScaleCurves", "attachments": "ptKeyedAttachments",
    "nonlinear_offsets": "pwvNonlinearOffset",
}


def unit(*q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def curves(shift=0.0):
    return A.DecodedPermutation(
        [A.Curve([0, 4], [(1.0 + shift, 2.0, 3.0), (1.5, 2.0, 2.5)]), A.Curve([], [])],
        [A.Curve([0, 4], [(0.0, 0.0, 0.0, 1.0), unit(0.0, shift, 0.5, 1.0)]),
         A.Curve([0], [(0.0, 0.0, 0.0, 1.0)])],
        [A.Curve([0], [(1.0 + shift,)]), A.Curve([], [])],
    )


def perm(name, hashes=(0x11, 0x22), shift=0.0, blend=((0.5, 0.5), (1.0, 0.0)),
         attachments=(bytes(range(192)),), nonlinear_y=0.0):
    p = W.new_permutation("switch", name, 5, 0.5, list(hashes), blend_weights=list(blend),
                          attachments=list(attachments),
                          nonlinear_offsets=[(0.0, 0.0, 0.0)] + [(0.0, nonlinear_y, 0.0)] * 4)
    W.encode_animation([(p, curves(shift))], "switch")
    return p


def pair_offsets(data):
    """{(permutation, field): (offset, size)} of every non-empty per-permutation pair."""
    ani = A.parse_ani(data)
    found = {}
    tree = L.parse_tree(data, "ani")
    for i in range(len(ani.permutations)):
        for name, path in FIELD_PATHS.items():
            _, _, value = L.find_pair(tree, f"ptPermutations[{i}].{path}")
            if value is not None and value.block is not None:
                found[i, name] = value.stored
    return found


def content(p):
    return (p.name, p.track_hashes, p.blend_weights, p.translation_blobs, p.rotation_blobs,
            p.scale_blobs, p.attachments, p.nonlinear_offsets)


class SharedBlocks(unittest.TestCase):
    """V18-3: equal blocks of one field are stored once (new files)."""

    def test_equal_blocks_written_once_and_read_back(self):
        perms = [perm("A"), perm("B", hashes=(0x33, 0x44), shift=0.25, blend=((0.0, 1.0),) * 2,
                                  attachments=(bytes(192),), nonlinear_y=1.0), perm("C")]
        data = W.write_ani(perms)
        offsets = pair_offsets(data)
        for name in FIELD_PATHS:
            self.assertEqual(offsets[2, name], offsets[0, name], name)  # C points at A
            self.assertNotEqual(offsets[1, name], offsets[0, name], name)
        shared = {path for path, _ in L.shared_pairs(L.parse_tree(data, "ani"))}
        for path in FIELD_PATHS.values():
            self.assertIn(f"ptPermutations[2].{path}", shared)
        self.assertEqual(data.count(bytes(range(192))), 1)
        ani = A.parse_ani(data, strict=True)  # the parser reads shared blocks
        self.assertEqual([content(p) for p in ani.permutations], [content(p) for p in perms])
        self.assertEqual(L.rebuild_sno(data, "ani"), data)
        self.assertEqual(W.write_ani(ani.permutations, head=data), data)

    def test_sharing_is_by_field_only(self):
        # 12 zero bytes in the hash list and in the nonlinear offsets (K = 1); 48 zero bytes in
        # each of the three curve tables: equal bytes, different fields
        p = W.new_permutation("switch", "Z", 1, 0.5, [0, 0, 0])
        W.encode_animation([(p, A.DecodedPermutation([A.Curve([], [])] * 3, [A.Curve([], [])] * 3,
                                                     [A.Curve([], [])] * 3))], "switch")
        q = copy.deepcopy(p)
        q.name = "Y"
        data = W.write_ani([p, q])
        offsets = pair_offsets(data)
        firsts = {offsets[0, name] for name in FIELD_PATHS if (0, name) in offsets}
        self.assertEqual(len(firsts), 5)  # hashes, three tables, nonlinear: all apart
        for name in ("track_hashes", "translation", "rotation", "scale", "nonlinear_offsets"):
            self.assertEqual(offsets[1, name], offsets[0, name])

    def test_option_off(self):
        a, b = perm("A"), perm("B")
        data = W.write_ani([a, b], share_blocks=False)
        offsets = pair_offsets(data)
        for name in FIELD_PATHS:
            self.assertNotEqual(offsets[1, name], offsets[0, name], name)
        self.assertEqual(data.count(bytes(range(192))), 2)
        # curve tables are not shared without shared blobs (their bytes differ)
        offsets = pair_offsets(W.write_ani([a, b], share_blobs=False))
        self.assertNotEqual(offsets[1, "rotation"], offsets[0, "rotation"])
        self.assertEqual(offsets[1, "track_hashes"], offsets[0, "track_hashes"])


class StoredLayout(unittest.TestCase):
    """V18-3: an unchanged file keeps its stored layout."""

    def stored_twice(self):
        """A file whose two permutations store equal blend weights twice (all else differs)."""
        a = perm("A")
        b = perm("B", hashes=(0x33, 0x44), shift=0.25, attachments=(bytes(192),), nonlinear_y=1.0)
        data = W.write_ani([a, b], share_blocks=False)
        offsets = pair_offsets(data)
        self.assertNotEqual(offsets[0, "blend_weights"], offsets[1, "blend_weights"])
        return data

    def test_blend_weights_stored_twice_reproduced(self):
        data = self.stored_twice()
        ani = A.parse_ani(data)
        self.assertEqual(set(ani.permutations[1].stored_blocks), set(FIELD_PATHS))
        self.assertEqual(W.write_ani(ani.permutations, head=data), data)
        self.assertEqual(W.write_ani(ani.permutations, head=ani), data)
        # the same content as a new file shares the blend weights
        fresh = copy.deepcopy(ani.permutations)
        for p in fresh:
            p.stored_blocks = {}
        new = W.write_ani(fresh, head=data)
        self.assertNotEqual(new, data)
        offsets = pair_offsets(new)
        self.assertEqual(offsets[0, "blend_weights"], offsets[1, "blend_weights"])

    def test_stored_twice_with_another_change(self):
        data = self.stored_twice()
        ani = A.parse_ani(data)
        ani.permutations[1].name = "Renamed"  # a record field, not a block
        offsets = pair_offsets(W.write_ani(ani.permutations, head=data))
        self.assertNotEqual(offsets[0, "blend_weights"], offsets[1, "blend_weights"])
        # an appended new permutation with equal blend weights points at the first copy
        extra = perm("C", hashes=(0x55, 0x66), shift=0.5, attachments=(), nonlinear_y=2.0)
        offsets = pair_offsets(W.write_ani(ani.permutations + [extra], head=data))
        self.assertEqual(offsets[2, "blend_weights"], offsets[0, "blend_weights"])
        # a copy of a parsed permutation shares with its original
        twin = copy.deepcopy(ani.permutations[1])
        offsets = pair_offsets(W.write_ani(ani.permutations + [twin], head=data))
        self.assertEqual(offsets[2, "blend_weights"], offsets[1, "blend_weights"])
        self.assertEqual(offsets[2, "track_hashes"], offsets[1, "track_hashes"])

    def test_only_one_permutation_changes(self):
        perms = [perm("A"), perm("B", hashes=(0x33, 0x44), shift=0.25), perm("C")]
        data = W.write_ani(perms)
        ani = A.parse_ani(data)
        self.assertEqual(W.write_ani(ani.permutations, head=data), data)
        changed = ani.permutations
        changed[2].track_hashes = [0x77, 0x88]
        changed[2].nonlinear_offsets = [(0.0, 0.0, 0.0)] + [(0.0, 0.0, 3.0)] * 4
        out = W.write_ani(changed, head=data)
        offsets = pair_offsets(out)
        self.assertNotIn(offsets[2, "track_hashes"], {offsets[0, "track_hashes"],
                                                       offsets[1, "track_hashes"]})
        self.assertNotEqual(offsets[2, "nonlinear_offsets"], offsets[0, "nonlinear_offsets"])
        for name in ("blend_weights", "translation", "rotation", "scale", "attachments"):
            self.assertEqual(offsets[2, name], offsets[0, name], name)  # still shared
        for name in ("blend_weights", "attachments", "nonlinear_offsets"):
            self.assertEqual(offsets[1, name], offsets[0, name], name)  # B shares these with A
        again = A.parse_ani(out)
        self.assertEqual(again.permutations[2].track_hashes, [0x77, 0x88])
        self.assertEqual([content(p) for p in again.permutations[:2]],
                         [content(p) for p in perms[:2]])
        # the result is what a new file of the same content gives
        fresh = copy.deepcopy(changed)
        for p in fresh:
            p.stored_blocks = {}
        self.assertEqual(W.write_ani(fresh, head=data), out)

    def test_new_block_before_stored_ones(self):
        # a new permutation placed first: the unchanged equal blocks after it point at its block
        data = self.stored_twice()
        ani = A.parse_ani(data)
        first = perm("N", hashes=(0x55, 0x66), shift=0.5, attachments=(), nonlinear_y=2.0)
        offsets = pair_offsets(W.write_ani([first] + ani.permutations, head=data))
        self.assertEqual(offsets[1, "blend_weights"], offsets[0, "blend_weights"])
        self.assertEqual(offsets[2, "blend_weights"], offsets[0, "blend_weights"])

    def test_change_that_makes_blocks_equal(self):
        data = self.stored_twice()
        ani = A.parse_ani(data)
        ani.permutations[1].track_hashes = list(ani.permutations[0].track_hashes)
        offsets = pair_offsets(W.write_ani(ani.permutations, head=data))
        self.assertEqual(offsets[1, "track_hashes"], offsets[0, "track_hashes"])
        self.assertNotEqual(offsets[1, "blend_weights"], offsets[0, "blend_weights"])

    def test_stored_blocks_from_parser(self):
        data = W.write_ani([perm("A"), perm("B")])
        ani = A.parse_ani(data)
        first, second = ani.permutations
        self.assertEqual(first.stored_blocks, second.stored_blocks)
        self.assertEqual(first.stored_blocks["blend_weights"][1],
                         A.block_keys(first)["blend_weights"])
        self.assertEqual(A.parse_ani(W.write_ani([perm("A")])).permutations[0].stored_blocks.keys(),
                         first.stored_blocks.keys())
        self.assertEqual(W.new_permutation("switch", "N", 1, 0.5, [1]).stored_blocks, {})


if __name__ == "__main__":
    unittest.main()
