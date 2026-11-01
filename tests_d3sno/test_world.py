import math
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import world as W

BUILDS = ("ps3", "switch")
SQRT = math.sqrt(0.5)
Z_90 = (0.0, 0.0, SQRT, SQRT)


class Scene(unittest.TestCase):
    def test_fields(self):
        for build in BUILDS:
            read = W.parse_scene_info(B.build_scene_info(
                build, marker_sets=[5, 6], exclusions=[7], inclusions=[8, 9],
            ))
            self.assertEqual(read.bounds, ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
            self.assertEqual(read.marker_bounds[0], (7.0, 8.0, 9.0))
            self.assertEqual(read.marker_sets, [5, 6])
            self.assertEqual((read.exclusions, read.inclusions), ([7], [8, 9]))
            self.assertEqual(read.look_hash, 0xABCD)
            self.assertEqual((read.appearance, read.collision_mesh), (11, 12))

    def test_xp_density(self):
        self.assertEqual(W.parse_scene_info(B.build_scene_info("switch")).xp_density, 1.5)
        self.assertIsNone(W.parse_scene_info(B.build_scene_info("ps3")).xp_density)

    def test_too_short(self):
        with self.assertRaises(FormatError):
            W.parse_scene_info(B.build_scene_info("switch")[:0x200])

    def test_list_size(self):
        data = bytearray(B.build_scene_info("switch", marker_sets=[1]))
        struct.pack_into("<I", data, 0x134, 3)
        with self.assertRaises(FormatError):
            W.parse_scene_info(bytes(data))


class Markers(unittest.TestCase):
    def test_fields(self):
        for build in BUILDS:
            marker = {
                "hash": 0x99, "type": 7, "rotation": (0.0, 0.0, 1.0, 0.0),
                "position": (1.0, 2.0, 3.0), "group": 27, "handle": 44, "links": [(1, 2), (3, 4)],
            }
            read = W.parse_marker_set(B.build_mrk(
                build, [marker], [(1.0, 2.0, 3.0)], label=5, special=[1, 2],
            ))
            m = read.markers[0]
            self.assertEqual((m.name_hash, m.type, m.type_name), (0x99, 7, "particle"))
            self.assertEqual(m.rotation, (0.0, 0.0, 1.0, 0.0))
            self.assertEqual(m.position, (1.0, 2.0, 3.0))
            self.assertEqual((m.group, m.handle), (27, 44))
            self.assertEqual(m.links, [(1, 2), (3, 4)])
            self.assertEqual(read.circles, [(1.0, 2.0, 3.0)])
            self.assertEqual(read.bounds, ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)))
            self.assertEqual((read.flags, read.label), (3, 5))
            self.assertEqual(read.special_indices, [1, 2])
            self.assertFalse(read.always_active)

    def test_always_active(self):
        self.assertTrue(W.parse_marker_set(B.build_mrk("switch")).always_active)

    def test_type_names(self):
        self.assertEqual(W.MARKER_TYPES[50], "gizmo_location_A")
        self.assertEqual(W.MARKER_TYPES[75], "gizmo_location_Z")
        self.assertEqual(W.MARKER_TYPES[31], "effect_group")
        self.assertEqual(W.MARKER_GROUPS[10], 15)
        self.assertEqual(W.MARKER_TYPES[13], "script")
        self.assertEqual(W.MARKER_TYPES[101], "location_101")
        self.assertEqual(W.MARKER_GROUPS[13], 35)
        self.assertEqual((W.MARKER_GROUPS[5], W.MARKER_GROUPS[14]), (47, 2))
        self.assertEqual((W.MARKER_GROUPS[29], W.MARKER_GROUPS[31]), (31, 14))

    def test_scale(self):
        for build in BUILDS:
            order = ">" if build == "ps3" else "<"
            tags = B.tag_map([(1, 0x80000, 2.5)], order)
            read = W.parse_marker_set(B.build_mrk(build, [{"tags": tags}]))
            self.assertEqual(read.markers[0].tag_map_raw, tags)
            self.assertEqual(read.markers[0].scale, 2.5)

    def test_scale_defaults_to_1(self):
        read = W.parse_marker_set(B.build_mrk("switch", [{}, {"tags": b"\xFF\xFF"}]))
        self.assertEqual(read.markers[0].scale, 1.0)
        self.assertEqual(read.markers[1].scale, 1.0)
        self.assertIsNotNone(read.markers[1].tags.error)

    def test_type_7_in_a_marker_map(self):
        tags = B.tag_map([(7, 0x80A03, 5), (1, 0x80000, 2.5), (7, 0x80A04, 6)], "<")
        marker = W.parse_marker_set(B.build_mrk("switch", [{"tags": tags}])).markers[0]
        self.assertEqual(marker.tags.get(0x80A04).value, 6)
        self.assertEqual(marker.scale, 2.5)
        self.assertIsNone(marker.tags.error)

    def test_a_bad_record_keeps_the_scale_before_it(self):
        tags = B.tag_map([(1, 0x80000, 2.5)], "<") + struct.pack("<II", 99, 1)
        marker = W.parse_marker_set(B.build_mrk("switch", [{"tags": tags}])).markers[0]
        self.assertEqual(marker.scale, 2.5)
        self.assertIsNotNone(marker.tags.error)

    def actor(self, scale=None, kind=0):
        order = "<"
        tags = B.tag_map([(1, 0x80000, scale)], order) if scale is not None else b""
        return W.parse_marker_set(B.build_mrk("switch", [{"tags": tags, "type": kind}])).markers[0]

    def test_actor_scale_keeps_the_marker_value(self):
        self.assertEqual(self.actor(2.0).actor_scale(0, 5.0, 1.0), 2.0)

    def test_actor_scale_replaced_when_tiny(self):
        self.assertEqual(self.actor(0.0).actor_scale(0, 5.0, 2.0, lambda: 0.5), 6.0)

    def test_actor_scale_replaced_by_flag_bit_0(self):
        self.assertEqual(self.actor(2.0).actor_scale(1, 5.0, 2.0, lambda: 0.5), 6.0)

    def test_actor_scale_zero_delta_needs_no_draw(self):
        self.assertEqual(self.actor(0.0).actor_scale(0, 5.0, 0.0), 5.0)

    def test_actor_scale_only_on_actor_markers(self):
        self.assertEqual(self.actor(0.0, kind=7).actor_scale(1, 5.0, 2.0), 0.0)

    def test_actor_scale_with_delta_needs_a_source(self):
        with self.assertRaises(FormatError):
            self.actor(0.0).actor_scale(0, 5.0, 2.0)

    def test_marker_array_size(self):
        data = bytearray(B.build_mrk("switch", [{}]))
        struct.pack_into("<I", data, 0x20, 87)
        with self.assertRaises(FormatError):
            W.parse_marker_set(bytes(data))

    def test_link_count(self):
        with self.assertRaises(FormatError):
            W.parse_marker_set(B.build_mrk("switch", [{"links": [(1, 2)], "link_count": 2}]))

    def test_circles_size(self):
        data = bytearray(B.build_mrk("switch", circles=[(1.0, 2.0, 3.0)]))
        struct.pack_into("<I", data, 0x64, 8)
        with self.assertRaises(FormatError):
            W.parse_marker_set(bytes(data))

    def test_special_count(self):
        with self.assertRaises(FormatError):
            W.parse_marker_set(B.build_mrk("switch", special=[1], special_count=2))

    def test_fixed_part(self):
        with self.assertRaises(FormatError):
            W.parse_marker_set(B.build_mrk("switch")[:0xC0])


class Worlds(unittest.TestCase):
    def test_fields(self):
        for build in BUILDS:
            entry = B.scene_entry(build, 5, (1.0, 2.0, 3.0), Z_90, cluster=8)
            read = W.parse_world(B.build_wrl(build, [[entry]], marker_sets=[9], generated=0))
            self.assertEqual(read.marker_sets, [9])
            self.assertEqual(read.skybox_word, 100)
            self.assertEqual(read.skybox_actor, None if build == "ps3" else 100)
            self.assertEqual((read.music, read.ambient), (101, 102))
            self.assertEqual((read.reverb, read.weather), (103, 104))
            server = read.server
            self.assertEqual((server.next_cluster_id, server.cluster_count), (8, 1))
            self.assertEqual(server.clusters_raw, b"clus")
            self.assertEqual(server.navigation_powers, (21, 22, 23, 24))
            self.assertEqual((server.script_handle, server.seed), (31, 32))
            scene = read.scene_entries[0]
            self.assertEqual((scene.group, scene.handle, scene.cluster_id), (33, 5, 8))
            self.assertEqual(scene.position, (1.0, 2.0, 3.0))
            self.assertAlmostEqual(scene.rotation[2], SQRT, places=6)
            self.assertEqual(scene.level_areas, (1, 2, 3, 4))

    def test_several_chunks(self):
        chunks = [[B.scene_entry("ps3", 1)], [B.scene_entry("ps3", 2), B.scene_entry("ps3", 3)]]
        read = W.parse_world(B.build_wrl("ps3", chunks))
        self.assertEqual([e.handle for e in read.scene_entries], [1, 2, 3])

    def test_block_size(self):
        with self.assertRaises(FormatError):
            W.parse_world(B.build_wrl("switch", block_size=112))

    def test_entry_count_must_match(self):
        with self.assertRaises(FormatError):
            W.parse_world(B.build_wrl("switch", [[B.scene_entry("switch", 1)]], chunk_count=2))

    def test_fixed_part(self):
        with self.assertRaises(FormatError):
            W.parse_world(B.build_wrl("ps3")[:0x80])

    def test_generated(self):
        for build in BUILDS:
            read = W.parse_world(B.build_wrl(build, tiles=[(5, 2, 77, 9)], generated=1))
            self.assertEqual(read.is_generated, 1)
            generator = read.server.generators[0]
            self.assertEqual(generator.tiles[0], W.Tile(5, 2, 77, 9))
            self.assertEqual((generator.command_count, generator.commands_raw), (1, b"cmd!"))
            self.assertTrue(generator.tiles[0].exits & W.EXIT_NORTH)
            self.assertTrue(generator.tiles[0].exits & W.EXIT_EAST)
            self.assertEqual(read.scene_entries, [])

    def test_neither_scene_lists_nor_generators(self):
        read = W.parse_world(B.build_wrl("switch"))
        self.assertEqual((read.server.generators, read.scene_entries), ([], []))
        self.assertEqual(read.warnings, [])

    def test_cluster_none(self):
        entry = B.scene_entry("ps3", 5, cluster=-1)
        self.assertEqual(W.parse_world(B.build_wrl("ps3", [[entry]])).scene_entries[0].cluster_id, -1)

    def test_scene_lists_and_generator_together(self):
        data = B.build_wrl("switch", [[B.scene_entry("switch", 1)]], tiles=[(1, 2, 3, 4)])
        self.assertTrue(W.parse_world(data).warnings)
        with self.assertRaises(FormatError):
            W.parse_world(data, strict=True)


class Layout(unittest.TestCase):
    def setUp(self):
        self.entry = B.scene_entry("switch", 5, (100.0, 200.0, 0.0), Z_90)
        self.scene = W.parse_scene_info(B.build_scene_info("switch", marker_sets=[1, 2]))
        self.fixed = W.parse_marker_set(B.build_mrk("switch", [
            {"position": (1.0, 0.0, 5.0), "rotation": Z_90},
            {"position": (0.0, 0.0, 0.0), "type": 16},
        ]))
        self.conditional = W.parse_marker_set(B.build_mrk("switch", [{}], label=3))

    def layout(self, marker_sets=None, world_sets=()):
        world = W.parse_world(B.build_wrl("switch", [[self.entry]], marker_sets=world_sets))
        sets = {1: self.fixed, 2: self.conditional} if marker_sets is None else marker_sets
        return W.lay_out_world(world, {5: self.scene}, sets)

    def test_marker_position_is_rotated_then_moved(self):
        placed = self.layout().markers[0]
        self.assertAlmostEqual(placed.position[0], 100.0, places=5)
        self.assertAlmostEqual(placed.position[1], 201.0, places=5)
        self.assertAlmostEqual(placed.position[2], 5.0, places=5)

    def test_marker_rotation_is_composed(self):
        rotation = self.layout().markers[0].rotation
        self.assertAlmostEqual(rotation[2], 1.0, places=5)
        self.assertAlmostEqual(rotation[3], 0.0, places=5)

    def test_rotation_order_is_scene_then_marker(self):
        x_90 = (SQRT, 0.0, 0.0, SQRT)
        marker_set = W.parse_marker_set(B.build_mrk("switch", [{"rotation": x_90}]))
        rotation = self.layout({1: marker_set, 2: marker_set}).markers[0].rotation
        for got, want in zip(rotation, (0.5, 0.5, 0.5, 0.5)):
            self.assertAlmostEqual(got, want, places=5)

    def world(self, world_sets=()):
        return W.parse_world(B.build_wrl("switch", [[self.entry]], marker_sets=world_sets))

    def test_active_labels_filter_a_scene_set(self):
        sets = {1: self.fixed, 2: self.conditional}
        scenes = {5: self.scene}
        off = W.lay_out_world(self.world(), scenes, sets, active_labels={})
        self.assertEqual(len(off.markers), 2)
        on = W.lay_out_world(self.world(), scenes, sets, active_labels={5: {3}})
        self.assertEqual(len(on.markers), 3)
        other = W.lay_out_world(self.world(), scenes, sets, active_labels={6: {3}})
        self.assertEqual(len(other.markers), 2)

    def test_world_sets_are_never_filtered(self):
        sets = {1: self.fixed, 2: self.conditional}
        layout = W.lay_out_world(self.world([2]), {5: self.scene}, sets, active_labels={})
        self.assertEqual(len(layout.markers), 3)
        self.assertFalse(layout.markers[-1].conditional)

    def test_world_sets_move_by_minus_the_origin(self):
        world = self.world([1])
        layout = W.lay_out_world(world, {}, {1: self.fixed}, origin=(10.0, 20.0))
        self.assertEqual(layout.markers[0].position, (-9.0, -20.0, 5.0))
        self.assertEqual(layout.markers[0].rotation, self.fixed.markers[0].rotation)

    def test_conditional_sets_are_flagged_not_dropped(self):
        markers = self.layout().markers
        self.assertEqual([m.conditional for m in markers], [False, False, True])

    def test_sub_scene_slots(self):
        layout = self.layout()
        self.assertEqual(len(layout.sub_scene_slots), 1)
        self.assertAlmostEqual(layout.sub_scene_slots[0].position[1], 200.0)

    def test_world_sets_are_in_world_space(self):
        layout = self.layout(world_sets=[1])
        placed = layout.markers[-2]
        self.assertIsNone(placed.scene)
        self.assertEqual(placed.position, (1.0, 0.0, 5.0))

    def test_missing_sets_are_listed(self):
        layout = self.layout(marker_sets={})
        self.assertEqual(layout.missing_marker_sets, [1, 2])

    def test_missing_scene_is_listed(self):
        world = W.parse_world(B.build_wrl("switch", [[self.entry]]))
        self.assertEqual(W.lay_out_world(world, {}, {}).missing_scenes, [5])
