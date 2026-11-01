import random
import struct
import unittest

import builders as B
from d3sno import FormatError
from d3sno import archive as A


def contents(data):
    arc = A.read_archive(data)
    return {
        e.name: [arc.read_file(e, c) for c in range(e.count)] for e in arc.files
    }


FILES = {
    "Power\\Zeta.pow": b"zeta" * 3000,
    "Actor\\Alpha.acr": random.Random(7).randbytes(70000),
    "StringList\\Foo.stl|frFR": b"bonjour" * 500,
    "Empty\\Nothing.bin": b"",
}


def packets_of(data):
    arc = A.read_archive(data)
    return arc, [p for k in range(len(arc.sector_offsets)) for p in arc.sector(k).packets]


class Writer(unittest.TestCase):
    def test_v7_verifies(self):
        data = A.write_archive(FILES, read_sector=0x800)
        report = A.verify_archive(data)
        self.assertTrue(report.ok, report.format())
        self.assertEqual({k: v[0] for k, v in contents(data).items()}, FILES)

    def test_v6_big_endian_with_empty_first_packet(self):
        data = A.write_archive(FILES, version=6)
        self.assertEqual(data[:4], b"\xA1\xB2\xC3\xD4")
        self.assertTrue(A.verify_archive(data).ok)
        arc, packets = packets_of(data)
        self.assertEqual((packets[0].declared, packets[0].stored), (0, 8))
        self.assertEqual(arc.header.header_size, 64)

    def test_v7_starts_with_real_data(self):
        _, packets = packets_of(A.write_archive(FILES))
        self.assertGreater(packets[0].declared, 0)

    def test_compact_flag(self):
        data = A.write_archive(FILES, flags=10)
        arc = A.read_archive(data)
        self.assertEqual(arc.header.start, arc.tables_end + len(b"".join(
            n.encode() + b"\0" for n in arc.names)))
        self.assertTrue(A.verify_archive(data).ok)

    def test_uncompressed(self):
        data = A.write_archive(FILES, flags=0)
        report = A.verify_archive(data)
        self.assertTrue(report.ok, report.format())
        self.assertEqual({k: v[0] for k, v in contents(data).items()}, FILES)

    def test_level_9_header(self):
        arc = A.read_archive(A.write_archive(FILES))
        first = arc.sector(0).packets[0]
        self.assertEqual(arc.data[first.offset + 6:first.offset + 8], b"\x78\xDA")

    def test_packets_run_past_the_slot_end_by_default(self):
        big = {"big.bin": random.Random(3).randbytes(100000)}
        data = A.write_archive(big)
        arc, packets = packets_of(data)
        cut = [p for p in packets if p.cut]
        self.assertTrue(cut)
        self.assertEqual(sum(p.padded for p in packets), 0)
        for packet in cut:
            self.assertEqual(packet.offset + 6 + packet.stored, arc.slot_position(packet.slot + 1))
        self.assertTrue(A.verify_archive(data).ok)
        self.assertEqual(arc.read_file("big.bin"), big["big.bin"])

    def test_running_past_the_slot_end_saves_room(self):
        big = {"big.bin": random.Random(3).randbytes(400000)}
        self.assertLessEqual(len(A.write_archive(big)), len(A.write_archive(big, run_past_slot=False)))

    def test_cut_packet_can_be_encrypted(self):
        big = {"big.bin": random.Random(3).randbytes(50000)}
        data = A.write_archive(big, encrypt=lambda n: True)
        self.assertEqual(A.read_archive(data).read_file("big.bin"), big["big.bin"])
        self.assertTrue(A.verify_archive(data).ok)

    def test_packets_above_0x10000_are_accepted(self):
        big = {"big.bin": bytes(300000)}
        data = A.write_archive(big, max_packet=0x40000)
        arc, packets = packets_of(data)
        self.assertGreater(max(p.declared for p in packets), 0x10000)
        self.assertTrue(A.verify_archive(data).ok)
        self.assertEqual(arc.read_file("big.bin"), big["big.bin"])

    def test_conditions(self):
        data = A.write_archive(
            FILES, read_sector=0x800, comp_sector=0x2000, run_past_slot=False,
        )
        arc, packets = packets_of(data)
        h = arc.header
        self.assertEqual(h.sector_offset_bits, h.location_bits)
        self.assertEqual(h.sector_index_bits, h.comp_sector_count.bit_length())
        self.assertEqual(len(data), h.start + h.comp_sector_count * h.comp_sector)
        self.assertEqual(arc.sector_offsets[0], 0)
        for packet in packets:
            self.assertLess(packet.stored, 0x4000)
            self.assertLessEqual(packet.declared, 0x10000)
            self.assertFalse(packet.cut)
            self.assertEqual(packet.excess, 0)
            self.assertEqual(packet.underrun, 0)
            end = packet.offset + 6 + packet.stored
            self.assertLessEqual(end, arc.slot_position(packet.slot) + h.comp_sector)
        hashes = [e.hash for e in arc.files]
        self.assertEqual(hashes, sorted(hashes))

    def test_stored_sizes_stay_below_bit_14(self):
        _, packets = packets_of(A.write_archive(FILES))
        self.assertTrue(all(p.stored < 0x4000 for p in packets))

    def test_no_gaps_between_packets(self):
        arc, packets = packets_of(A.write_archive(FILES))
        for a, b in zip(packets, packets[1:]):
            if a.slot == b.slot:
                self.assertEqual(a.offset + 6 + a.stored, b.offset)

    def test_refuses_big_sectors(self):
        with self.assertRaises(FormatError):
            A.write_archive(FILES, comp_sector=0x8000)
        with self.assertRaises(FormatError):
            A.write_archive(FILES, read_sector=0x20000)

    def test_refuses_v6_custom_sectors(self):
        with self.assertRaises(FormatError):
            A.write_archive(FILES, version=6, comp_sector=0x2000)

    def test_refuses_runtime_flags(self):
        with self.assertRaises(FormatError):
            A.write_archive(FILES, flags=3)

    def test_refuses_duplicate_case_names(self):
        with self.assertRaises(FormatError):
            A.write_archive({"a.bin": b"1", "A.BIN": b"2"})

    def test_encrypted_packets_read_back(self):
        data = A.write_archive(FILES, encrypt=lambda n: n % 2 == 0)
        report = A.verify_archive(data)
        self.assertTrue(report.ok, report.format())
        self.assertGreater(report.encrypted_packets, 0)
        self.assertEqual({k: v[0] for k, v in contents(data).items()}, FILES)

    def test_empty_archive(self):
        data = A.write_archive({})
        arc = A.read_archive(data)
        self.assertEqual((arc.files, arc.sector_offsets), ([], []))
        self.assertTrue(A.verify_archive(data).ok)


class Rebuild(unittest.TestCase):
    def test_round_trip_of_writer_output_is_byte_identical(self):
        for options in (
            {}, {"version": 6}, {"flags": 10}, {"flags": 0}, {"read_sector": 0x800},
        ):
            with self.subTest(options):
                first = A.write_archive(FILES, **options)
                self.assertEqual(A.rebuild(first).data, first)

    def test_rebuild_option_for_complete_packets(self):
        first = A.write_archive(FILES)
        _, packets = packets_of(A.rebuild(first, run_past_slot=False).data)
        self.assertFalse(any(p.cut for p in packets))

    def test_round_trip_with_copies(self):
        files = [
            A.InputFile("two.bin", [b"a" * 100, b"a" * 100]),
            A.InputFile("one.bin", [b"b" * 50]),
        ]
        first = A.write_archive(files, version=6)
        self.assertEqual(A.rebuild(first).data, first)

    def test_hand_made_source_keeps_file_bytes(self):
        """Packets differ from the source, so file bytes are what round-trips."""
        stream = random.Random(9).randbytes(20000)
        offsets, slots = B.simple_slots(stream, 0x1000, "<", chunk=900)
        source = B.build_cpk(B.CpkSpec(
            files=[("Foo.bin", 8000, 1, 0), ("Bar.bin", 12000, 1, 1)], locations=[0, 8000],
            sector_offsets=offsets, stream_size=len(stream), slots=slots,
        ))
        result = A.rebuild(source)
        self.assertEqual(contents(result.data), contents(source))
        self.assertTrue(A.verify_archive(result.data).ok)
        head_a, head_b = A.read_archive(source).header, A.read_archive(result.data).header
        for name in ("version", "order", "flags", "read_sector", "comp_sector"):
            self.assertEqual(getattr(head_a, name), getattr(head_b, name))

    def test_replace_updates_every_copy(self):
        first = A.write_archive(
            [A.InputFile("two.bin", [b"a" * 100, b"a" * 100])], version=6,
        )
        result = A.rebuild(first, replace={"two.bin": b"new" * 10})
        self.assertEqual(contents(result.data)["two.bin"], [b"new" * 10] * 2)
        self.assertTrue(A.verify_archive(result.data).ok)

    def test_single_copy(self):
        first = A.write_archive(
            [A.InputFile("two.bin", [b"a" * 100, b"a" * 100])], version=6,
        )
        result = A.rebuild(first, single_copy=True)
        self.assertEqual(A.read_archive(result.data).find("two.bin").count, 1)

    def test_add_and_remove(self):
        first = A.write_archive(FILES)
        result = A.rebuild(first, remove=["Empty\\Nothing.bin"], add={"new.bin": b"new"})
        arc = A.read_archive(result.data)
        self.assertIsNone(arc.find("Empty\\Nothing.bin"))
        self.assertEqual(arc.read_file("new.bin"), b"new")
        self.assertTrue(A.verify_archive(result.data).ok)

    def test_replace_unknown_name(self):
        with self.assertRaises(FormatError):
            A.rebuild(A.write_archive(FILES), replace={"nope": b""})

    def test_dropped_encryption_is_reported(self):
        first = A.write_archive(FILES, encrypt=lambda n: True)
        result = A.rebuild(first)
        self.assertGreater(result.encrypted_packets_dropped, 0)
        self.assertTrue(result.notes)
        self.assertEqual(contents(result.data), contents(first))

    def test_unnamed_file_cannot_be_rebuilt(self):
        source = B.build_cpk(B.CpkSpec(
            files=[("a.bin", 5, 1, 0)], locations=[0], sector_offsets=[0], stream_size=5,
            slots=[B.packet(b"12345", "<")], wrong_name_hash=True,
        ))
        with self.assertRaises(FormatError):
            A.rebuild(source)


class Overlay(unittest.TestCase):
    def setUp(self):
        self.low = A.read_archive(A.write_archive({"a.stl|frFR": b"low locale", "b.bin": b"low"}))
        self.high = A.read_archive(A.write_archive({"a.stl": b"high plain", "b.bin": b"high"}))
        self.overlay = A.ArchiveOverlay()
        self.overlay.add(self.low, -4)
        self.overlay.add(self.high, 12)

    def read(self, *args):
        hit = self.overlay.resolve(*args)
        return hit[0].read_file(hit[1])

    def test_locale_in_low_priority_beats_plain_in_high(self):
        self.assertEqual(self.read("a.stl", "frFR"), b"low locale")

    def test_plain_when_no_locale(self):
        self.assertEqual(self.read("a.stl", "deDE"), b"high plain")

    def test_audio_locale_is_tried_second(self):
        self.assertEqual(self.read("a.stl", "deDE", "frFR"), b"low locale")

    def test_higher_priority_wins_same_hash(self):
        self.assertEqual(self.read("b.bin"), b"high")

    def test_tie_goes_to_last_opened(self):
        last = A.read_archive(A.write_archive({"b.bin": b"last"}))
        self.overlay.add(last, 12)
        self.assertEqual(self.read("b.bin"), b"last")

    def test_miss(self):
        self.assertIsNone(self.overlay.resolve("zzz"))

    def test_priority_table(self):
        self.assertEqual(A.SWITCH_PRIORITIES["hotfix"], 12)
        self.assertEqual(A.SWITCH_PRIORITIES["cutscene"], -6)
