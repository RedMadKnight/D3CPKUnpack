import os
import random
import struct
import unittest
import zlib

import builders as B
from d3sno import FormatError
from d3sno import archive as A


def sample_stream():
    return b"".join(bytes([i]) * 700 for i in range(1, 6))


def sample_spec(**changes):
    """Two files, one with two copies, on a little-endian v7 archive."""
    first = b"A" * 1000
    second = random.Random(1).randbytes(5000)
    stream = first + first + second
    offsets, slots = B.simple_slots(stream, 0x1000, "<", chunk=700)
    spec = B.CpkSpec(
        files=[("Actor\\One.acr", 1000, 2, 0), ("StringList\\Foo.stl|frFR", 5000, 1, 2)],
        locations=[0, 1000, 2000],
        sector_offsets=offsets,
        stream_size=len(stream),
        slots=slots,
    )
    for key, value in changes.items():
        setattr(spec, key, value)
    return spec, first, second


def plain_spec(**changes):
    """The same two files with one location each, which is what v7 allows."""
    spec, first, second = sample_spec(**changes)
    stream = first + second
    spec.sector_offsets, spec.slots = B.simple_slots(stream, 0x1000, "<", chunk=700)
    spec.files = [(spec.files[0][0], 1000, 1, 0), (spec.files[1][0], 5000, 1, 1)]
    spec.locations = [0, 1000]
    spec.stream_size = len(stream)
    return spec, first, second


def failing(report):
    return {name for name, check in report.checks.items() if check.problems}


class Hashing(unittest.TestCase):
    def test_fnv_known_value(self):
        self.assertEqual(A.fnv1a64(b""), 0xCBF29CE484222325)
        self.assertEqual(A.fnv1a64(b"a"), 0xAF63DC4C8601EC8C)

    def test_name_hash_lowers_and_keeps_suffix(self):
        self.assertEqual(A.name_hash("ABC\\D.x|frFR"), A.fnv1a64(b"abc\\d.x|frfr"))
        self.assertNotEqual(A.name_hash("a.x|frFR"), A.name_hash("a.x"))


class Cipher(unittest.TestCase):
    def test_known_first_byte_and_round_trip(self):
        self.assertEqual(A.decrypt_packet(b"\x00"), b"\x61")
        plain = bytes(range(200))
        self.assertEqual(A.decrypt_packet(A.encrypt_packet(plain)), plain)
        self.assertEqual(A.decrypt_packet(B.encrypt(plain)), plain)


class Header(unittest.TestCase):
    def test_v7_fields(self):
        spec, *_ = sample_spec()
        h = A.read_archive(B.build_cpk(spec)).header
        self.assertEqual((h.version, h.order, h.header_size), (7, "<", 72))
        self.assertEqual((h.read_sector, h.comp_sector), (0x200, 0x1000))

    def test_v6_big_endian_fixed_sectors(self):
        stream = b"hello" * 100
        offsets, slots = B.simple_slots(stream, 0x4000, ">", empty_first=True)
        data = B.build_cpk(B.CpkSpec(
            files=[("a.bin", len(stream), 1, 0)], locations=[0], sector_offsets=offsets,
            stream_size=len(stream), slots=slots, version=6, order=">", comp_sector=0x4000,
            read_sector=0x10000,
        ))
        arc = A.read_archive(data)
        self.assertEqual(arc.header.header_size, 64)
        self.assertEqual((arc.header.read_sector, arc.header.comp_sector), (0x10000, 0x4000))
        self.assertEqual(arc.read_file("a.bin"), stream)

    def test_bad_magic(self):
        with self.assertRaises(FormatError):
            A.read_header(b"\0" * 80)

    def test_bad_version(self):
        data = bytearray(B.build_cpk(sample_spec()[0]))
        data[4] = 5
        with self.assertRaises(FormatError) as caught:
            A.read_header(bytes(data))
        self.assertEqual(caught.exception.offset, 4)

    def test_short_header(self):
        data = B.build_cpk(sample_spec()[0])[:40]
        with self.assertRaises(FormatError):
            A.read_header(data)

    def test_runtime_flag_bits(self):
        spec, *_ = sample_spec()
        data = bytearray(B.build_cpk(spec))
        data[0x10] |= 0x01
        with self.assertRaises(FormatError):
            A.read_archive(bytes(data))
        self.assertIn("header", failing(A.verify_archive(bytes(data))))

    def test_wrong_byte_order_for_version(self):
        spec, *_ = sample_spec(version=6, comp_sector=0x4000, read_sector=0x10000)
        spec.slots = B.simple_slots(b"", 0x4000, "<", empty_first=True)[1]
        spec.sector_offsets = [0]
        spec.stream_size = 0
        spec.files, spec.locations = [], []
        self.assertIn("header", failing(A.verify_archive(B.build_cpk(spec))))

    def test_oversized_sectors(self):
        spec, *_ = sample_spec()
        data = bytearray(B.build_cpk(spec))
        struct.pack_into("<I", data, 0x40, 0x8000)
        self.assertIn("header", failing(A.verify_archive(bytes(data))))

    def test_region_past_file(self):
        spec, *_ = sample_spec()
        data = bytearray(B.build_cpk(spec))
        struct.pack_into("<I", data, 0x1C, 0xFFFFF)
        with self.assertRaises(FormatError):
            A.read_archive(bytes(data))

    def test_sector_offset_bits_must_equal_location_bits(self):
        spec, *_ = sample_spec()
        data = bytearray(B.build_cpk(spec))
        struct.pack_into("<I", data, 0x30, 31)
        with self.assertRaises(FormatError):
            A.read_archive(bytes(data))
        self.assertIn("widths", failing(A.verify_archive(bytes(data))))

    def test_sector_index_bits_rule(self):
        spec, *_ = sample_spec()
        spec.sector_index_bits = 7
        data = B.build_cpk(spec)
        with self.assertRaises(FormatError):
            A.read_archive(data)
        self.assertIn("widths", failing(A.verify_archive(data)))

    def test_trailing_bytes_shift_tables(self):
        spec, *_ = sample_spec()
        data = B.build_cpk(spec) + bytes(0x2000)
        with self.assertRaises(FormatError):
            A.read_archive(data)

    def test_compact_flag_is_byte_offset(self):
        spec, first, second = sample_spec(flags=10)
        data = B.build_cpk(spec)
        arc = A.read_archive(data)
        self.assertEqual(arc.header.start, arc.header.header_sector)
        self.assertEqual(arc.read_file("Actor\\One.acr"), first)

    def test_uncompressed_raw_payload(self):
        stream = b"raw payload!" * 10
        data = B.build_cpk(B.CpkSpec(
            files=[("x.bin", len(stream), 1, 0)], locations=[0], sector_offsets=[],
            stream_size=len(stream), flags=0,
            raw_region=stream.ljust(0x1000, b"\0"),
        ))
        arc = A.read_archive(data)
        self.assertEqual(arc.sector_offsets, [])
        self.assertEqual(arc.read_file("x.bin"), stream)
        self.assertTrue(A.verify_archive(data).ok)


class Tables(unittest.TestCase):
    def test_join_names_and_locale_suffix(self):
        spec, first, second = sample_spec()
        arc = A.read_archive(B.build_cpk(spec))
        self.assertEqual(arc.names[0], "StringList\\Foo.stl|frFR")
        self.assertEqual(arc.read_file("stringlist\\FOO.stl|frfr"), second)
        self.assertEqual(arc.find("StringList\\Foo.stl|frFR").name, "StringList\\Foo.stl|frFR")
        self.assertIsNone(arc.find("StringList\\Foo.stl"))

    def test_copies(self):
        spec, first, _ = sample_spec()
        arc = A.read_archive(B.build_cpk(spec))
        self.assertEqual(arc.read_file("Actor\\One.acr", 1), first)
        with self.assertRaises(FormatError):
            arc.read_file("Actor\\One.acr", 2)

    def test_padding_reads_as_blob(self):
        spec, first, _ = sample_spec(padding_in_blob=40)
        arc = A.read_archive(B.build_cpk(spec))
        self.assertEqual(arc.read_file("Actor\\One.acr"), first)

    def test_name_offset_outside_blob(self):
        spec, *_ = sample_spec()
        data = bytearray(B.build_cpk(spec))
        arc = A.read_archive(bytes(data))
        struct.pack_into("<I", data, arc.tables_end - 8, 0xFFFF)
        with self.assertRaises(FormatError):
            A.read_archive(bytes(data))

    def test_table_truncation(self):
        spec, *_ = sample_spec()
        with self.assertRaises(FormatError):
            A.read_archive(B.build_cpk(spec)[:80])

    def test_missing_file(self):
        arc = A.read_archive(B.build_cpk(sample_spec()[0]))
        with self.assertRaises(FormatError):
            arc.read_file("no such file")

    def test_range_outside_stream(self):
        arc = A.read_archive(B.build_cpk(sample_spec()[0]))
        with self.assertRaises(FormatError):
            arc.read_stream(arc.header.stream_size - 1, 2)


class SectorRegion(unittest.TestCase):
    def one_file(self, slots, offsets, stream, order="<", **kw):
        return B.build_cpk(B.CpkSpec(
            files=[("f.bin", len(stream), 1, 0)], locations=[0], sector_offsets=offsets,
            stream_size=len(stream), slots=slots, order=order, **kw,
        ))

    def test_slot_zero_is_ordinary_on_switch(self):
        stream = b"first bytes are real"
        data = self.one_file([B.packet(stream, "<")], [0], stream)
        self.assertEqual(A.read_archive(data).read_file("f.bin"), stream)

    def test_slot_zero_with_empty_packet(self):
        stream = b"after the empty packet"
        slot = B.packet(b"", "<") + B.packet(stream, "<")
        arc = A.read_archive(self.one_file([slot], [0], stream))
        packets = arc.sector(0).packets
        self.assertEqual([p.declared for p in packets], [0, len(stream)])
        self.assertEqual(packets[0].stored, 8)
        self.assertEqual(arc.read_file("f.bin"), stream)

    def test_declared_size_above_16_bits(self):
        stream = bytes(0x12345)
        data = self.one_file([B.packet(stream, "<")], [0], stream)
        arc = A.read_archive(data)
        self.assertEqual(arc.sector(0).packets[0].declared, 0x12345)
        self.assertEqual(arc.read_file("f.bin"), stream)

    def test_big_endian_packet_header_low_word_first(self):
        stream = bytes(0x12345)
        spec = B.CpkSpec(
            files=[("f.bin", len(stream), 1, 0)], locations=[0], sector_offsets=[0],
            stream_size=len(stream), slots=[B.packet(stream, ">")], version=6, order=">",
            comp_sector=0x4000, read_sector=0x10000,
        )
        self.assertEqual(A.read_archive(B.build_cpk(spec)).read_file("f.bin"), stream)

    def test_stored_zero_ends_slot(self):
        stream = b"x" * 100
        junk = struct.pack("<HHH", 7, 0, 0) + b"NOT A PACKET"
        data = self.one_file([B.packet(stream, "<") + junk], [0], stream)
        self.assertEqual(A.read_archive(data).read_file("f.bin"), stream)

    def test_slot_ends_with_fewer_than_seven_bytes(self):
        # A slot filled to within six bytes of its end by one complete packet.
        plain = random.Random(6).randbytes(0x1000 - 6 - 11 - 3)
        packet = B.packet(plain, "<", stored=zlib.compress(plain, 0))
        self.assertLess(0x1000 - len(packet), 7)
        data = self.one_file([packet], [0], plain)
        self.assertEqual(A.read_archive(data).read_file("f.bin"), plain)

    def test_empty_header_before_real_data_is_hopped(self):
        stream = b"hopped" * 40
        slot = struct.pack("<HHH", 0, 0, 0) + B.packet(stream, "<")
        data = self.one_file([slot], [0], stream)
        self.assertEqual(A.read_archive(data).read_file("f.bin"), stream)

    def test_empty_header_with_a_declared_size_ends_the_slot(self):
        slot = struct.pack("<HHH", 9, 0, 0) + B.packet(b"x" * 100, "<")
        arc = A.read_archive(self.one_file([slot], [0], b""))
        self.assertEqual(arc.sector(0).packets, [])

    def test_bit_14_after_the_first_packet_is_reported(self):
        stream = b"one" * 50 + b"two" * 50
        tail = zlib.compress(stream[150:])
        slot = B.packet(stream[:150], "<") + struct.pack("<HHH", 150, 0, 0x4000) + tail
        data = self.one_file([slot], [0], stream)
        self.assertEqual(A.read_archive(data).read_file("f.bin"), stream)
        self.assertIn("packets", failing(A.verify_archive(data)))

    def test_cut_packet_trailer_missing(self):
        """The cut falls in the zlib checksum: all the data comes out, the stream never ends."""
        plain = random.Random(2).randbytes(0x1000 + 2 - 17)
        stored = zlib.compress(plain, 0)
        self.assertEqual(len(stored) + 6, 0x1000 + 2)
        slot = B.packet(plain, "<", stored=stored)[:0x1000]
        arc = A.read_archive(self.one_file([slot], [0], plain))
        packet = arc.sector(0).packets[0]
        self.assertTrue(packet.cut)
        self.assertEqual(packet.padded, 0)
        self.assertEqual(arc.read_file("f.bin"), plain)

    def test_cut_packet_zero_pads_and_reports(self):
        plain = random.Random(3).randbytes(0x1000 + 100)
        slot = B.packet(plain, "<", stored=zlib.compress(plain, 0))[:0x1000]
        data = self.one_file([slot], [0], plain)
        arc = A.read_archive(data)
        packet = arc.sector(0).packets[0]
        self.assertGreater(packet.padded, 0)
        self.assertTrue(arc.read_file("f.bin").endswith(bytes(packet.padded)))
        report = A.verify_archive(data)
        self.assertEqual(report.padded_bytes, packet.padded)
        self.assertEqual(report.cut_packets, 1)
        self.assertTrue(report.checks["packets"].ok)
        self.assertTrue(report.checks["packets"].notes)

    def test_excess_is_truncated(self):
        plain = bytes(range(250)) * 400
        declared = 65536 if len(plain) > 65536 else len(plain) - 100
        extra = zlib.compress(plain)
        slot = B.packet(plain, "<", declared=declared, stored=extra)
        stream = plain[:declared]
        data = self.one_file([slot], [0], stream)
        arc = A.read_archive(data)
        self.assertEqual(arc.sector(0).packets[0].excess, len(plain) - declared)
        self.assertEqual(arc.read_file("f.bin"), stream)

    def test_next_header_follows_stream_end_not_stored_size(self):
        one, two = b"one" * 50, b"two" * 50
        first = zlib.compress(one)
        # The stored size is stated too large: the inflater decides where the next header is.
        head = struct.pack("<HHH", len(one), 0, len(first) + 5)
        slot = head + first + B.packet(two, "<")
        stream = one + two
        data = self.one_file([slot], [0], stream)
        self.assertEqual(A.read_archive(data).read_file("f.bin"), stream)

    def test_encrypted_packet(self):
        one, two = b"secret" * 80, b"plain" * 90
        slot = B.packet(one, "<", encrypted=True) + B.packet(two, "<")
        stream = one + two
        data = self.one_file([slot], [0], stream)
        arc = A.read_archive(data)
        self.assertEqual([p.encrypted for p in arc.sector(0).packets], [True, False])
        self.assertEqual(arc.read_file("f.bin"), stream)
        self.assertEqual(A.verify_archive(data).encrypted_packets, 1)

    def test_empty_stub_slot(self):
        stream = b"payload" * 20
        slots = [B.packet(b"", "<"), B.packet(stream, "<")]
        data = self.one_file(slots, [0, 0], stream)
        self.assertEqual(A.read_archive(data).read_file("f.bin"), stream)
        report = A.verify_archive(data)
        self.assertTrue(report.ok, report.format())
        self.assertTrue(report.checks["sector_tables"].notes)

    def test_range_across_sectors(self):
        stream = random.Random(4).randbytes(9000)
        offsets, slots = B.simple_slots(stream, 0x1000, "<", chunk=1000)
        self.assertGreater(len(slots), 1)
        data = self.one_file(slots, offsets, stream)
        arc = A.read_archive(data)
        self.assertEqual(arc.read_stream(3500, 4000), stream[3500:7500])

    def test_bad_packet_raises(self):
        stream = b"z" * 50
        slot = struct.pack("<HHH", 50, 0, 20) + b"\xFF" * 20
        data = self.one_file([slot], [0], stream)
        with self.assertRaises(FormatError):
            A.read_archive(data).read_file("f.bin")
        self.assertIn("packets", failing(A.verify_archive(data)))

    def test_gap_mismatch_raises_on_read(self):
        stream = b"q" * 100 + b"r" * 100
        slots = [B.packet(stream[:100], "<"), B.packet(stream[100:], "<")]
        data = self.one_file(slots, [0, 120], stream)
        with self.assertRaises(FormatError):
            A.read_archive(data).read_file("f.bin")
        self.assertIn("sector_lengths", failing(A.verify_archive(data)))


class Verify(unittest.TestCase):
    def test_clean(self):
        report = A.verify_archive(B.build_cpk(plain_spec()[0]))
        self.assertTrue(report.ok, report.format())
        self.assertGreater(report.packets, 0)

    def test_unsorted(self):
        spec, *_ = plain_spec(unsorted=True)
        self.assertIn("order", failing(A.verify_archive(B.build_cpk(spec))))

    def test_unsigned_sort(self):
        """A hash with the top bit set sorts last, which a signed comparison would break."""
        names = ["n%d.bin" % i for i in range(30)]
        hashes = sorted(B.fnv(n) for n in names)
        self.assertGreaterEqual(hashes[-1], 1 << 63)
        data = A.write_archive({n: b"1" for n in names})
        self.assertTrue(A.verify_archive(data).ok)

    def test_duplicate_hash(self):
        spec, *_ = plain_spec()
        spec.files = [spec.files[0], spec.files[0]]
        spec.files[1] = ("Actor\\One.acr", 1000, 1, 2)
        self.assertIn("order", failing(A.verify_archive(B.build_cpk(spec))))

    def test_name_hash_mismatch(self):
        spec, *_ = plain_spec(wrong_name_hash=True)
        self.assertIn("names", failing(A.verify_archive(B.build_cpk(spec))))

    def test_crc(self):
        spec, *_ = plain_spec(bad_crc=True)
        self.assertEqual(failing(A.verify_archive(B.build_cpk(spec))), {"crc"})

    def test_crc_covers_region_to_end_only(self):
        spec, *_ = plain_spec()
        data = bytearray(B.build_cpk(spec))
        arc = A.read_archive(bytes(data))
        data[arc.tables_end] ^= 0  # the header area is not covered
        data[-1] ^= 0xFF
        self.assertEqual(failing(A.verify_archive(bytes(data))), {"crc"})

    def test_sector_offset_first_not_zero(self):
        spec, *_ = plain_spec()
        spec.sector_offsets = [1] + spec.sector_offsets[1:]
        self.assertIn("sector_tables", failing(A.verify_archive(B.build_cpk(spec))))

    def test_sector_index_past_true_sector(self):
        spec, *_ = plain_spec()
        k = (spec.stream_size + spec.comp_sector - 1) // spec.comp_sector
        spec.sector_index = [len(spec.sector_offsets) - 1] * k
        self.assertIn("sector_tables", failing(A.verify_archive(B.build_cpk(spec))))

    def test_stream_size_not_sum(self):
        spec, *_ = plain_spec()
        spec.files[1] = (spec.files[1][0], 4000, 1, 1)
        report = A.verify_archive(B.build_cpk(spec))
        self.assertIn("stream", failing(report))

    def test_gap_flagged_as_note(self):
        spec, *_ = plain_spec()
        spec.files[1] = (spec.files[1][0], 4000, 1, 1)
        spec.locations[1] = 2000
        data = B.build_cpk(spec)
        self.assertIn("stream", failing(A.verify_archive(data)))
        allowed = A.verify_archive(data, allow_gaps=True)
        self.assertNotIn("stream", failing(allowed))
        self.assertTrue(allowed.checks["stream"].notes)

    def test_location_out_of_range(self):
        spec, *_ = plain_spec()
        spec.locations[1] = spec.stream_size
        self.assertIn("locations", failing(A.verify_archive(B.build_cpk(spec))))

    def test_copy_table_overrun(self):
        spec, *_ = plain_spec()
        spec.files[1] = (spec.files[1][0], 5000, 2, 1)
        self.assertIn("locations", failing(A.verify_archive(B.build_cpk(spec))))

    def test_copies_differ_only_in_header_bytes(self):
        base = b"\xEF\xBE\xAD\xDE" + bytes(60)
        one = base[:8] + b"\x01" * 8 + base[16:]
        two = base[:8] + b"\x02" * 8 + base[16:]
        stream = one + two
        spec = B.CpkSpec(
            files=[("c.bin", 64, 2, 0)], locations=[0, 64], sector_offsets=[0],
            stream_size=128, slots=[B.packet(stream, ">")], version=6, order=">",
            comp_sector=0x4000, read_sector=0x10000,
        )
        self.assertTrue(A.verify_archive(B.build_cpk(spec)).ok)
        spec.slots = [B.packet(one + two[:40] + b"!" + two[41:], ">")]
        self.assertIn("copies", failing(A.verify_archive(B.build_cpk(spec))))

    def test_v7_with_two_locations(self):
        spec, *_ = sample_spec()
        self.assertIn("copies", failing(A.verify_archive(B.build_cpk(spec))))

    def test_format_mentions_failures(self):
        spec, *_ = plain_spec(bad_crc=True)
        text = A.verify_archive(B.build_cpk(spec)).format()
        self.assertIn("FAIL crc", text)

    def test_unreadable_header_skips_the_rest(self):
        report = A.verify_archive(b"nonsense" * 20)
        self.assertEqual(failing(report), {"header"})
        self.assertTrue(report.checks["crc"].skipped)
