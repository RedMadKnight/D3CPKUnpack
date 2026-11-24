"""Stage 15, sound: ``30_create_content.md`` W15-B4 to W15-B7 (FSB5 writer, sound-bank rows, switch
``.ogg`` samples, mode-16 frame layout), synthetic data only."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import struct
import unittest

import builders as B
from d3sno import derived as D
from d3sno import presentation as P
from d3sno import sound as S
from d3sno import sound_write as W
from d3sno.binary import FormatError

BUILDS = ("ps3", "switch", "x360")
ORDER = {"ps3": ">", "switch": "<", "x360": ">"}


def packets(n=100, size=37):
    return [bytes([(i * 7 + k) & 0xFF for k in range(size + i % 5)]) for i in range(n)]


def vorbis_fsb(n=100, duration=1000, channels=2, rate_index=8, crc=0xC0FFEE, name="Bank-0001",
               extra=None, chunks=()):
    pk = packets(n)
    return W.build_vorbis_fsb5(pk, W.packet_positions([duration] * n), n * duration, rate_index,
                               channels, crc, name, extra, chunks), pk


def id_header(channels=1, rate=44100):
    return (b"\x01vorbis" + struct.pack("<IBIiiiBB", 0, channels, rate, 0, 128000, 0, 0xB8, 1))


def ogg_file(channels=1, rate=44100, final=5000, serial=7):
    audio = [bytes([i]) * (50 + i) for i in range(4)]
    pk = [id_header(channels, rate), b"\x03vorbis" + bytes(20), b"\x05vorbis" + bytes(300)] + audio
    granules = [0, 0, 0, 1000, 2000, 4000, final]
    return W.build_ogg(pk, granules, serial)


def frames_fsb(pcm=1000, channels=1, name="Bank-0002"):
    count = W.padded_frame_samples(pcm)
    frame = W.write_frame(W.SwitchFrame((0, 1, 2, 3, 4, 0, 1, 2), (0, 12, 3, 4, 5, 6, 7, 8), (-5, 9),
                                        tuple((i % 16) - 8 for i in range(256))))
    data = frame * (count // 256 * channels)
    return W.build_fsb5(16, 8, channels, count, data, (), name)


# --- W15-B4 -----------------------------------------------------------------------------------

class Fsb5Writer(unittest.TestCase):
    def test_vorbis_header_and_sizes(self):
        extra = bytes(range(24))
        data, pk = vorbis_fsb(extra=extra, chunks=[(3, struct.pack("<II", 0, 99))])
        version, count, h, n, d, mode, flag, zero = struct.unpack_from("<8I", data, 4)
        self.assertEqual(data[:4], b"FSB5")
        self.assertEqual((version, count, mode, flag, zero), (1, 1, 15, 1, 0))
        fsb = P.parse_fsb5(data)
        self.assertEqual(h, 8 + sum(4 + len(c.data) for c in fsb.sample.chunks))
        self.assertEqual(len(data), 60 + h + n + d)
        self.assertEqual(d % 32, 0)
        self.assertEqual((60 + h + n) % 32, 0)
        self.assertEqual(data[0x24:0x3C], extra)
        self.assertEqual(fsb.name, "Bank-0001")
        self.assertEqual((fsb.sample.rate_index, fsb.sample.stereo, fsb.sample.data_offset,
                          fsb.sample.sample_count), (8, True, 0, 100000))
        self.assertEqual(fsb.packets(), pk)
        self.assertEqual(P.fsb5_problems(fsb, "SoundBank\\Bank\\0001.fsb"), [])
        self.assertEqual(fsb.sample.loop, (0, 99))
        self.assertEqual(struct.unpack_from("<Q", data, 60)[0] & 1, 1)

    def test_round_trip(self):
        data, _ = vorbis_fsb()
        self.assertEqual(P.write_fsb5(P.parse_fsb5(data)), data)
        f16 = frames_fsb()
        self.assertEqual(P.write_fsb5(P.parse_fsb5(f16)), f16)

    def test_no_chunks_bit(self):
        data = W.build_fsb5(15, 8, 1, 10, W.vorbis_data_area([b"ab"]), (), "x-0000")
        self.assertEqual(struct.unpack_from("<Q", data, 60)[0] & 1, 0)
        self.assertEqual(struct.unpack_from("<I", data, 12)[0], 8)

    def test_data_area(self):
        area = W.vorbis_data_area([b"abc", b"de"])
        self.assertEqual(area[:9], b"\x03\x00abc\x02\x00de")
        self.assertEqual(len(area), 32)
        self.assertFalse(any(area[9:]))
        self.assertEqual(W.vorbis_packet_offsets([b"abc", b"de", b"f"]), [0, 5, 9])
        with self.assertRaises(FormatError):
            W.vorbis_data_area([b""])
        with self.assertRaises(FormatError):
            W.vorbis_data_area([bytes(0x10000)])

    def test_seek_table(self):
        positions = W.packet_positions([1000] * 100)
        offsets = list(range(0, 400, 4))
        table = W.vorbis_seek_table(positions, offsets, 100000)
        self.assertEqual(table, [(44000, 176), (88000, 352)])
        self.assertEqual(W.vorbis_seek_table(positions, offsets, 44099), [])
        self.assertEqual(W.vorbis_seek_table(positions, offsets, 44100), [(44000, 176)])
        self.assertEqual(W.vorbis_seek_table(positions, offsets, 100000, first=0)[0], (0, 0))
        self.assertEqual(W.vorbis_seek_table([0, 44100], [0, 9], 44100), [(44100, 9)])

    def test_vorbis_chunk(self):
        chunk = W.vorbis_chunk(0xABCD, [(1, 2), (3, 4)])
        self.assertEqual(chunk.type, 11)
        self.assertEqual(len(chunk.data), 8 + 16)
        self.assertEqual(struct.unpack_from("<II", chunk.data), (0xABCD, 16))
        data, pk = vorbis_fsb()
        crc, entries = P.parse_fsb5(data).sample.vorbis_setup
        self.assertEqual(crc, 0xC0FFEE)
        offsets = W.vorbis_packet_offsets(pk)
        self.assertEqual(entries, [(44000, offsets[44]), (88000, offsets[88])])

    def test_setup_crcs(self):
        a, _ = vorbis_fsb(crc=5)
        b, _ = vorbis_fsb(crc=6, channels=1)
        known = W.setup_crcs([a, b, frames_fsb()])
        self.assertEqual(known, {(8, 2): {5}, (8, 1): {6}})
        self.assertEqual(W.pick_setup_crc(known, 8, 1), 6)
        with self.assertRaises(FormatError):
            W.pick_setup_crc(known, 5, 1)

    def test_header_extra(self):
        a, _ = vorbis_fsb(extra=bytes(range(1, 25)))
        self.assertEqual(W.fsb5_header_extra(a), bytes(range(1, 25)))
        with self.assertRaises(FormatError):
            W.build_fsb5(15, 8, 1, 1, b"\1\0x", (), "n", bytes(23))

    def test_mode16(self):
        data = frames_fsb(pcm=1000, channels=2)
        fsb = P.parse_fsb5(data)
        self.assertEqual(struct.unpack_from("<I", data, 28)[0], 0)
        self.assertEqual(fsb.sample.sample_count, 1024)
        self.assertEqual(len(fsb.data_area), -(-4 * 2 * 140 // 32) * 32)
        self.assertEqual(P.fsb5_problems(fsb, "SoundBank\\Bank\\0002.fsb"), [])
        self.assertFalse(any(data[0x24:0x2C]))

    def test_mode16_rules(self):
        frame = bytes(140)
        with self.assertRaises(FormatError):
            W.build_fsb5(16, 8, 1, 1000, frame * 4, (), "n")  # not a multiple of 256
        with self.assertRaises(FormatError):
            W.build_fsb5(16, 8, 1, 1024, frame * 3, (), "n")  # too few frames
        with self.assertRaises(FormatError):
            W.build_fsb5(16, 8, 1, 1024, frame * 5, (), "n")  # too many
        with self.assertRaises(FormatError):
            W.build_fsb5(16, 8, 1, 256, frame, (), "n", b"\1" + bytes(23))
        ok = W.build_fsb5(16, 8, 1, 256, frame, (), "n", bytes(8) + b"\1" * 16)
        self.assertEqual(ok[0x2C:0x3C], b"\1" * 16)
        self.assertEqual(len(W.frame_data_area(frame + bytes(20), 256, 1)), 160)
        with self.assertRaises(FormatError):
            W.frame_data_area(frame + b"\1" + bytes(19), 256, 1)

    def test_bad_arguments(self):
        for args in ((14, 8, 1), (15, 16, 1), (15, 8, 3)):
            with self.assertRaises(FormatError):
                W.build_fsb5(*args, 1, b"", (), "n")


# --- W15-B5 -----------------------------------------------------------------------------------

class BankRows(unittest.TestCase):
    def test_codes(self):
        self.assertEqual(W.compression_code("switch", "ogg"), 3)
        self.assertEqual(W.compression_code("switch", "vorbis"), 4)
        self.assertEqual(W.compression_code("switch", "frames"), 7)
        self.assertEqual(W.compression_code("x360", "vorbis"), 4)
        self.assertEqual(W.compression_code("ps3", "vorbis"), 5)
        for build, kind in (("ps3", "ogg"), ("x360", "frames")):
            with self.assertRaises(FormatError):
                W.compression_code(build, kind)
        self.assertEqual((W.format_code(1, 5), W.format_code(1, 8), W.format_code(2, 8)), (7, 8, 11))
        with self.assertRaises(FormatError):
            W.format_code(2, 5)

    def test_row_pack(self):
        v = S.BankRowValues(0x1234, 7, 11, 300, 4000, 999, True, False)
        for build in BUILDS:
            raw = S.pack_bank_row(v, build)
            self.assertEqual(len(raw), 40 if build == "ps3" else 16)
            self.assertEqual(S.bank_row_values(raw, build), v)
        word = struct.unpack_from("<I", S.pack_bank_row(v, "switch"), 4)[0]
        self.assertEqual(word, 1 | 7 << 2 | 300 << 12 | 11 << 22)
        self.assertEqual(struct.unpack(">10I", S.pack_bank_row(v, "ps3")),
                         (0x1234, 1, 0, 7, 300, 4000, 11, 0, 999, 0))

    def test_vorbis_row(self):
        data, _ = vorbis_fsb()
        for build in BUILDS:
            row = W.new_bank_row(build, data, "vorbis", hash=0x55, quality=40)
            v = S.bank_row_values(row, build)
            self.assertEqual((v.hash, v.stored_size, v.decompressed_size, v.format, v.quality),
                             (0x55, len(data), 2 * 2 * 100000, 11, 40))
            self.assertEqual(v.compression, {"ps3": 5, "switch": 4, "x360": 4}[build])
        self.assertEqual(W.new_bank_row("switch", data, None, hash=0x55),
                         W.new_bank_row("switch", data, "vorbis", hash=0x55))

    def test_frames_row(self):
        data = frames_fsb(pcm=1000, channels=1)
        v = S.bank_row_values(W.new_bank_row("switch", data, "frames", 1000, hash=1), "switch")
        self.assertEqual((v.compression, v.format, v.decompressed_size), (7, 8, 2000))
        v = S.bank_row_values(W.new_bank_row("switch", data, "frames", hash=1), "switch")
        self.assertEqual(v.decompressed_size, 2048)
        with self.assertRaises(FormatError):
            W.new_bank_row("switch", data, "frames", 700, hash=1)
        with self.assertRaises(FormatError):
            W.new_bank_row("switch", data, "vorbis", hash=1)

    def test_ogg_row(self):
        data = ogg_file(channels=1, final=5000)
        v = S.bank_row_values(W.new_bank_row("switch", data, "ogg", hash=1), "switch")
        self.assertEqual((v.compression, v.format, v.decompressed_size, v.stored_size), (3, 8, 10000, len(data)))
        stereo = ogg_file(channels=2, final=300)
        v = S.bank_row_values(W.new_bank_row("switch", stereo, hash=1), "switch")
        self.assertEqual((v.format, v.decompressed_size), (11, 1200))
        mono22 = ogg_file(channels=1, rate=22050)
        self.assertEqual(S.bank_row_values(W.new_bank_row("switch", mono22, hash=1), "switch").format, 7)
        with self.assertRaises(FormatError):
            W.new_bank_row("ps3", data, hash=1)

    def test_hash(self):
        data, _ = vorbis_fsb()
        for build in BUILDS:
            bank = B.build_sbk(build, [(0x11, 5), (0x12, 6)])
            row = W.new_bank_row(build, data, bank=bank)
            h = S.bank_row_values(row, build).hash
            self.assertNotIn(h, (0, 0x11, 0x12))
            self.assertEqual(W.unused_sample_hash(bank, 0x11, build), 0x13)
            self.assertEqual(W.unused_sample_hash(bank, 0, build), 1)
            with self.assertRaises(FormatError):
                W.new_bank_row(build, data, hash=0x12, bank=bank)
        with self.assertRaises(FormatError):
            W.new_bank_row("switch", data)

    def test_append_row(self):
        data, _ = vorbis_fsb()
        for build in BUILDS:
            bank = B.build_sbk(build, [(0x11, 5)])
            row = W.new_bank_row(build, data, hash=0x99)
            out, index = W.append_bank_row(bank, row, build)
            self.assertEqual(index, 1)
            parsed = S.parse_sound_bank(out, build)
            self.assertEqual([r.hash for r in parsed.rows], [0x11, 0x99])
            self.assertEqual(parsed.rows[1].stored_size, len(data))
            self.assertEqual(out[:len(bank)][0x40:], bank[0x40:])
            with self.assertRaises(FormatError):
                W.append_bank_row(out, row, build)

    def test_update_bank_row(self):
        order = "<"
        bank = B.build_sbk("switch", [(1, 0)])
        at = struct.unpack_from(order + "I", bank, 0x1C)[0] + 0x10
        ogg = ogg_file(channels=2, final=777)
        out = D.update_bank_row(bank, 0, ogg, "switch")
        self.assertEqual(struct.unpack_from("<II", out, at + 8), (2 * 2 * 777, len(ogg)))
        self.assertEqual(struct.unpack_from("<I", out, at + 4)[0], 0)  # codes kept by default
        out = D.update_bank_row(bank, 0, ogg, "switch", codes=True)
        self.assertEqual(S.bank_row_values(out[at:at + 16], "switch").compression, 3)
        self.assertEqual(S.bank_row_values(out[at:at + 16], "switch").format, 11)
        f16 = frames_fsb(pcm=1000)
        self.assertEqual(struct.unpack_from("<I", D.update_bank_row(bank, 0, f16, "switch", original_samples=1000),
                                            at + 8)[0], 2000)
        self.assertEqual(struct.unpack_from("<I", D.update_bank_row(bank, 0, f16, "switch"), at + 8)[0], 2048)
        broken = bytearray(ogg)
        broken[-1] ^= 1
        out = D.update_bank_row(bank, 0, bytes(broken), "switch")
        self.assertEqual(struct.unpack_from("<II", out, at + 8), (0, len(ogg)))
        ps3 = B.build_sbk("ps3", [(1, 0)])
        pat = struct.unpack_from(">I", ps3, 0x1C)[0] + 0x10
        data, _ = vorbis_fsb()
        v = S.bank_row_values(D.update_bank_row(ps3, 0, data, "ps3", codes=True)[pat:pat + 40], "ps3")
        self.assertEqual((v.compression, v.format, v.decompressed_size), (5, 11, 400000))


# --- W15-B6 -----------------------------------------------------------------------------------

class OggSamples(unittest.TestCase):
    def test_good_file(self):
        data = ogg_file()
        self.assertEqual(W.ogg_problems(data), [])
        pages, rest = W.parse_ogg(data)
        self.assertEqual(rest, b"")
        self.assertTrue(pages[0].bos and pages[-1].eos)
        self.assertEqual(W.ogg_final_granule(data), 5000)
        self.assertEqual(W.vorbis_identification(W.ogg_packets(pages)[0][0]), (1, 44100))

    def test_crc(self):
        # the CRC of a page with an empty body, worked by hand from the polynomial
        self.assertEqual(W.ogg_crc(b""), 0)
        self.assertEqual(W.ogg_crc(b"\x00\x00\x00\x01"), 0x04C11DB7)
        data = bytearray(ogg_file())
        data[40] ^= 0xFF
        self.assertTrue(any("CRC" in p for p in W.ogg_problems(bytes(data))))

    def test_long_packet_lacing(self):
        pk = [id_header(), b"\x03vorbis", b"\x05vorbis" + bytes(600), bytes(255)]
        data = W.build_ogg(pk, [0, 0, 0, 10])
        pages, _ = W.parse_ogg(data)
        self.assertEqual(pages[2].lacing, (255, 255, 97))  # 7 + 600 bytes
        self.assertEqual(pages[3].lacing, (255, 0))
        self.assertEqual(W.ogg_packets(pages)[0], pk)
        self.assertEqual(W.ogg_problems(data), [])

    def test_streams_and_flags(self):
        good = ogg_file()
        pages, _ = W.parse_ogg(good)
        other = W.ogg_page(0, 5, 99, len(pages) - 1, [b"x"])
        mixed = b"".join(p.raw for p in pages[:-1]) + other
        self.assertTrue(any("streams" in p for p in W.ogg_problems(mixed)))
        no_eos = b"".join(p.raw for p in pages[:-1])
        self.assertTrue(any("end-of-stream flag missing" in p for p in W.ogg_problems(no_eos)))
        early = W.ogg_page(2 | 4, 0, 7, 0, [id_header()]) + b"".join(p.raw for p in pages[1:])
        self.assertTrue(any("page 0: end-of-stream flag set" in p for p in W.ogg_problems(early)))
        self.assertTrue(any("bytes after" in p for p in W.ogg_problems(good + b"junk")))
        gap = good[:pages[3].offset] + W.ogg_page(0, 1000, 7, 9, [b"a"]) + good[pages[4].offset:]
        self.assertTrue(any("page number" in p for p in W.ogg_problems(gap)))
        no_headers = W.build_ogg([b"a", b"b", b"c"], [0, 0, 3])
        self.assertEqual(len([p for p in W.ogg_problems(no_headers) if "Vorbis header" in p]), 3)
        self.assertTrue(W.ogg_problems(b"RIFF" + bytes(40)))

    def test_parse_ogg_rejects(self):
        with self.assertRaises(FormatError):
            W.parse_ogg(b"OggS" + bytes(10))


# --- W15-B7 -----------------------------------------------------------------------------------

class FrameLayout(unittest.TestCase):
    def test_fields(self):
        frame = bytearray(140)
        struct.pack_into("<IIhh", frame, 0, 0x43210, 0xC0000005, -2, 300)
        frame[12] = 0x8F  # low nibble -1, high nibble -8
        frame[13] = 0x71
        f = W.parse_frame(bytes(frame))
        self.assertEqual(f.predictors, (0, 1, 2, 3, 4, 0, 0, 0))
        self.assertEqual(f.shifts, (5, 0, 0, 0, 0, 0, 0, 12))
        self.assertEqual(f.history, (-2, 300))
        self.assertEqual(f.samples[:4], (-1, -8, 1, 7))
        self.assertEqual(len(f.samples), 256)
        self.assertEqual(W.write_frame(f), bytes(frame))
        self.assertEqual(W.frame_problems(bytes(frame)), [])

    def test_ranges(self):
        frame = bytearray(140)
        struct.pack_into("<II", frame, 0, 5 << 8, 13 << 28)
        problems = W.frame_problems(bytes(frame))
        self.assertEqual(problems, ["sub-frame 2: predictor 5 > 4", "sub-frame 7: shift 13 > 12"])
        self.assertTrue(W.frame_problems(bytes(139)))
        with self.assertRaises(FormatError):
            W.parse_frame(bytes(141))
        with self.assertRaises(FormatError):
            W.write_frame(W.SwitchFrame((0,) * 8, (0,) * 8, (0, 0), (8,) + (0,) * 255))

    def test_file(self):
        self.assertEqual(W.fsb5_frame_problems(frames_fsb(pcm=600, channels=2)), [])
        bad = bytearray(frames_fsb(pcm=256))
        data_at = len(bad) - len(P.parse_fsb5(bytes(bad)).data_area)
        bad[data_at] = 0x0F
        self.assertEqual(W.fsb5_frame_problems(bytes(bad)), ["frame 0: sub-frame 0: predictor 15 > 4"])
        self.assertEqual(W.fsb5_frame_problems(vorbis_fsb()[0]), ["FSB5 mode 15, not 16"])


if __name__ == "__main__":
    unittest.main()
