"""Stage 15 sound content (``30_create_content.md`` W15-B4 to W15-B7).

- W15-B4 ``build_fsb5``: a one-sample FSB5 file (mode 15 Vorbis, mode 16 switch frame codec), with
  ``vorbis_data_area`` / ``vorbis_packet_offsets`` / ``vorbis_seek_table`` / ``vorbis_chunk`` for the
  Vorbis data area and its type-11 chunk, and ``frame_data_area`` for mode 16.
- W15-B5 ``new_bank_row`` (and ``derived.update_bank_row``): sound-bank row values, compression and
  format codes, an unused sample hash, ``append_bank_row``.
- W15-B6 ``parse_ogg`` / ``ogg_problems``: switch ``.ogg`` samples page by page.
- W15-B7 ``parse_frame`` / ``frame_problems``: the 140-byte mode-16 frame, layout only (the
  predictor coefficients are OPEN; nothing here decodes or encodes audio).

Vorbis audio is never encoded here: the caller supplies packets (FSB5) or a whole Ogg file.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from .binary import FormatError, build_order
from .presentation import (
    CHUNK_VORBIS, DATA_ALIGN, FRAME_SAMPLES, FRAME_SIZE, FSB5_HEADER, MODE_FRAMES, MODE_VORBIS,
    FsbChunk, fsb5_name_table, parse_fsb5,
)
from .sound import BankRowValues, SoundBank, pack_bank_row, parse_sound_bank
from .sno import parse_header, resolve_build

# --- W15-B4. FSB5 sample file writer --------------------------------------------------------

FSB5_VERSION = 1
HEADER_EXTRA = 24  # header bytes 0x24..0x3B: 8 bytes (zero for mode 16) + 16 bytes (OPEN)
FLAG_WORD = {MODE_VORBIS: 1, MODE_FRAMES: 0}  # header +0x1C by mode (PROVEN)
SEEK_INTERVAL = 44100  # samples per seek entry (SUPPORTED)
RATE_44100 = 8
# The public FSB5 rate-index table; only index 8 = 44,100 Hz is stated by the description.
FSB5_RATES = {1: 8000, 2: 11000, 3: 11025, 4: 16000, 5: 22050, 6: 24000, 7: 32000, 8: 44100, 9: 48000}
MAX_CHUNK = (1 << 24) - 1
MAX_SAMPLES = (1 << 30) - 1


def _pad(data: bytes, align: int = DATA_ALIGN) -> bytes:
    return bytes(data) + bytes(-len(data) % align)


def _chunk(chunk) -> FsbChunk:
    if isinstance(chunk, FsbChunk):
        return chunk
    kind, data = chunk
    return FsbChunk(int(kind), bytes(data))


def fsb5_header_extra(fsb_bytes: bytes) -> bytes:
    """W15-B4: header bytes 0x24..0x3B of another FSB5 file (another copy of the same sample),
    for ``build_fsb5(header_extra=...)``."""
    if len(fsb_bytes) < FSB5_HEADER or fsb_bytes[:4] != b"FSB5":
        raise FormatError("not an FSB5 file", 0)
    return bytes(fsb_bytes[0x24:FSB5_HEADER])


def frame_data_area(frames: bytes, sample_count: int, channels: int) -> bytes:
    """W15-B4 / S11-3: the mode-16 data area, ceil(samples / 256) x channels frames of 140 bytes
    (in the caller's order), then zeros to a multiple of 32. ``frames`` may already carry that
    zero padding."""
    if sample_count % FRAME_SAMPLES:
        raise FormatError(f"mode-16 sample count {sample_count} is not a multiple of {FRAME_SAMPLES}")
    need = -(-sample_count // FRAME_SAMPLES) * channels * FRAME_SIZE
    area = _pad(frames)
    if len(frames) < need or len(area) != len(_pad(bytes(need))) or any(area[need:]):
        raise FormatError(f"mode-16 data of {len(frames)} bytes, {need} bytes of frames expected")
    return area


def vorbis_data_area(packets: Iterable[bytes]) -> bytes:
    """W15-B4: each packet as a u16 length and its bytes, then zeros to a multiple of 32."""
    out = bytearray()
    for i, packet in enumerate(packets):
        if not 0 < len(packet) <= 0xFFFF:
            raise FormatError(f"Vorbis packet {i} of {len(packet)} bytes (1..65535)")
        out += struct.pack("<H", len(packet)) + packet
    return _pad(bytes(out))


def vorbis_packet_offsets(packets: Sequence[bytes]) -> list[int]:
    """Byte offset of each packet (of its u16 length word) in the data area."""
    offsets, at = [], 0
    for packet in packets:
        offsets.append(at)
        at += 2 + len(packet)
    return offsets


def packet_positions(durations: Iterable[int]) -> list[int]:
    """Sample position at the start of each packet from the samples each packet gives."""
    out, at = [], 0
    for d in durations:
        out.append(at)
        at += d
    return out


def vorbis_seek_table(positions: Sequence[int], offsets: Sequence[int], sample_count: int,
                      interval: int = SEEK_INTERVAL, first: int = 1,
                      count: int | None = None) -> list[tuple[int, int]]:
    """W15-B4 (SUPPORTED): entries (sample position, byte offset) for k = ``first`` ..; entry k is
    the last packet boundary at or before sample ``interval`` x k. ``positions`` are the packets'
    start samples (ascending), ``offsets`` their data-area offsets. ``count`` defaults to
    floor(samples / interval) (``31`` V18-8: 99.1 % of the files; ps3 99.8 %, x360 99.3 %,
    switch 97.1 %), still the rule for new sounds. Stored tables are not a reference for their
    last entry: in about 1.5 % of files one entry (as far as seen always the last) points at no
    packet (see ``vorbis_seek_problems``)."""
    if len(positions) != len(offsets):
        raise FormatError(f"{len(positions)} positions for {len(offsets)} packets")
    if any(b < a for a, b in zip(positions, positions[1:])):
        raise FormatError("packet positions are not ascending")
    if count is None:
        count = sample_count // interval
    entries = []
    for k in range(first, first + count):
        target = interval * k
        j = None
        for i, p in enumerate(positions):
            if p > target:
                break
            j = i
        if j is None:
            raise FormatError(f"no packet starts at or before sample {target}")
        entries.append((positions[j], offsets[j]))
    return entries


def vorbis_chunk(setup_crc: int, entries: Iterable[tuple[int, int]] = ()) -> FsbChunk:
    """W15-B4: the type-11 chunk {u32 setup CRC, u32 table size in bytes, entries
    {u32 sample position, u32 byte offset}}; chunk size = 8 + table size (PROVEN)."""
    table = b"".join(struct.pack("<II", s, o) for s, o in entries)
    return FsbChunk(CHUNK_VORBIS, struct.pack("<II", setup_crc & 0xFFFFFFFF, len(table)) + table)


def vorbis_seek_problems(fsb, allow_unmatched_last: bool = True,
                         check_count: bool = False) -> list[str]:
    """W15-B4 / ``31`` V18-8: a mode-15 file's type-11 seek table against its own packets.

    Reports an entry whose byte offset is not the start of a packet, a sample position that
    goes down, and (``check_count``) a count other than floor(samples / 44,100), which is only
    the rule of 99.1 % of the files and therefore off by default. In about 1.5 % of the stored
    files one entry, as far as seen always the last, points at no packet; that is not a writer
    error, so with ``allow_unmatched_last`` (default) such a last entry is exempt from every
    per-entry test, the sample order included (``32`` V19-2: it may hold a lower sample position
    than the entry before it). ``33`` V20-2: the last entry is exempt whenever it is tolerated,
    also when its offset happens to be a packet start (one x360 junk entry holds sample 1 at
    the start of an earlier packet). Only the last entry is exempt: an unmatched or out-of-order
    entry elsewhere is still reported, and ``allow_unmatched_last=False`` tests every entry. ``fsb`` is FSB5 bytes or a parsed ``Fsb5``."""
    if isinstance(fsb, (bytes, bytearray)):
        fsb = parse_fsb5(bytes(fsb))
    if fsb.mode != MODE_VORBIS:
        return []
    setup = fsb.sample.vorbis_setup
    if setup is None:
        return ["no Vorbis chunk (type 11)"]
    entries = setup[1]
    found = []
    boundaries = set(vorbis_packet_offsets(fsb.packets()))
    for k, (position, offset) in enumerate(entries):
        # ``32`` V19-2 / ``33`` V20-2: with ``allow_unmatched_last`` the last entry is tolerated and
        # exempt from every per-entry test, even when its offset happens to be a packet start
        if allow_unmatched_last and k == len(entries) - 1:
            continue
        if offset not in boundaries:
            found.append(f"seek entry {k}: offset {offset} is not a packet start")
        if k and position < entries[k - 1][0]:
            found.append(f"seek entry {k}: sample {position} before the previous entry's")
    if check_count:
        rule = fsb.sample.sample_count // SEEK_INTERVAL
        if len(entries) != rule:
            found.append(f"{len(entries)} seek entries, floor(samples / {SEEK_INTERVAL}) = {rule}")
    return found


def setup_crcs(fsb_files: Iterable[bytes]) -> dict[tuple[int, int], set[int]]:
    """W15-B4: the Vorbis setup CRCs a set of a build's FSB5 files uses, by (rate index, channels)."""
    found: dict[tuple[int, int], set[int]] = {}
    for data in fsb_files:
        fsb = parse_fsb5(data)
        setup = fsb.sample.vorbis_setup if fsb.mode == MODE_VORBIS else None
        if setup is not None:
            found.setdefault((fsb.sample.rate_index, fsb.channels), set()).add(setup[0])
    return found


def pick_setup_crc(known: dict[tuple[int, int], set[int]], rate_index: int, channels: int) -> int:
    """W15-B4: a setup CRC the build already uses with this rate and channel count."""
    values = known.get((rate_index, channels))
    if not values:
        raise FormatError(f"no setup CRC known for rate index {rate_index}, {channels} channels")
    return min(values)


def build_fsb5(mode: int, rate_index: int, channels: int, sample_count: int, data: bytes,
               chunks: Iterable = (), name: str = "", header_extra: bytes | None = None) -> bytes:
    """W15-B4: a one-sample FSB5 file.

    Header: magic, version 1, count 1, sample-header size H = 8 + sum(4 + chunk size), name-table
    size N (V11-6 padding rule), data size D, mode, flag word (1 for mode 15, 0 for mode 16), four
    zero bytes, then ``header_extra`` (24 bytes for 0x24..0x3B; zeros by default; the first 8 must
    be zero for mode 16). Sample word: bit 0 chunks follow, rate index, stereo bit, data offset 0,
    sample count. ``data`` is the data area (``vorbis_data_area`` / ``frame_data_area``); it is
    padded with zeros to a multiple of 32. Chunks are (type, bytes) or ``FsbChunk``. File size =
    60 + H + N + D."""
    if mode not in FLAG_WORD:
        raise FormatError(f"FSB5 mode {mode}: 15 (Vorbis) or 16 (frame codec)")
    if not 0 <= rate_index < 16:
        raise FormatError(f"rate index {rate_index} does not fit 4 bits")
    if channels not in (1, 2):
        raise FormatError(f"{channels} channels: 1 or 2")
    if not 0 <= sample_count <= MAX_SAMPLES:
        raise FormatError(f"sample count {sample_count} does not fit 30 bits")
    extra = bytes(HEADER_EXTRA) if header_extra is None else bytes(header_extra)
    if len(extra) != HEADER_EXTRA:
        raise FormatError(f"header_extra of {len(extra)} bytes, {HEADER_EXTRA} expected")
    if mode == MODE_FRAMES:
        if any(extra[:8]):
            raise FormatError("header bytes 0x24..0x2B must be zero for mode 16")
        area = frame_data_area(data, sample_count, channels)
    else:
        area = _pad(data)
    chunk_list = [_chunk(c) for c in chunks]
    raw = bytearray()
    for i, chunk in enumerate(chunk_list):
        if not 0 <= chunk.type < 128 or len(chunk.data) > MAX_CHUNK:
            raise FormatError(f"chunk {i}: type {chunk.type}, {len(chunk.data)} bytes")
        more = 1 if i + 1 < len(chunk_list) else 0
        raw += struct.pack("<I", more | len(chunk.data) << 1 | chunk.type << 25) + chunk.data
    word = (1 if chunk_list else 0) | rate_index << 1 | (channels == 2) << 5 | sample_count << 34
    headers = struct.pack("<Q", word) + bytes(raw)
    table = fsb5_name_table(name, len(headers))
    head = b"FSB5" + struct.pack("<8I", FSB5_VERSION, 1, len(headers), len(table), len(area), mode,
                                 FLAG_WORD[mode], 0) + extra
    return head + headers + table + area


def build_vorbis_fsb5(packets: Sequence[bytes], positions: Sequence[int], sample_count: int,
                      rate_index: int, channels: int, setup_crc: int, name: str,
                      header_extra: bytes | None = None, extra_chunks: Iterable = (),
                      seek_table: bool = True) -> bytes:
    """W15-B4 convenience: mode 15 from audio packets and their start positions; the type-11
    chunk gets the seek table (SUPPORTED; ``seek_table=False`` writes an empty table)."""
    entries = vorbis_seek_table(positions, vorbis_packet_offsets(packets), sample_count) if seek_table else []
    chunks = list(extra_chunks) + [vorbis_chunk(setup_crc, entries)]
    return build_fsb5(MODE_VORBIS, rate_index, channels, sample_count, vorbis_data_area(packets),
                      chunks, name, header_extra)


def padded_frame_samples(samples: int) -> int:
    """W15-B5: the mode-16 sample count of the file, the PCM length rounded up to 256."""
    return -(-samples // FRAME_SAMPLES) * FRAME_SAMPLES


# --- W15-B6. Ogg pages -----------------------------------------------------------------------

OGG_CONTINUED, OGG_BOS, OGG_EOS = 1, 2, 4
OGG_HEADER = 27


def _ogg_table() -> list[int]:
    table = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if r & 0x80000000 else (r << 1) & 0xFFFFFFFF
        table.append(r)
    return table


_OGG_CRC = _ogg_table()


def ogg_crc(data: bytes) -> int:
    """The Ogg page CRC (polynomial 0x04C11DB7, initial 0, not reflected)."""
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _OGG_CRC[(crc >> 24) ^ b]
    return crc


@dataclass(frozen=True)
class OggPage:
    offset: int
    version: int
    flags: int
    granule: int  # signed; -1 when no packet ends on the page
    serial: int
    sequence: int
    crc: int
    lacing: tuple[int, ...]
    body: bytes
    raw: bytes

    @property
    def continued(self) -> bool:
        return bool(self.flags & OGG_CONTINUED)

    @property
    def bos(self) -> bool:
        return bool(self.flags & OGG_BOS)

    @property
    def eos(self) -> bool:
        return bool(self.flags & OGG_EOS)

    @property
    def crc_ok(self) -> bool:
        blank = bytearray(self.raw)
        blank[22:26] = bytes(4)
        return ogg_crc(bytes(blank)) == self.crc


def parse_ogg(data: bytes) -> tuple[list[OggPage], bytes]:
    """W15-B6: the pages of an Ogg file and any bytes after the last whole page."""
    pages, pos = [], 0
    while pos < len(data):
        if data[pos:pos + 4] != b"OggS" or pos + OGG_HEADER > len(data):
            break
        version, flags, granule, serial, sequence, crc, nseg = struct.unpack_from("<BBqIIIB", data, pos + 4)
        if pos + OGG_HEADER + nseg > len(data):
            break
        lacing = tuple(data[pos + OGG_HEADER:pos + OGG_HEADER + nseg])
        end = pos + OGG_HEADER + nseg + sum(lacing)
        if end > len(data):
            break
        pages.append(OggPage(pos, version, flags, granule, serial, sequence, crc, lacing,
                             bytes(data[pos + OGG_HEADER + nseg:end]), bytes(data[pos:end])))
        pos = end
    if not pages:
        raise FormatError("not an Ogg file", 0)
    return pages, bytes(data[pos:])


def ogg_packets(pages: Sequence[OggPage]) -> tuple[list[bytes], bool]:
    """The packets of the pages by their lacing; True when the last packet is complete."""
    packets, current, at = [], bytearray(), 0
    complete = True
    for page in pages:
        at = 0
        for size in page.lacing:
            current += page.body[at:at + size]
            at += size
            if size < 255:
                packets.append(bytes(current))
                current = bytearray()
                complete = True
            else:
                complete = False
    if current:
        complete = False
    return packets, complete


def vorbis_identification(packet: bytes) -> tuple[int, int]:
    """(channels, rate) of a Vorbis identification header."""
    if len(packet) < 30 or packet[:7] != b"\x01vorbis":
        raise FormatError("not a Vorbis identification header")
    channels = packet[11]
    rate = struct.unpack_from("<I", packet, 12)[0]
    return channels, rate


def ogg_problems(data: bytes) -> list[str]:
    """W15-B6: a switch ``.ogg`` sample parses page by page to its end with one logical stream,
    good CRCs, version 0, consecutive page numbers, BOS on the first page only, EOS on the last
    page only, and the three Vorbis headers (identification, comment, setup) first."""
    try:
        pages, rest = parse_ogg(data)
    except FormatError as exc:
        return [str(exc)]
    found = []
    if rest:
        found.append(f"{len(rest)} bytes after the last whole page at 0x{len(data) - len(rest):X}")
    serials = {p.serial for p in pages}
    if len(serials) != 1:
        found.append(f"{len(serials)} logical streams, one expected")
    for i, p in enumerate(pages):
        if p.version:
            found.append(f"page {i}: version {p.version}")
        if p.flags & ~7:
            found.append(f"page {i}: flags 0x{p.flags:X}")
        if not p.crc_ok:
            found.append(f"page {i}: CRC mismatch")
        if p.bos != (i == 0):
            found.append(f"page {i}: beginning-of-stream flag {'set' if p.bos else 'missing'}")
        if p.eos != (i == len(pages) - 1):
            found.append(f"page {i}: end-of-stream flag {'set' if p.eos else 'missing'}")
        if i and p.sequence != pages[i - 1].sequence + 1:
            found.append(f"page {i}: page number {p.sequence} after {pages[i - 1].sequence}")
    if pages[0].continued:
        found.append("page 0 continues a packet")
    packets, complete = ogg_packets(pages)
    if not complete:
        found.append("the last packet is not complete")
    for k, kind in enumerate((1, 3, 5)):
        if len(packets) <= k or packets[k][:7] != bytes([kind]) + b"vorbis":
            found.append(f"packet {k} is not the Vorbis header of type {kind}")
    return found


def ogg_final_granule(data: bytes) -> int:
    """W15-B5: the final granule position of the stream (the last page that sets one)."""
    pages, _ = parse_ogg(data)
    for page in reversed(pages):
        if page.granule != -1:
            return page.granule
    raise FormatError("no page sets a granule position")


def ogg_page(flags: int, granule: int, serial: int, sequence: int, segments: Sequence[bytes]) -> bytes:
    """One Ogg page with ``segments`` as whole packets (each laced; for tests and simple files)."""
    lacing, body = bytearray(), bytearray()
    for packet in segments:
        n = len(packet)
        lacing += bytes([255] * (n // 255) + [n % 255])
        body += packet
    if len(lacing) > 255:
        raise FormatError("too many lacing values for one page")
    head = b"OggS" + struct.pack("<BBqIIIB", 0, flags, granule, serial, sequence, 0, len(lacing))
    page = bytearray(head + bytes(lacing) + bytes(body))
    struct.pack_into("<I", page, 22, ogg_crc(bytes(page)))
    return bytes(page)


def build_ogg(packets: Sequence[bytes], granules: Sequence[int], serial: int = 1) -> bytes:
    """An Ogg stream of one packet per page; BOS on the first page and EOS on the last."""
    if len(packets) != len(granules):
        raise FormatError(f"{len(granules)} granules for {len(packets)} packets")
    out = bytearray()
    for i, (packet, granule) in enumerate(zip(packets, granules)):
        flags = (OGG_BOS if i == 0 else 0) | (OGG_EOS if i == len(packets) - 1 else 0)
        out += ogg_page(flags, granule, serial, i, [packet])
    return bytes(out)


# --- W15-B5. Sound-bank row values -----------------------------------------------------------

KIND_OGG, KIND_VORBIS, KIND_FRAMES = "ogg", "vorbis", "frames"
COMPRESSION = {
    "switch": {KIND_OGG: 3, KIND_VORBIS: 4, KIND_FRAMES: 7},
    "x360": {KIND_VORBIS: 4},
    "ps3": {KIND_VORBIS: 5},
}
FORMAT = {(1, 5): 7, (1, 8): 8, (2, 8): 11}  # (channels, rate index) -> format code (PROVEN)


def sample_kind(sample_bytes: bytes) -> str:
    """``ogg``, ``vorbis`` (FSB5 mode 15) or ``frames`` (FSB5 mode 16)."""
    if sample_bytes[:4] == b"OggS":
        return KIND_OGG
    if sample_bytes[:4] == b"FSB5":
        mode = parse_fsb5(sample_bytes).mode
        if mode == MODE_VORBIS:
            return KIND_VORBIS
        if mode == MODE_FRAMES:
            return KIND_FRAMES
        raise FormatError(f"FSB5 mode {mode}")
    raise FormatError("neither FSB5 nor Ogg", 0)


@dataclass(frozen=True)
class SampleInfo:
    kind: str
    channels: int
    rate_index: int | None
    decompressed_size: int


def sample_info(sample_bytes: bytes, kind: str | None = None,
                original_samples: int | None = None) -> SampleInfo:
    """W15-B5: decompressed size = 2 x channels x the original sample count: the FSB5 count for
    Vorbis; for the frame codec the caller's PCM length (``original_samples``; without it the
    padded count of the file, W14-22's upper bound); for ``.ogg`` the final granule position."""
    kind = kind or sample_kind(sample_bytes)
    if kind == KIND_OGG:
        problems = ogg_problems(sample_bytes)
        if problems:
            raise FormatError(f"Ogg sample: {problems[0]}")
        pages, _ = parse_ogg(sample_bytes)
        channels, rate = vorbis_identification(ogg_packets(pages)[0][0])
        index = next((i for i, r in FSB5_RATES.items() if r == rate), None)
        return SampleInfo(kind, channels, index, 2 * channels * ogg_final_granule(sample_bytes))
    fsb = parse_fsb5(sample_bytes)
    want = {KIND_VORBIS: MODE_VORBIS, KIND_FRAMES: MODE_FRAMES}.get(kind)
    if want != fsb.mode:
        raise FormatError(f"FSB5 mode {fsb.mode} is not a {kind} sample")
    samples = fsb.sample.sample_count
    if kind == KIND_FRAMES and original_samples is not None:
        if padded_frame_samples(original_samples) != samples:
            raise FormatError(f"{original_samples} PCM samples do not pad to the file's {samples}")
        samples = original_samples
    return SampleInfo(kind, fsb.channels, fsb.sample.rate_index, 2 * fsb.channels * samples)


def format_code(channels: int, rate_index: int | None) -> int:
    """W15-B5: 7 mono at rate index 5, 8 mono at rate index 8, 11 stereo at rate index 8."""
    code = FORMAT.get((channels, rate_index))
    if code is None:
        raise FormatError(f"no format code for {channels} channels at rate index {rate_index}")
    return code


def compression_code(build: str, kind: str) -> int:
    """W15-B5: switch 3 Ogg, 4 FSB5 Vorbis, 7 FSB5 frame codec; x360 4; ps3 5."""
    code = COMPRESSION.get(build, {}).get(kind)
    if code is None:
        raise FormatError(f"{build} has no {kind} samples")
    return code


def _bank(bank, build: str | None) -> SoundBank:
    return bank if isinstance(bank, SoundBank) else parse_sound_bank(bank, build)


def unused_sample_hash(bank, seed: int = 0, build: str | None = None) -> int:
    """W15-B5: a nonzero sample hash no row of ``bank`` (bytes or ``SoundBank``) uses; the first
    free value from ``seed`` upward."""
    used = {row.hash for row in _bank(bank, build).rows}
    value = seed & 0xFFFFFFFF or 1
    while value in used or value == 0:
        value = (value + 1) & 0xFFFFFFFF
    return value


def new_bank_row(build: str, sample_bytes: bytes, kind: str | None = None,
                 original_samples: int | None = None, hash: int | None = None, *,
                 bank=None, quality: int = 0, verified: bool = False, temporary: bool = False,
                 format: int | None = None) -> bytes:
    """W15-B5: a ``.sbk`` row for a sample file: stored size = file size, decompressed size and
    format by ``sample_info`` / ``format_code``, compression by build and kind. ``hash`` is
    authored; without one, ``bank`` must be given and a value it does not use is taken (seeded
    by the CRC-32 of the file). Quality and the two flag bits are inputs."""
    if build not in COMPRESSION:
        raise FormatError(f"unknown build {build!r}")
    info = sample_info(sample_bytes, kind, original_samples)
    if hash is None:
        if bank is None:
            raise FormatError("a new row needs a hash or the bank to pick an unused one")
        hash = unused_sample_hash(bank, zlib.crc32(sample_bytes), build)
    elif bank is not None and _bank(bank, build).find(hash) is not None:
        raise FormatError(f"sample hash 0x{hash:08X} is already in the bank")
    values = BankRowValues(
        hash & 0xFFFFFFFF, compression_code(build, info.kind),
        format_code(info.channels, info.rate_index) if format is None else format,
        quality, info.decompressed_size, len(sample_bytes), verified, temporary,
    )
    return pack_bank_row(values, build)


_TABLE_PAIR = 0x1C  # file offset of the sample table pair
_COUNT_AT = {"switch": 0x30, "x360": 0x30, "ps3": 0x24}  # file offset of the sample count


def append_bank_row(bank: bytes, row: bytes, build: str | None = None) -> tuple[bytes, int]:
    """Add a row at the end of a bank's sample table, which must be the last block of the file;
    the pair size and the count word are set. Returns the new bank and the row index (the
    ``NNNN`` of the sample file name)."""
    header = parse_header(bank, build)
    build = resolve_build("sbk", header, build)
    parsed = parse_sound_bank(bank, build)
    size = len(parsed.rows[0].raw) if parsed.rows else {"ps3": 40}.get(build, 16)
    if len(row) != size:
        raise FormatError(f"row of {len(row)} bytes, {size} expected")
    if parsed.find(struct.unpack_from(header.order + "I", row, 0)[0]) is not None:
        raise FormatError("the row's sample hash is already in the bank")
    offset, length = struct.unpack_from(header.order + "II", bank, _TABLE_PAIR)
    if offset == 0 and length == 0:
        offset = len(bank) - 0x10
    if offset + 0x10 + length != len(bank):
        raise FormatError("the sample table is not the last block of the file", _TABLE_PAIR)
    out = bytearray(bank) + row
    struct.pack_into(header.order + "II", out, _TABLE_PAIR, offset, length + len(row))
    struct.pack_into(header.order + "i", out, _COUNT_AT[build], len(parsed.rows) + 1)
    return bytes(out), len(parsed.rows)


# --- W15-B7. Switch frame codec (mode 16): layout only ---------------------------------------

PREDICTOR_MAX = 4
SHIFT_MAX = 12
SUBFRAMES = 8


@dataclass(frozen=True)
class SwitchFrame:
    predictors: tuple[int, ...]  # one per 32-sample sub-frame, lowest nibble first
    shifts: tuple[int, ...]
    history: tuple[int, int]  # int16 x 2 (HYPOTHESIS: last and second-to-last previous output)
    samples: tuple[int, ...]  # 256 signed 4-bit values, low nibble first


def _nibbles(word: int) -> tuple[int, ...]:
    return tuple(word >> (4 * i) & 15 for i in range(SUBFRAMES))


def parse_frame(frame: bytes) -> SwitchFrame:
    """W15-B7: the fields of one 140-byte frame (little-endian). No decoding."""
    if len(frame) != FRAME_SIZE:
        raise FormatError(f"frame of {len(frame)} bytes, {FRAME_SIZE} expected")
    pred, shift, h0, h1 = struct.unpack_from("<IIhh", frame, 0)
    samples = []
    for b in frame[12:]:
        for n in (b & 15, b >> 4):
            samples.append(n - 16 if n >= 8 else n)
    return SwitchFrame(_nibbles(pred), _nibbles(shift), (h0, h1), tuple(samples))


def write_frame(frame: SwitchFrame) -> bytes:
    """W15-B7: a frame from its fields (layout only, not an encoder)."""
    if len(frame.predictors) != SUBFRAMES or len(frame.shifts) != SUBFRAMES:
        raise FormatError("eight predictors and eight shifts expected")
    if len(frame.samples) != FRAME_SAMPLES:
        raise FormatError(f"{len(frame.samples)} samples, {FRAME_SAMPLES} expected")
    pred = sum((p & 15) << (4 * i) for i, p in enumerate(frame.predictors))
    shift = sum((s & 15) << (4 * i) for i, s in enumerate(frame.shifts))
    body = bytearray()
    for i in range(0, FRAME_SAMPLES, 2):
        lo, hi = frame.samples[i], frame.samples[i + 1]
        if not (-8 <= lo <= 7 and -8 <= hi <= 7):
            raise FormatError(f"sample {i} out of the 4-bit range")
        body.append((lo & 15) | (hi & 15) << 4)
    return struct.pack("<IIhh", pred, shift, *frame.history) + bytes(body)


def frame_problems(frame: bytes) -> list[str]:
    """W15-B7 (SUPPORTED): frame size, predictor indexes 0..4 and shifts 0..12."""
    if len(frame) != FRAME_SIZE:
        return [f"frame of {len(frame)} bytes, {FRAME_SIZE} expected"]
    f = parse_frame(frame)
    found = [f"sub-frame {i}: predictor {p} > {PREDICTOR_MAX}" for i, p in enumerate(f.predictors)
             if p > PREDICTOR_MAX]
    found += [f"sub-frame {i}: shift {s} > {SHIFT_MAX}" for i, s in enumerate(f.shifts) if s > SHIFT_MAX]
    return found


def fsb5_frame_problems(fsb_bytes: bytes) -> list[str]:
    """W15-B7 over every frame of a mode-16 FSB5 file (padding excluded)."""
    fsb = parse_fsb5(fsb_bytes)
    if fsb.mode != MODE_FRAMES:
        return [f"FSB5 mode {fsb.mode}, not 16"]
    found = []
    expected = fsb.frame_bytes()
    if len(fsb.data) < expected:
        found.append(f"{len(fsb.data)} bytes of frames, {expected} expected")
    for i, frame in enumerate(fsb.packets()):
        found += [f"frame {i}: {p}" for p in frame_problems(frame)]
    return found


__all__ = [
    "COMPRESSION", "FORMAT", "FSB5_RATES", "OggPage", "SampleInfo", "SwitchFrame",
    "append_bank_row", "build_fsb5", "build_ogg", "build_vorbis_fsb5", "compression_code",
    "format_code", "frame_data_area", "frame_problems", "fsb5_frame_problems", "fsb5_header_extra",
    "new_bank_row", "ogg_crc", "ogg_final_granule", "ogg_packets", "ogg_page", "ogg_problems",
    "packet_positions", "padded_frame_samples", "parse_frame", "parse_ogg", "pick_setup_crc",
    "sample_info", "sample_kind", "setup_crcs", "unused_sample_hash", "vorbis_chunk",
    "vorbis_data_area", "vorbis_identification", "vorbis_packet_offsets", "vorbis_seek_problems",
    "vorbis_seek_table", "write_frame",
]
