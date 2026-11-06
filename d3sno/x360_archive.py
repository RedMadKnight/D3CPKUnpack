"""Byte-identical writer for big-endian version-7 archives (``20_corrections9.md`` G2-A1 to A9).

``archive.write_archive`` stays as it was: it writes valid archives with equal content. This
writer follows the nine rules under which every x360 archive is rebuilt byte for byte:

- A1: pieces are (file, copy) pairs, each with its own location; in the given order they tile the
  stream. Copies need not be adjacent.
- A2: files in hash order; ``first`` counts the locations of the files before it; its locations
  ascend. Name offsets follow the file order and the blob is in hash order. Widths are the bit
  lengths of: the largest size, the largest copy count, the stream size, the location count
  (first location), the slot count (sector index; 0 when uncompressed).
- A3: each piece is compressed on its own. A packet never holds bytes of two pieces; at a slot
  boundary the piece's stream is cut and the rest begins a new stream in the next slot. A slot is
  closed when its packet was cut, when it is full, or when fewer than 7 bytes remain.
- A4: ``.fsb`` and ``.sbt`` pieces of flag-2 archives use level 0; the rest level 9, window 15,
  memory level 8, default strategy, with canonical zlib output (see ``ZLIB_IS_CANONICAL``).
- A5: a cut level-9 packet stores the first ``room`` bytes of the stream of the whole rest of the
  piece and declares what they give when inflated into 65,536-byte buffers, not counting output
  still pending when the stored bytes run out.
- A6: a level-0 piece that does not fit: ``78 01``, one stored-block header, data cut at the slot
  end; block length and final bit from 16,384-byte reads (``stored_block_length``).
- A7: flag 2 pads the last slot; flag 10 ends with an empty 6-byte packet header whose encrypted
  bit copies the last packet's; flag 0 ends at the end of the stream.
- A8: header-sector word: flag 10 the byte offset of the sector region (name blob end rounded up
  to 4); flags 2 and 0 the number of read sectors that hold header and tables.
- A9: encryption is per packet, after compression; to reproduce an archive pass the source's set
  of (piece, position in piece, slot) (V9-A5; (piece, position) pairs are also accepted).

V9-A5: a cut level-9 packet can declare 0 bytes; the next packet of the piece then starts at the
same position in the next slot. Source packets are therefore keyed by (piece, position, slot).

Speed (writer note of ``21_corrections10.md``): a cut packet needs only the first ``room`` bytes of
the stream of the rest of the piece. With the default deflater (or a ``compressor`` factory) the
rest is fed to an incremental compressor that stops as soon as its output passes ``room``, so a
piece is read about once instead of once per cut packet. Deflate output does not depend on how its
input is split, so the bytes are those of compressing the whole rest.

``rebuild_x360`` takes the stream order and the encrypted packets from a source archive. When
this Python's zlib does not give the source's packet bytes (zlib-ng, for example), it keeps the
source packet at that place and reports how many it kept.
"""

from __future__ import annotations

import bisect
import struct
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .archive import (
    CPK_MAGIC, FLAG_COMPACT, FLAG_COMPRESSED, CpkArchive, decrypt_packet, encrypt_packet,
    name_hash, read_archive,
)
from .binary import BitWriter, FormatError

ORDER = ">"
HEADER_SIZE = 72
PACKET_HEADER = 6
MIN_PACKET_ROOM = 7  # a slot with fewer bytes left is closed
FEED_CHUNK = 0x1000  # input fed to an incremental compressor per step
INFLATE_BUFFER = 0x10000
STORED_READ = 0x4000
STORED_BLOCK_MIN = 32506
STORED_HEADER = bytes((0x78, 0x01))
LEVEL0_EXTENSIONS = (".fsb", ".sbt")
ZERO = bytes(1)


def zlib_implementation() -> str:
    """The zlib this Python runs, as it reports itself."""
    ng = getattr(zlib, "ZLIBNG_VERSION", None)
    runtime = getattr(zlib, "ZLIB_RUNTIME_VERSION", zlib.ZLIB_VERSION)
    return f"zlib-ng {ng} (zlib API {runtime})" if ng else f"zlib {runtime}"


# G2-A4: zlib-ng and other deflaters give other level-9 bytes than canonical zlib.
ZLIB_IS_CANONICAL = not (hasattr(zlib, "ZLIBNG_VERSION")
                         or "ng" in getattr(zlib, "ZLIB_RUNTIME_VERSION", zlib.ZLIB_VERSION))


def deflate9_object():
    """A fresh level-9 compressor (window 15, memory level 8, default strategy)."""
    return zlib.compressobj(9, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)


def deflate9(data: bytes) -> bytes:
    """Level 9, window 15, memory level 8, default strategy, with this Python's zlib."""
    packer = deflate9_object()
    return packer.compress(data) + packer.flush()


def stream_head(data, start: int, limit: int,
                compressor: Callable[[], object] = deflate9_object) -> tuple[bytes, bool]:
    """The stream of ``data[start:]``, or only enough of its start to pass ``limit`` bytes.

    Returns (bytes, complete). When ``complete`` is False the bytes are longer than ``limit`` and
    are a prefix of the whole stream: the compressor stopped early (writer speed note)."""
    view = memoryview(data)
    packer = compressor()
    out = bytearray()
    at = start
    while at < len(view):
        out += packer.compress(view[at:at + FEED_CHUNK])
        at += FEED_CHUNK
        if len(out) > limit:
            return bytes(out), False
    out += packer.flush()
    return bytes(out), True


def declared_for_cut(stored: bytes) -> int:
    """A5: bytes the stored part of a cut stream gives, inflated into 65,536-byte buffers,
    stopping when the stored bytes are used up (pending output is not counted)."""
    inflater = zlib.decompressobj()
    feed, total = stored, 0
    while True:
        out = inflater.decompress(feed, INFLATE_BUFFER)
        total += len(out)
        feed = inflater.unconsumed_tail
        if not feed or inflater.eof:
            return total


def stored_block_length(piece_size: int, start: int) -> tuple[int, bool]:
    """A6: the block length and final bit of a stored stream that starts ``start`` bytes into a
    piece of ``piece_size`` bytes. Reads are 16,384-byte pieces counted from the piece start
    (the first read runs to the next multiple of 16,384); the 32,506 threshold is checked first."""
    total = 0
    position = start
    while position < piece_size:
        step = min((position // STORED_READ + 1) * STORED_READ, piece_size) - position
        total += step
        position += step
        if total >= STORED_BLOCK_MIN:
            return total, False
    return piece_size - start, True


def stored_stream(rest: bytes, block: int, final: bool) -> bytes:
    """A6: ``78 01``, one stored-block header, the block, and the Adler-32 when final."""
    header = bytes((1 if final else 0,)) + struct.pack("<HH", block & 0xFFFF, ~block & 0xFFFF)
    body = STORED_HEADER + header + rest[:block]
    if final:
        body += struct.pack(">I", zlib.adler32(rest[:block]) & 0xFFFFFFFF)
    return body


@dataclass
class Piece:
    """A1: one copy of one file, in stream order."""

    name: str
    copy: int
    data: bytes


@dataclass
class PacketPlan:
    piece: int
    position: int  # bytes of the piece before this packet
    slot: int
    slot_offset: int
    declared: int
    stored: bytes  # before encryption
    encrypted: bool
    from_source: bool = False


@dataclass
class X360WriteResult:
    data: bytes
    packets: list[PacketPlan] = field(default_factory=list)
    kept_from_source: int = 0
    notes: list[str] = field(default_factory=list)


# (piece, position, slot, slot offset, room, level) -> (stored bytes, declared) or None
SourcePackets = Callable[[int, int, int, int, int, int], "tuple[bytes, int] | None"]


def _packets(pieces: Sequence[Piece], levels: Sequence[int], comp_sector: int,
             encrypted: frozenset, source: SourcePackets | None,
             compress: Callable[[bytes], bytes],
             compressor: Callable[[], object] | None = None,
             ) -> tuple[list[PacketPlan], list[int], list[bytearray]]:
    """A3 to A6, A9: the packets of every piece, slot by slot (slots not yet padded)."""
    plans: list[PacketPlan] = []
    offsets: list[int] = []
    slots: list[bytearray] = []
    slot_open = False
    stream_pos = 0
    for number, piece in enumerate(pieces):
        data, level = piece.data, levels[number]
        position = 0
        while position < len(data):
            if not slot_open:
                offsets.append(stream_pos)
                slots.append(bytearray())
                slot_open = True
            slot = slots[-1]
            room = comp_sector - len(slot) - PACKET_HEADER
            left = len(data) - position
            slot_index = len(slots) - 1
            if level == 0:
                block, final = stored_block_length(len(data), position)
                stream = stored_stream(data[position:position + block], block, final)
                if len(stream) <= room:
                    stored, declared, cut = stream, min(block, left), False
                else:
                    stored, cut = stream[:room], True
                    declared = min(room - 7, left) if room >= 7 else 0
            else:
                if compressor is not None:
                    stream, complete = stream_head(data, position, room, compressor)
                else:
                    stream, complete = compress(data[position:]), True
                if complete and len(stream) <= room:
                    stored, declared, cut = stream, left, False
                else:
                    stored, cut = stream[:room], True
                    declared = declared_for_cut(stored)  # may be 0 (V9-A5)
            kept = False
            if source is not None:
                found = source(number, position, slot_index, len(slot), room, level)
                if found is not None and found != (stored, declared):
                    stored, declared = found
                    cut = len(stored) >= room
                    kept = True
            is_encrypted = ((number, position, slot_index) in encrypted
                            or (number, position) in encrypted)
            written = encrypt_packet(stored) if is_encrypted else stored
            word = len(written) | (0x8000 if is_encrypted else 0)
            plans.append(PacketPlan(number, position, slot_index, len(slot), declared,
                                    stored, is_encrypted, kept))
            slot += struct.pack(ORDER + "HHH", declared & 0xFFFF, declared >> 16, word) + written
            position += declared
            stream_pos += declared
            if cut or comp_sector - len(slot) < MIN_PACKET_ROOM:
                slot_open = False
    return plans, offsets, slots


def default_levels(pieces: Sequence[Piece], flags: int) -> list[int]:
    """A4: level 0 for ``.fsb`` and ``.sbt`` pieces of flag-2 archives, else 9."""
    return [0 if flags == 2 and p.name.lower().endswith(LEVEL0_EXTENSIONS) else 9 for p in pieces]


def write_x360_archive(
    pieces: Sequence[Piece],
    *,
    flags: int = 2,
    read_sector: int = 0x10000,
    comp_sector: int = 0x4000,
    encrypted: Iterable[tuple[int, int]] = (),
    levels: Sequence[int] | None = None,
    source: SourcePackets | None = None,
    compress: Callable[[bytes], bytes] = deflate9,
    compressor: Callable[[], object] | None = None,
) -> X360WriteResult:
    """Write a big-endian version-7 archive by G2-A1 to A9.

    ``pieces`` are in stream order (any order for a new archive). ``levels`` gives each piece's
    level (default: A4). ``encrypted`` is the set of (piece index, position in piece, slot) or
    (piece index, position in piece) whose packets are encrypted. ``source`` may supply packet
    bytes to keep (see ``rebuild_x360``).

    ``compress`` turns a whole rest of a piece into a stream; ``compressor`` makes an incremental
    (``compressobj``-like) compressor used instead, which stops early on cut packets. With the
    default ``compress`` the default ``compressor`` is used; a custom ``compress`` without a
    ``compressor`` is called on the whole rest for every packet."""
    if flags not in (0, 2, 10):
        raise FormatError(f"cannot write flags {flags}")
    by_name: dict[str, list[int]] = {}
    for number, piece in enumerate(pieces):
        by_name.setdefault(piece.name, []).append(number)
    sizes = {}
    for name, numbers in by_name.items():
        found = {len(pieces[n].data) for n in numbers}
        if len(found) != 1:
            raise FormatError(f"copies of {name!r} differ in size")
        sizes[name] = found.pop()
    if levels is None:
        levels = default_levels(pieces, flags)
    if compressor is None and compress is deflate9:
        compressor = deflate9_object

    # A1: a location per piece, in stream order.
    locations_of: dict[int, int] = {}
    cursor = 0
    for number, piece in enumerate(pieces):
        locations_of[number] = cursor
        cursor += len(piece.data)
    stream_size = cursor

    # A2: files in hash order, each file's locations together and ascending.
    hashes = {name: name_hash(name) for name in by_name}
    if len(set(hashes.values())) != len(hashes):
        raise FormatError("two names share a hash")
    files = sorted(by_name, key=lambda n: hashes[n])
    locations: list[int] = []
    entries = []
    for name in files:
        entries.append((hashes[name], sizes[name], len(by_name[name]), len(locations)))
        locations.extend(sorted(locations_of[n] for n in by_name[name]))

    plans: list[PacketPlan] = []
    notes: list[str] = []
    sector_offsets: list[int] = []
    region = bytearray()
    if flags & FLAG_COMPRESSED:
        plans, sector_offsets, slots = _packets(
            pieces, levels, comp_sector, frozenset(encrypted), source, compress, compressor)
        for slot in slots[:-1]:
            region += bytes(slot).ljust(comp_sector, ZERO)
        last_slot = bytes(slots[-1]) if slots else b""
        if flags & FLAG_COMPACT:  # A7, flag 10: an empty header ends the file inside the slot
            last = plans[-1].encrypted if plans else False
            end = struct.pack(ORDER + "HHH", 0, 0, 0x8000 if last else 0)
            if comp_sector - len(last_slot) >= len(end):
                region += last_slot + end
            else:
                region += last_slot.ljust(comp_sector, ZERO)
                notes.append("no room for the end header in the last slot; the slot is padded")
        elif slots:  # A7, flag 2
            region += last_slot.ljust(comp_sector, ZERO)
    else:
        region += b"".join(p.data for p in pieces)  # A7, flag 0

    size_bits = max(sizes.values(), default=0).bit_length()
    copies_bits = max((len(v) for v in by_name.values()), default=0).bit_length()
    location_bits = stream_size.bit_length()
    first_bits = len(locations).bit_length()
    index_bits = len(sector_offsets).bit_length() if flags & FLAG_COMPRESSED else 0
    decomp_count = (stream_size + comp_sector - 1) // comp_sector

    bits = BitWriter()
    for file_hash, size, count, first in entries:
        bits.write(file_hash, 64)
        bits.write(size, size_bits)
        bits.write(count, copies_bits)
        bits.write(first, first_bits)
    tables = bytearray(bits.finish())
    bits = BitWriter()
    for location in locations:
        bits.write(location, location_bits)
    tables += bits.finish()
    if flags & FLAG_COMPRESSED:
        bits = BitWriter()
        for offset in sector_offsets:
            bits.write(offset, location_bits)
        tables += bits.finish()
        bits = BitWriter()
        for k in range(decomp_count):
            bits.write(max(bisect.bisect_right(sector_offsets, k * comp_sector) - 1, 0), index_bits)
        tables += bits.finish()
    blob = bytearray()
    for name in files:  # A2: name offsets follow the file order, the blob is in hash order
        tables += struct.pack(ORDER + "I", len(blob))
        blob += name.encode("latin-1") + ZERO
    tables += blob

    tables_end = HEADER_SIZE + len(tables)
    if flags & FLAG_COMPACT:  # A8
        header_sector = start = (tables_end + 3) // 4 * 4
    else:
        header_sector = (tables_end + read_sector - 1) // read_sector
        start = header_sector * read_sector
    crc = zlib.crc32(region) & 0xFFFFFFFF
    header = struct.pack(
        ORDER + "IIQ12I", CPK_MAGIC, 7, stream_size, flags, len(files), len(locations),
        header_sector, size_bits, copies_bits, first_bits, location_bits, location_bits,
        index_bits, crc, read_sector,
    ) + struct.pack(ORDER + "II", comp_sector, 0)
    data = header + bytes(tables).ljust(start - HEADER_SIZE, ZERO) + bytes(region)
    return X360WriteResult(data, plans, sum(p.from_source for p in plans), notes)


# --- Rebuild from a source archive ---------------------------------------------------------

def source_pieces(archive: CpkArchive) -> list[Piece]:
    """A1: the source's pieces in location (stream) order; a zero-size piece goes before the
    piece it shares its location with."""
    found = []
    for entry in archive.files:
        if entry.name is None:
            raise FormatError(f"file 0x{entry.hash:016X} has no name, cannot rebuild it")
        for copy in range(entry.count):
            found.append((archive.locations[entry.first + copy], entry.size != 0, entry, copy))
    found.sort(key=lambda item: (item[0], item[1]))
    return [Piece(entry.name, copy, archive.read_file(entry, copy)) for _, _, entry, copy in found]


def source_packets(archive: CpkArchive, pieces: Sequence[Piece]):
    """Every source packet keyed by (piece, position in piece, slot) (V9-A5: a cut packet that
    declares 0 bytes shares its position with the next one): its slot offset, stored bytes
    (decrypted) and declared size; and the set of encrypted keys (A9)."""
    h = archive.header
    starts, numbers = [], []
    cursor = 0
    for number, piece in enumerate(pieces):
        if piece.data:
            starts.append(cursor)
            numbers.append(number)
        cursor += len(piece.data)
    found: dict[tuple[int, int, int], tuple[int, bytes, int]] = {}
    encrypted: set[tuple[int, int, int]] = set()
    if not h.compressed:
        return found, encrypted
    data = archive.data
    for slot in range(len(archive.sector_offsets)):
        stream_at = archive.sector_offsets[slot]
        slot_start = archive.slot_position(slot)
        for packet in archive.sector(slot).packets:
            k = bisect.bisect_right(starts, stream_at) - 1
            if k < 0:
                break
            key = (numbers[k], stream_at - starts[k], slot)
            payload = packet.offset + PACKET_HEADER
            stored = bytes(data[payload:min(payload + packet.stored, slot_start + h.comp_sector)])
            if packet.encrypted:
                stored = decrypt_packet(stored)
                encrypted.add(key)
            found[key] = (packet.offset - slot_start, stored, packet.declared)
            stream_at += packet.declared
    return found, encrypted


def rebuild_x360(source: bytes | CpkArchive, compress: Callable[[bytes], bytes] = deflate9,
                 keep_source_packets: bool = True,
                 compressor: Callable[[], object] | None = None) -> X360WriteResult:
    """Rebuild a big-endian version-7 archive from its parsed content by G2-A.

    Taken from the source: the stream order (A1) and the encrypted packets (A9). Where this
    Python's compressor gives other packet bytes at the same place (non-canonical zlib), the
    source packet is kept and counted in ``kept_from_source``."""
    archive = source if isinstance(source, CpkArchive) else read_archive(source)
    h = archive.header
    if (h.version, h.order) != (7, ">"):
        raise FormatError("not a big-endian version-7 archive")
    pieces = source_pieces(archive)
    packets, encrypted = source_packets(archive, pieces)

    def lookup(number: int, position: int, slot: int, slot_offset: int, room: int, level: int):
        hit = packets.get((number, position, slot))
        if hit is None or hit[0] != slot_offset:
            return None
        return hit[1], hit[2]

    result = write_x360_archive(
        pieces, flags=h.flags, read_sector=h.read_sector, comp_sector=h.comp_sector,
        encrypted=encrypted, source=lookup if keep_source_packets else None, compress=compress,
        compressor=compressor,
    )
    if result.kept_from_source:
        result.notes.append(
            f"{result.kept_from_source} of {len(result.packets)} packets kept from the source: "
            f"{zlib_implementation()} does not reproduce them"
            + ("" if ZLIB_IS_CANONICAL else " (not canonical zlib, G2-A4)"))
    if result.data != bytes(archive.data):
        result.notes.append("the rebuild is not byte-identical to the source")
    return result
