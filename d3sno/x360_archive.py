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

``47`` Y1: ``write_x360_archive_to`` writes into a file object; a piece's data may be a
``archive.StreamView`` of a source archive, read in pieces while its packets are made, and closed
slots go to a temporary region file at once. ``write_x360_archive`` streams into memory.
"""

from __future__ import annotations

import bisect
import io
import struct
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .archive import (
    CPK_MAGIC, FLAG_COMPACT, FLAG_COMPRESSED, CpkArchive, StreamView, Written, _Region, _spool,
    as_output, as_workers, decrypt_packet, encrypt_packet, file_span, name_hash, read_archive,
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
    view = memoryview(data) if isinstance(data, (bytes, bytearray, memoryview)) else data
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


# ``48`` Z4: input compressed ahead from a piece's start, in comp sectors (a piece whose stream start
# does not pass the largest room within it is compressed by the packer)
AHEAD_INPUT_SECTORS = 16


def _head_ahead(data: bytes, whole: bool, limit: int) -> tuple[bytes, list[int], bool]:
    """Z4, in a worker: ``stream_head`` of ``data`` from 0 for every limit up to ``limit`` at once:
    the output, its length after each fed chunk, and whether the stream is complete (``whole``:
    ``data`` is the whole piece)."""
    packer = deflate9_object()
    out = bytearray()
    lengths = []
    for at in range(0, len(data), FEED_CHUNK):
        out += packer.compress(data[at:at + FEED_CHUNK])
        lengths.append(len(out))
        if len(out) > limit:
            return bytes(out), lengths, False
    if not whole:
        return bytes(out), lengths, False
    out += packer.flush()
    return bytes(out), lengths, True


def _answer_head(found, room: int):
    """Z4: what ``stream_head(data, 0, room)`` gives, from ``_head_ahead``; None when the input
    compressed ahead ends before the stream passes ``room``."""
    out, lengths, complete = found
    for length in lengths:
        if length > room:
            return out[:length], False
    return (out, True) if complete else None


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
    data: bytes | None
    packets: list[PacketPlan] = field(default_factory=list)
    kept_from_source: int = 0
    notes: list[str] = field(default_factory=list)
    written: Written | None = None  # Y1: size and SHA-256 of what was written


@dataclass
class _Packed:
    plans: list
    offsets: list
    last_slot: bytearray | None  # the last slot, not yet padded (None: no slot)
    kept: int = 0
    last_encrypted: bool = False
    notes: list = field(default_factory=list)


# (piece, position, slot, slot offset, room, level) -> (stored bytes, declared) or None
SourcePackets = Callable[[int, int, int, int, int, int], "tuple[bytes, int] | None"]


def _packets(pieces: Sequence[Piece], levels: Sequence[int], comp_sector: int,
             encrypted: frozenset, source: SourcePackets | None,
             compress: Callable[[bytes], bytes],
             compressor: Callable[[], object] | None = None,
             emit: Callable[[bytes], None] | None = None, keep_plans: bool = True,
             workers=None,
             ) -> _Packed:
    """A3 to A6, A9: the packets of every piece, slot by slot. Y1: each closed slot but the last
    goes, padded, to ``emit`` as soon as the next one opens; the last is returned unpadded.
    ``keep_plans`` False keeps no packet plan (their stored bytes would grow with the archive).
    ``48`` Z4: with ``workers`` (and the default compressor) the start of each following level-9
    piece is compressed ahead, for any room; the bytes do not depend on it."""
    ahead: dict = {}
    # ``52`` AC5: nothing is compressed ahead of the packer (the workers only read and inflate the source
    # ahead); the piece starts are compressed where they are packed
    workers = None
    try:
        return _packets_in(pieces, levels, comp_sector, encrypted, source, compress, compressor, emit,
                           keep_plans, workers, ahead)
    finally:
        for future in ahead.values():
            workers.drop(future)


def _plan_ahead(pieces, levels, comp_sector, number, workers, ahead) -> None:
    """Z4: submit the starts of the pieces after ``number`` while there is room in flight."""
    from . import archive as _archive

    for stale in [k for k in ahead if not isinstance(k, int) or k <= number]:
        workers.drop(ahead.pop(stale))
    limit = AHEAD_INPUT_SECTORS * comp_sector
    for following in range(number + 1, len(pieces)):
        if len(ahead) >= workers.limit:
            break
        key = _archive._ahead_key(following)
        data = pieces[following].data
        if key in ahead or levels[following] == 0 or not len(data):
            continue
        size = min(len(data), limit)
        prefix = data.read(0, size) if isinstance(data, StreamView) else bytes(data[:size])
        future = workers.submit(_head_ahead, prefix, size == len(data), comp_sector)
        if future is None:
            break
        ahead[key] = future


def _packets_in(pieces, levels, comp_sector, encrypted, source, compress, compressor, emit, keep_plans,
                workers, ahead) -> _Packed:
    plans: list[PacketPlan] = []
    offsets: list[int] = []
    slot: bytearray | None = None
    slot_count = 0
    slot_open = False
    stream_pos = 0
    kept_total = 0
    last_encrypted = False
    for number, piece in enumerate(pieces):
        data, level = piece.data, levels[number]
        position = 0
        head = None  # Z4: this piece's start, compressed ahead
        if workers is not None:
            future = ahead.pop(number, None)
            head = workers.take(future) if future is not None else None
            _plan_ahead(pieces, levels, comp_sector, number, workers, ahead)
        while position < len(data):
            if not slot_open:
                offsets.append(stream_pos)
                if slot is not None:
                    emit(bytes(slot).ljust(comp_sector, ZERO))
                slot = bytearray()
                slot_count += 1
                slot_open = True
            room = comp_sector - len(slot) - PACKET_HEADER
            left = len(data) - position
            slot_index = slot_count - 1
            if level == 0:
                block, final = stored_block_length(len(data), position)
                stream = stored_stream(data[position:position + block], block, final)
                if len(stream) <= room:
                    stored, declared, cut = stream, min(block, left), False
                else:
                    stored, cut = stream[:room], True
                    declared = min(room - 7, left) if room >= 7 else 0
            else:
                answer = _answer_head(head, room) if head is not None and position == 0 else None
                if answer is not None:
                    stream, complete = answer
                elif compressor is not None:
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
            if keep_plans:
                plans.append(PacketPlan(number, position, slot_index, len(slot), declared,
                                        stored, is_encrypted, kept))
            kept_total += kept
            last_encrypted = is_encrypted
            slot += struct.pack(ORDER + "HHH", declared & 0xFFFF, declared >> 16, word) + written
            position += declared
            stream_pos += declared
            if cut or comp_sector - len(slot) < MIN_PACKET_ROOM:
                slot_open = False
        if isinstance(data, StreamView):
            data.release()
    return _Packed(plans, offsets, slot, kept_total, last_encrypted)


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
    """Write a big-endian version-7 archive by G2-A1 to A9 (the bytes in ``data``; see
    ``write_x360_archive_to``).

    ``pieces`` are in stream order (any order for a new archive). ``levels`` gives each piece's
    level (default: A4). ``encrypted`` is the set of (piece index, position in piece, slot) or
    (piece index, position in piece) whose packets are encrypted. ``source`` may supply packet
    bytes to keep (see ``rebuild_x360``).

    ``compress`` turns a whole rest of a piece into a stream; ``compressor`` makes an incremental
    (``compressobj``-like) compressor used instead, which stops early on cut packets. With the
    default ``compress`` the default ``compressor`` is used; a custom ``compress`` without a
    ``compressor`` is called on the whole rest for every packet."""
    buffer = io.BytesIO()
    result = write_x360_archive_to(
        buffer, pieces, flags=flags, read_sector=read_sector, comp_sector=comp_sector,
        encrypted=encrypted, levels=levels, source=source, compress=compress, compressor=compressor)
    result.data = buffer.getvalue()
    return result


def write_x360_archive_to(
    out,
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
    keep_plans: bool = True,
    jobs=1,
    segment: bool = False,
) -> X360WriteResult:
    """``47`` Y1: ``write_x360_archive`` into ``out`` (a binary file object or an ``Output``);
    ``data`` of the result is None. The region is laid out in a temporary file (the tables before
    it need its slot count and checksum), then header, tables and region are written in order.
    ``48`` Z4: ``jobs`` workers (or an ``archive.Workers``) compress piece starts ahead; the bytes are
    the same for any number. ``51`` AB7: ``segment`` gives the segmented layout (``_segments``)."""
    out = as_output(out)
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

    with _spool(out) as spool, as_workers(jobs) as workers:
        region = _Region(spool)
        if segment and flags & FLAG_COMPRESSED:
            packed = _segments(region, pieces, flags, comp_sector, levels, frozenset(encrypted), source,
                               compress, compressor, workers)
        else:
            packed = _region(region, pieces, flags, comp_sector, levels, frozenset(encrypted), source,
                             compress, compressor, keep_plans, workers)
        _x360_tables(out, region, packed.offsets, flags, read_sector, comp_sector, by_name, sizes,
                     files, entries, locations, stream_size)
        region.copy_to(out)
    found = out.result()
    written = Written(found.size, found.sha256, "segmented" if segment and flags & FLAG_COMPRESSED else "default")
    return X360WriteResult(None, packed.plans, packed.kept, packed.notes, written)


def _pack_x360_segment(pieces, levels, comp_sector, compress, compressor):
    """``51`` AB7: the packets of one segment's pieces as a stream of their own: (offsets from the
    segment's start, the closed slots, the last slot unpadded, whether its last packet is encrypted)."""
    closed: list[bytes] = []
    packed = _packets_in(pieces, levels, comp_sector, frozenset(), None, compress, compressor, closed.append,
                         False, None, {})
    last = bytes(packed.last_slot) if packed.last_slot is not None else None
    return packed.offsets, b"".join(closed), last, packed.last_encrypted


def _segments(region, pieces, flags, comp_sector, levels, encrypted, source, compress, compressor,
              workers) -> _Packed:
    """``51`` AB7, the segmented layout of x360: the pieces are grouped into segments, a new one starting at
    the first piece that begins ``SEGMENT_SIZE`` plain bytes or more after its segment's start, so each
    segment starts a new piece and a new slot (the slot before it ends early, padded with zeros: a
    stored size of 0). Each segment is packed as A3 to A6 pack a stream, by the workers, at most
    ``2 × jobs`` in flight, and written in order. The bytes depend on the layout only."""
    from collections import deque

    from . import archive as _archive

    if encrypted or source is not None:
        raise FormatError("the segmented layout writes no encrypted or kept source packets")
    groups: list[list[int]] = []
    start = position = 0
    for number, piece in enumerate(pieces):
        if not groups or position - start >= _archive.SEGMENT_SIZE:
            groups.append([])
            start = position
        groups[-1].append(number)
        position += len(piece.data)
    limit = workers.limit if workers else 1
    queue: deque = deque()
    offsets: list[int] = []
    last_slot, last_encrypted = None, False
    following = 0
    stream_at = 0
    starts = []
    for group in groups:
        starts.append(stream_at)
        stream_at += sum(len(pieces[n].data) for n in group)
    while following < len(groups) or queue:
        while following < len(groups) and len(queue) < limit:
            numbers = groups[following]
            chunk = []
            for n in numbers:  # read here: a source view is not read by two threads
                data = pieces[n].data
                if isinstance(data, StreamView):
                    plain = data.read(0, len(data))
                    data.release()
                    data = plain
                chunk.append(Piece(pieces[n].name, pieces[n].copy, bytes(data)))
            chunk_levels = [levels[n] for n in numbers]
            future = workers.submit(_pack_x360_segment, chunk, chunk_levels, comp_sector, compress,
                                    compressor) if workers else None
            queue.append((following, future, None if future is not None else (chunk, chunk_levels)))
            following += 1
        index, future, held = queue.popleft()
        if future is not None:
            found, closed, last, encrypted_end = workers.take(future)
        else:
            found, closed, last, encrypted_end = _pack_x360_segment(*held, comp_sector, compress, compressor)
        if last_slot is not None:  # the slot before this segment ends early
            region.write(last_slot.ljust(comp_sector, ZERO))
        offsets.extend(starts[index] + at for at in found)
        region.write(closed)
        if last is not None:
            last_slot, last_encrypted = last, encrypted_end
    packed = _Packed([], offsets, None if last_slot is None else bytearray(last_slot), 0, last_encrypted)
    tail = bytes(last_slot) if last_slot is not None else b""
    if flags & FLAG_COMPACT:  # A7, flag 10
        end = struct.pack(ORDER + "HHH", 0, 0, 0x8000 if last_encrypted else 0)
        if comp_sector - len(tail) >= len(end):
            region.write(tail + end)
        else:
            region.write(tail.ljust(comp_sector, ZERO))
            packed.notes.append("no room for the end header in the last slot; the slot is padded")
    elif last_slot is not None:
        region.write(tail.ljust(comp_sector, ZERO))
    return packed


def _region(region, pieces, flags, comp_sector, levels, encrypted, source, compress, compressor,
            keep_plans, workers=None) -> _Packed:
    """A3 to A7: the sector region into ``region``."""
    if not flags & FLAG_COMPRESSED:
        for piece in pieces:  # A7, flag 0
            if isinstance(piece.data, StreamView):
                piece.data.copy_to(region)
                piece.data.release()
            else:
                region.write(bytes(piece.data))
        return _Packed([], [], None)
    packed = _packets(pieces, levels, comp_sector, encrypted, source, compress, compressor,
                      emit=region.write, keep_plans=keep_plans, workers=workers)
    last_slot = bytes(packed.last_slot) if packed.last_slot is not None else b""
    if flags & FLAG_COMPACT:  # A7, flag 10: an empty header ends the file inside the slot
        end = struct.pack(ORDER + "HHH", 0, 0, 0x8000 if packed.last_encrypted else 0)
        if comp_sector - len(last_slot) >= len(end):
            region.write(last_slot + end)
        else:
            region.write(last_slot.ljust(comp_sector, ZERO))
            packed.notes.append("no room for the end header in the last slot; the slot is padded")
    elif packed.last_slot is not None:  # A7, flag 2
        region.write(last_slot.ljust(comp_sector, ZERO))
    return packed


def _x360_tables(out, region, sector_offsets, flags, read_sector, comp_sector, by_name, sizes,
                 files, entries, locations, stream_size) -> None:
    """A2, A8: header and tables, from the laid-out region."""

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
    crc = region.crc & 0xFFFFFFFF
    header = struct.pack(
        ORDER + "IIQ12I", CPK_MAGIC, 7, stream_size, flags, len(files), len(locations),
        header_sector, size_bits, copies_bits, first_bits, location_bits, location_bits,
        index_bits, crc, read_sector,
    ) + struct.pack(ORDER + "II", comp_sector, 0)
    out.write(header + bytes(tables).ljust(start - HEADER_SIZE, ZERO))


# --- Rebuild from a source archive ---------------------------------------------------------

def source_pieces(archive: CpkArchive, lazy: bool = False) -> list[Piece]:
    """A1: the source's pieces in location (stream) order; a zero-size piece goes before the
    piece it shares its location with. Y1: with ``lazy`` each piece's data is a ``StreamView``
    of the source, read only while it is written."""
    found = []
    for entry in archive.files:
        if entry.name is None:
            raise FormatError(f"file 0x{entry.hash:016X} has no name, cannot rebuild it")
        for copy in range(entry.count):
            found.append((archive.locations[entry.first + copy], entry.size != 0, entry, copy))
    found.sort(key=lambda item: (item[0], item[1]))
    if lazy:
        return [Piece(entry.name, copy, StreamView([file_span(archive, entry, copy)]))
                for _, _, entry, copy in found]
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
