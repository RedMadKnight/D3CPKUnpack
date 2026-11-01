"""``.stl`` string lists and ``declension.txt``."""

from __future__ import annotations

from dataclasses import dataclass, field

from .binary import ByteView, FormatError, Pair
from .sno import SnoHeader, hash33, parse_header

ENTRY_SIZE = 24
ENTRIES_AT = 0x38
CONVERSATION_CLASSES = ("", "DHunter", "Barb", "Wiz", "WitchDoc", "Monk", "Crusader", "Necromancer")
GENDERS = ("", "M", "F")
PS3_LOCALE_PREFIXES = ("enUS_", "esMX_", "frFR_")


@dataclass(frozen=True)
class StringEntry:
    label_hash: int
    text: str
    slot_raw: bytes = bytes(8)
    zero: int = 0


@dataclass
class StringList:
    header: SnoHeader
    build: str
    entries: list[StringEntry]
    by_hash: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    fixed: bytes = b""


def parse_string_list(data: bytes, build: str | None = None, strict: bool = False) -> StringList:
    """Read a ``.stl``. Entries keep file order, which is not sorted."""
    header = parse_header(data, build)
    view = ByteView(data, header.order)
    pair = view.pair(0x28, "entry array pair")
    raw = view.pair_data(pair, 0x28, "entry array")
    if len(raw) % ENTRY_SIZE:
        raise FormatError(f"entry array of {len(raw)} bytes is not a multiple of 24", 0x28)
    result = StringList(header, header.build, [], fixed=bytes(data[:ENTRIES_AT]))
    for i in range(len(raw) // ENTRY_SIZE):
        at = pair.file_offset + i * ENTRY_SIZE
        text_pair = view.pair(at + 8, "text pair")
        text_raw = view.pair_data(text_pair, at + 8, "text")
        if not text_raw or text_raw[-1] != 0:
            raise FormatError("text lacks its NUL", at + 8)
        try:
            text = text_raw[:-1].decode("utf-8")
        except UnicodeDecodeError:
            raise FormatError("text is not UTF-8", text_pair.file_offset) from None
        label_hash = view.u32(at + 16)
        result.entries.append(StringEntry(
            label_hash, text, view.take(at, 8), view.u32(at + 20),
        ))
        if label_hash in result.by_hash:
            result.warnings.append(f"label hash 0x{label_hash:08X} repeats")
        else:
            result.by_hash[label_hash] = text
    if strict and result.warnings:
        raise FormatError(result.warnings[0], pair.file_offset)
    return result


def lookup(strings: StringList, label: str, override: StringList | None = None) -> str | None:
    """Find a label. A run-time ``override`` list is searched first. There is no fallback."""
    label_hash = hash33(label)
    if override is not None and label_hash in override.by_hash:
        return override.by_hash[label_hash]
    return strings.by_hash.get(label_hash)


def lookup_power(strings: StringList, power: str, part: str = "name") -> str | None:
    """``part`` is ``name``, ``desc`` or ``simple_desc``."""
    return lookup(strings, f"{power}_{part}")


def gender_suffix(gender) -> str:
    """Gender 0 is ``M``, 1 is ``F``, anything else is empty."""
    return {0: "M", 1: "F"}.get(gender, "") if isinstance(gender, int) else gender


def line_labels(
    uid: int, line_class: str = "", line_gender="", player_class: str = "", player_gender="",
    player_speaks: bool = False,
) -> list[str]:
    """The labels of a conversation line, in the order they are tried."""
    line_gender = gender_suffix(line_gender)
    player_gender = gender_suffix(player_gender)
    if player_speaks:
        first = (line_class or player_class) + (line_gender or player_gender)
    else:
        first = line_class + (line_gender or ("F" if player_gender == "F" else ""))
    suffixes = (first, line_class + line_gender, player_class, player_gender, "")
    return [f"Line{uid}{suffix}" for suffix in suffixes]


def lookup_line(
    strings: StringList, uid: int, line_class: str = "", line_gender="", player_class: str = "",
    player_gender="", player_speaks: bool = False,
) -> str | None:
    """A conversation line. A miss is normal, not an error."""
    for label in line_labels(
        uid, line_class, line_gender, player_class, player_gender, player_speaks,
    ):
        text = lookup(strings, label)
        if text is not None:
            return text
    return None


def locale_of(name: str) -> str | None:
    """The ``|xxYY`` suffix of a switch archive name, without the bar."""
    return name.rpartition("|")[2] if "|" in name else None


def locale_of_archive(archive_name: str) -> str | None:
    """The locale prefix of a ps3 archive name."""
    for prefix in PS3_LOCALE_PREFIXES:
        if archive_name.startswith(prefix):
            return prefix[:-1]
    return None


def join_locales(lists: dict[str, StringList]) -> dict[int, dict[str, str]]:
    """Text per label hash, then per locale."""
    joined: dict[int, dict[str, str]] = {}
    for locale, strings in lists.items():
        for label_hash, text in strings.by_hash.items():
            joined.setdefault(label_hash, {})[locale] = text
    return joined


def parse_declension(data: bytes) -> dict[tuple[str, int], str]:
    """``declension.txt``: form per (word, case index). Case meanings are open."""
    raw = bytes(data)
    start = 3 if raw[:3] == b"\xEF\xBB\xBF" else 0
    try:
        text = raw[start:].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormatError("declension file is not UTF-8", exc.start + start) from None
    forms: dict[tuple[str, int], str] = {}
    for number, line in enumerate(text.split("\n")):
        line = line.rstrip("\r")
        if not line:
            continue
        columns = line.split("\t")
        if len(columns) != 3:
            raise FormatError(f"line {number + 1} has {len(columns)} columns", number)
        try:
            forms[(columns[0], int(columns[1]))] = columns[2]
        except ValueError:
            raise FormatError(f"line {number + 1}: bad case index", number) from None
    return forms
