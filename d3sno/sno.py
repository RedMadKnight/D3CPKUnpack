"""SNO file header, group table, offset pairs and string hashes."""

from __future__ import annotations

from dataclasses import dataclass

from .binary import ByteView, FormatError, Pair, build_order, order_build

__all__ = [
    "Group", "GROUPS", "NONE", "Pair", "SNO_HEADER_SIZE", "SNO_SIGNATURE", "SnoHeader",
    "VERSIONS", "check_version", "compact_handle", "header_problems", "group_by_extension", "group_by_id",
    "RawSnoFile", "hash33", "hash33_ci", "parse_header", "parse_raw", "read_pair", "resolve_build", "slice_pair",
    "split_compact", "X360_VERSIONS", "detect_build", "expected_version",
]

SNO_SIGNATURE = 0xDEADBEEF
SNO_HEADER_SIZE = 0x10
NONE = -1


@dataclass(frozen=True)
class Group:
    id: int
    name: str
    extension: str


_GROUP_TEXT = """
1 Actor acr; 2 Adventure adv; 3 AIBehavior aib; 4 AIState ais; 5 AmbientSound ams;
6 Anim ani; 7 Anim2D an2; 8 AnimSet ans; 9 Appearance app; 10 Hero hro; 11 Cloth clt;
12 Conversation cnv; 13 ConversationList cnl; 14 EffectGroup efg; 15 Encounter enc;
17 Explosion xpl; 18 FlagSet flg; 19 Font fnt; 20 GameBalance gam; 21 Globals glo;
22 LevelArea lvl; 23 Light lit; 24 MarkerSet mrk; 25 Monster mon; 26 Observer obs;
27 Particle prt; 28 Physics phy; 29 Power pow; 31 Quest qst; 32 Rope rop; 33 Scene scn;
34 SceneGroup scg; 35 Script scr; 36 ShaderMap shm; 37 Shaders shd; 38 Shakes shk;
39 SkillKit skl; 40 Sound snd; 41 SoundBank sbk; 42 StringList stl; 43 Surface srf;
44 Textures tex; 45 Trail trl; 46 UI ui; 47 Weather wth; 48 Worlds wrl; 49 Recipe rcp;
51 Condition cnd; 52 TreasureClass trs; 53 Account acc; 54 Conductor con;
55 TimedEvent tme; 56 Act act; 57 Material mat; 58 QuestRange qsr; 59 Lore lor;
60 Reverb rev; 61 PhysMesh phm; 62 Music mus; 63 Tutorial tut; 64 BossEncounter bos;
65 ControlScheme ctr; 66 Accolade aco; 67 AnimTree ant; 68 Vibrations vib;
69 DungeonFinder dfn
"""

GROUPS: tuple[Group, ...] = tuple(
    Group(int(parts[0]), parts[1], parts[2])
    for parts in (item.split() for item in _GROUP_TEXT.replace("\n", " ").split(";"))
)
_BY_ID = {g.id: g for g in GROUPS}
_BY_EXTENSION = {g.extension: g for g in GROUPS}

# File version per type, (ps3, switch).
VERSIONS = {
    "pow": (1442, 1722),
    "app": (295, 306),
    "ani": (123, 129),
    "scn": (163, 179),
    "scr": (86, 94),
    "con": (50, 58),
    "cnv": (88, 100),
    "tex": (58, 66),
    "sbk": (46, 46),
}


def _load_versions() -> None:
    """Add every extension of ``cov_sno_versions.tsv`` to ``VERSIONS``."""
    import csv

    from .datafiles import DATA

    with open(DATA / "cov_sno_versions.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ps3 = int(row["ps3_version"]) if row["ps3_version"] else None
            switch = int(row["switch_version"]) if row["switch_version"] else None
            VERSIONS.setdefault(row["extension"], (ps3, switch))


_load_versions()

# File version per type on x360 (X2); from ``x360_sno_versions.tsv``.
X360_VERSIONS: dict[str, int] = {}


def _load_x360_versions() -> None:
    import csv

    from .datafiles import DATA

    with open(DATA / "x360_sno_versions.tsv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="	"):
            if row["x360_version"]:
                X360_VERSIONS[row["extension"]] = int(row["x360_version"])


_load_x360_versions()


def expected_version(extension: str, build: str) -> int | None:
    """The table's file version of a type on a build; None where the type is unknown there."""
    extension = extension.lstrip(".")
    if build == "x360":
        return X360_VERSIONS.get(extension)
    known = VERSIONS.get(extension)
    if known is None:
        return None
    return known[0 if build == "ps3" else 1]


def header_problems(header: "SnoHeader") -> list[str]:
    """The two runtime words after the handle are zero in every shipped file."""
    problems = []
    if header.runtime_1:
        problems.append(f"word 0x14 is 0x{header.runtime_1:X}")
    if header.runtime_2:
        problems.append(f"word 0x18 is 0x{header.runtime_2:X}")
    return problems


def group_by_id(group_id: int) -> Group | None:
    return _BY_ID.get(group_id)


def group_by_extension(extension: str) -> Group | None:
    return _BY_EXTENSION.get(extension.lstrip(".").lower())


def compact_handle(group: int, handle: int) -> int:
    return (handle << 8) | (group & 0xFF)


def split_compact(value: int) -> tuple[int, int]:
    """Return ``(group, handle)``. A group of -1 comes back as 0xFF."""
    return value & 0xFF, value >> 8


@dataclass
class SnoHeader:
    order: str
    signature: int
    version: int
    revision: int
    source_hash: int
    handle: int
    runtime_1: int
    runtime_2: int
    given_build: str | None = None  # the caller's build, when one was passed

    @property
    def build(self) -> str:
        """The caller's build if given, else the build of the byte order (big-endian: ps3;
        ``resolve_build`` tells x360 from ps3 by the version)."""
        return self.given_build or order_build(self.order)


def parse_header(data: bytes, build: str | None = None) -> SnoHeader:
    """Read the header. Without ``build`` the signature gives the byte order."""
    if len(data) < 0x1C:
        raise FormatError(f"file of {len(data)} bytes is too short for an SNO header", 0)
    if build is not None:
        order = build_order(build)
    else:
        raw = bytes(data[:4])
        if raw == b"\xDE\xAD\xBE\xEF":
            order = ">"
        elif raw == b"\xEF\xBE\xAD\xDE":
            order = "<"
        else:
            raise FormatError("bad SNO signature", 0)
    view = ByteView(data, order)
    words = view.unpack("7I", 0, "SNO header")
    if words[0] != SNO_SIGNATURE:
        raise FormatError("bad SNO signature", 0)
    return SnoHeader(order, *words, given_build=build)


@dataclass
class RawSnoFile:
    """A type with no reader yet: its header, and the whole file as raw bytes."""

    header: SnoHeader
    raw: bytes


def parse_raw(data: bytes, build: str | None = None) -> RawSnoFile:
    return RawSnoFile(parse_header(data, build), bytes(data))


def check_version(extension: str, version: int, build: str, strict: bool = False) -> bool:
    """Compare with the table. A mismatch raises only when ``strict``."""
    extension = extension.lstrip(".")
    if extension not in VERSIONS and extension not in X360_VERSIONS:
        return True
    expected = expected_version(extension, build)
    if expected is None:
        if strict:
            raise FormatError(f".{extension} does not exist on {build}", 4)
        return False
    if version == expected:
        return True
    if strict:
        raise FormatError(f".{extension} version {version}, expected {expected} on {build}", 4)
    return False


def detect_build(
    extension: str, header: SnoHeader, fallback: str | None = None,
) -> str | None:
    """X2: the build of a file from its signature and version, without a caller's build.

    Little-endian is Switch. Big-endian is x360 or ps3 by the version; a version that is the
    same on both (``.aco``, ``.adv`` ...) gives ``fallback`` (the archive's or the caller's build).
    None when nothing decides (no fallback, or a version known to neither big-endian build).
    """
    if header.order == "<":
        return "switch"
    extension = extension.lstrip(".")
    ps3 = VERSIONS.get(extension, (None, None))[0]
    x360 = X360_VERSIONS.get(extension)
    is_ps3, is_x360 = header.version == ps3, header.version == x360
    if is_x360 and not is_ps3:
        return "x360"
    if is_ps3 and not is_x360:
        return "ps3"
    return fallback


def resolve_build(
    extension: str, header: SnoHeader, build: str | None = None, strict: bool = False,
    fallback: str | None = None,
) -> str:
    """Pick the build of a file: the caller's ``build`` if given; otherwise X2 (signature, then
    version; ``fallback`` for a big-endian version shared by ps3 and x360, else ps3)."""
    if build is not None:
        check_version(extension, header.version, build, strict)
        return build
    from_order = header.build
    known = VERSIONS.get(extension, ())
    if from_order == "ps3":
        detected = detect_build(extension, header, fallback)
        if detected is not None:
            return detected
    if len(set(known)) == 2 and None not in known and header.version in known:
        from_version = "ps3" if header.version == known[0] else "switch"
        if from_version != from_order:
            raise FormatError(
                f"version {header.version} is {from_version}, byte order is {from_order}", 4,
            )
    else:
        check_version(extension, header.version, from_order, strict)
    return from_order


def read_pair(data: bytes, pos: int, build: str) -> Pair:
    return ByteView(data, build_order(build)).pair(pos)


def slice_pair(data: bytes, pair: Pair, pos: int = 0) -> bytes:
    """Return the bytes a pair addresses. Byte order does not matter here."""
    return ByteView(data, ">").pair_data(pair, pos)


def _hash(text: bytes | str, fold: bool) -> int:
    raw = text.encode("latin-1") if isinstance(text, str) else bytes(text)
    value = 0
    for byte in raw:
        if fold and 0x41 <= byte <= 0x5A:
            byte += 32
        signed = byte - 256 if byte >= 0x80 else byte
        value = (value * 33 + signed) & 0xFFFFFFFF
    return value


def hash33(text: bytes | str) -> int:
    """Case-sensitive string hash."""
    return _hash(text, False)


def hash33_ci(text: bytes | str) -> int:
    """Case-insensitive string hash."""
    return _hash(text, True)
