"""Formula writing (stage 15, W15-A1 to A7): record frame, assembler, compiler and decompiler.

``formula.py`` reads and runs code; this module makes it.

- ``formula_record`` builds a type-4 tag-map record value (W15-A1).
- ``disassemble`` / ``assemble`` are an exact-bits text form of code (W15-A2). Any byte string
  round-trips: words that are not a canonical instruction are written as ``WORD 0x...`` and a
  tail shorter than a word as ``BYTES <hex>``.
- ``compile_formula`` turns formula text into code (W15-A3 tokens, W15-A4 grammar, W15-A5
  literals, W15-A6 names).
- ``decompile`` turns code back into fully parenthesised text (W15-A7).

Library extension, not a game feature: the compiler understands two escape names, which
``decompile`` writes where no game name or decimal token reaches the stored operand.

- ``__f32_XXXXXXXX``: a constant with these float32 bits (hex). It counts as a constant, so a
  formula of constants only is still folded (W15-A4).
- ``__id_C_A_B_D``: a PUSHID with these four words (hex).

Pass ``escapes=False`` to ``compile_formula`` to resolve such names by W15-A6 like any other.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Callable, Mapping

from .binary import FormatError, build_order, to_f32
from .formula import (
    CLASS_OTHER_POWER, FUNCTIONS, OP_NAMES, _DAMAGE_TYPES, _LAST_CLASS_OFFSET, class_name,
    evaluate, first_table_class,
)

POWER_GROUP = 29  # sno_group POWER_SNO in the CoreTOC
NO_PARAMETER = 0xFFFFFFFE  # b = -2
OTHER_PARAMETER = 0xFFFFFFFF  # b = -1
_MASK = 0xFFFFFFFF

OP_END, OP_CALL, OP_PUSHID1, OP_PUSHID, OP_PUSHF = 0, 1, 4, 5, 6
OP_LT, OP_GT, OP_EQ, OP_NE = 7, 8, 9, 10
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG, OP_SELECT = 11, 12, 13, 14, 16, 17
_MNEMONICS = {name: op for op, name in OP_NAMES.items()}
_FUNCTION_IDS = {name.lower(): fn for fn, (name, _) in FUNCTIONS.items()}

# Context names of W15-A6 rule 5 that the library's class names spell differently.
_CONTEXT_NAMES = {"mhealthmin": 14, "mareadamagemin": 24, "mareadamagedelta": 25,
                  "btraitactive": 26}
_LEVEL_NAMES = {"slevel": 1, "ilevel": 2, "mlevel": 3}
_LEVEL_SPELLING = {1: "sLevel", 2: "iLevel", 3: "mLevel"}


def _u32(value: int) -> int:
    return value & _MASK


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", to_f32(value)))[0]


def _bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & _MASK))[0]


# --- W15-A1: record frame ------------------------------------------------------------------

def formula_record(text: str, code: bytes, build: str) -> bytes:
    """A type-4 record value: eight u32 words, the text with its NUL, zero padding, the code.

    Words 0-4 and 6 are zero, word 5 is the text length with its NUL, word 7 the code length.
    Empty code means no compiled formula.
    """
    order = build_order(build)
    if len(code) % 4:
        raise FormatError(f"code length {len(code)} is not a multiple of 4")
    if "\0" in text:
        raise FormatError("formula text holds a NUL")
    raw = text.encode("latin-1") + b"\0"
    head = struct.pack(order + "8I", 0, 0, 0, 0, 0, len(raw), 0, len(code))
    return head + raw + bytes(-len(raw) % 4) + bytes(code)


# --- W15-A2: assembler and disassembler ----------------------------------------------------

def _opcode_word(op: int) -> bytes:
    return bytes((op, 0, 0, 0))


def _signed(word: int) -> int:
    return word - 0x100000000 if word >= 0x80000000 else word


def _float_comment(bits: int) -> str:
    return repr(_bits_f32(bits))


def disassemble(code: bytes, build: str) -> str:
    """One line per instruction, operands by their exact words. ``assemble`` reverses it.

    Every word is written, END and what follows it included. A word that is not a canonical
    instruction (non-zero bytes after the opcode, an unknown opcode, a CALL operand with
    non-zero bytes after the function id, operands running past the code) is a ``WORD`` line.
    """
    order = build_order(build)
    lines = []
    pos = 0
    size = len(code)
    while pos + 4 <= size:
        word = code[pos:pos + 4]
        op = word[0]
        operand_words = {OP_CALL: 1, OP_PUSHID1: 1, OP_PUSHID: 4, OP_PUSHF: 1}.get(op, 0)
        args_at = pos + 4
        canonical = (
            op in OP_NAMES and not any(word[1:]) and args_at + operand_words * 4 <= size
        )
        if canonical and op == OP_CALL and any(code[args_at + 1:args_at + 4]):
            canonical = False
        if not canonical:
            lines.append(f"WORD 0x{struct.unpack_from(order + 'I', code, pos)[0]:08X}")
            pos += 4
            continue
        name = OP_NAMES[op]
        if op == OP_CALL:
            fn = code[args_at]
            known = FUNCTIONS.get(fn)
            lines.append(f"CALL {known[0] if known else fn}")
        elif op == OP_PUSHID:
            cls, a, b, c = struct.unpack_from(order + "4I", code, args_at)
            lines.append(
                f"PUSHID {cls} {_signed(a)} {_signed(b)} {_signed(c)} ; {class_name(cls, build)}"
            )
        elif op == OP_PUSHID1:
            (cls,) = struct.unpack_from(order + "I", code, args_at)
            lines.append(f"PUSHID1 {cls} ; {class_name(cls, build)}")
        elif op == OP_PUSHF:
            (bits,) = struct.unpack_from(order + "I", code, args_at)
            lines.append(f"PUSHF 0x{bits:08X} ; {_float_comment(bits)}")
        else:
            lines.append(name)
        pos = args_at + operand_words * 4
    if pos < size:
        lines.append(f"BYTES {code[pos:].hex().upper()}")
    return "\n".join(lines)


def _int(token: str, line: int) -> int:
    try:
        return int(token, 0)
    except ValueError:
        raise FormatError(f"line {line}: {token!r} is not an integer") from None


def _float_operand(token: str, line: int) -> int:
    """PUSHF operand: ``0x`` bits, else a decimal float rounded to float32."""
    if token.lower().startswith(("0x", "-0x", "+0x")):
        return _u32(_int(token, line))
    try:
        return _f32_bits(float(token))
    except ValueError:
        raise FormatError(f"line {line}: {token!r} is not a float") from None


def assemble(text: str, build: str) -> bytes:
    """Code from ``disassemble`` text. Mnemonics are case-insensitive; ``;`` starts a comment.

    PUSHID words may be signed or unsigned, decimal or hex. PUSHF takes exact bits (``0x``)
    or a decimal float. CALL takes a function name or id.
    """
    order = build_order(build)
    out = bytearray()
    for number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        mnemonic, args = parts[0].upper(), parts[1:]

        def need(count: int) -> None:
            if len(args) != count:
                raise FormatError(f"line {number}: {mnemonic} takes {count} operand(s)")

        if mnemonic == "WORD":
            need(1)
            out += struct.pack(order + "I", _u32(_int(args[0], number)))
        elif mnemonic == "BYTES":
            need(1)
            try:
                out += bytes.fromhex(args[0])
            except ValueError:
                raise FormatError(f"line {number}: bad hex {args[0]!r}") from None
        elif mnemonic == "CALL":
            need(1)
            fn = _FUNCTION_IDS.get(args[0].lower())
            if fn is None:
                fn = _int(args[0], number)
            if not 0 <= fn <= 0xFF:
                raise FormatError(f"line {number}: function id {fn} is not a byte")
            out += _opcode_word(OP_CALL) + bytes((fn, 0, 0, 0))
        elif mnemonic == "PUSHID":
            need(4)
            words = [_u32(_int(a, number)) for a in args]
            out += _opcode_word(OP_PUSHID) + struct.pack(order + "4I", *words)
        elif mnemonic == "PUSHID1":
            need(1)
            out += _opcode_word(OP_PUSHID1) + struct.pack(order + "I", _u32(_int(args[0], number)))
        elif mnemonic == "PUSHF":
            need(1)
            out += _opcode_word(OP_PUSHF) + struct.pack(order + "I", _float_operand(args[0], number))
        elif mnemonic in _MNEMONICS:
            need(0)
            out += _opcode_word(_MNEMONICS[mnemonic])
        else:
            raise FormatError(f"line {number}: unknown mnemonic {parts[0]!r}")
    return bytes(out)


# --- W15-A5: literals ----------------------------------------------------------------------

def formula_literal(token: str) -> float:
    """The float32 value of a number token, rounded to float32 after every operation.

    v = v * 10 + d for each digit before the point; then div = div * 10, v = v + d / div for
    each digit after it.
    """
    whole, _, fraction = token.partition(".")
    if not (whole + fraction).isdigit() or not (whole + fraction).isascii():
        raise FormatError(f"{token!r} is not a number token")
    value = 0.0
    for ch in whole:
        value = to_f32(to_f32(value * 10.0) + (ord(ch) - 48))
    div = 1.0
    for ch in fraction:
        div = to_f32(div * 10.0)
        value = to_f32(value + to_f32((ord(ch) - 48) / div))
    return value


def literal_bits(token: str) -> int:
    return _f32_bits(formula_literal(token))


def _plain(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def literal_token(bits: int) -> str | None:
    """The shortest decimal token whose W15-A5 value has these bits, or None.

    Only non-negative finite values have a token (a token has no sign).
    """
    bits &= _MASK
    value = _bits_f32(bits)
    if bits >> 31 or value != value or value in (float("inf"),):
        return None
    if bits == 0:
        return "0"
    exact = Decimal(value)
    tried = set()
    with localcontext() as ctx:
        ctx.prec = 80
        for digits in range(1, 18):
            exponent = exact.adjusted() - digits + 1
            quantum = Decimal(1).scaleb(exponent)
            base = exact.quantize(quantum, rounding=ROUND_HALF_EVEN)
            for step in (0, -1, 1, -2, 2):
                candidate = base + step * quantum
                if candidate < 0:
                    continue
                token = _plain(candidate)
                if token in tried or len(token) > 60:
                    continue
                tried.add(token)
                if literal_bits(token) == bits:
                    return token
    return None


# --- W15-A6: names -------------------------------------------------------------------------

def _lower_map(table: Mapping[str, int] | None) -> dict[str, int]:
    found: dict[str, int] = {}
    for name, value in (table or {}).items():
        found.setdefault(name.lower(), _u32(int(value)))
    return found


def _reverse_map(table: Mapping[str, int] | None) -> dict[int, str]:
    found: dict[int, str] = {}
    for name, value in (table or {}).items():
        found.setdefault(_u32(int(value)), name)
    return found


def power_handles(toc) -> dict[str, int]:
    """Power name to handle from a ``CoreToc`` (power group 29)."""
    return {e.name: e.handle for e in toc.entries if e.group == POWER_GROUP}


def library_power_tags() -> dict[str, int]:
    """Power tag display names of the library tag table (General before tuning tags)."""
    from .tags import tag_table

    found: dict[str, int] = {}
    for table in ("power_general", "power_tuning"):
        for (name_table, tag), info in tag_table().items():
            if name_table == table and info.name:
                found.setdefault(info.name, tag)
    return found


def attribute_table(build: str) -> dict[str, int]:
    """Attribute name to id from the library tables: x360 its own ids, Switch the Switch ids
    of the x360 rows, ps3 the PROVEN and SUPPORTED ps3 rows. Incomplete; pass a full table
    from the game where one is known."""
    from .records import SAFE_STATUSES, ps3_attributes, x360_attributes

    found: dict[str, int] = {}
    if build == "ps3":
        for row in ps3_attributes().values():
            if row.name and row.status in SAFE_STATUSES:
                found.setdefault(row.name, row.ps3_id)
    elif build == "x360":
        for row in x360_attributes().values():
            found.setdefault(row.name, row.id)
    else:
        for row in x360_attributes().values():
            if row.switch_id is not None:
                found.setdefault(row.name, row.switch_id)
    return found


_SCRIPT_FORMULA = re.compile(r"script formula (\d+)$")
_SLOT_NAME = re.compile(r"sf_(\d+)$")
_DAMAGE_NAME = re.compile(r"mdamage(min|delta)#(.*)$")
_ESCAPE_F32 = re.compile(r"__f32_([0-9a-f]{1,8})$")
_ESCAPE_ID = re.compile(r"__id_([0-9a-f]{1,8})_([0-9a-f]{1,8})_([0-9a-f]{1,8})_([0-9a-f]{1,8})$")
_DAMAGE_INDEX = {name.lower(): i for i, name in enumerate(_DAMAGE_TYPES)}


def _table_names(build: str) -> dict[str, int]:
    """Rule 5 names to class offsets: the library class names, then the context names."""
    first = first_table_class(build)
    found = {}
    for offset in range(_LAST_CLASS_OFFSET + 1):
        found.setdefault(class_name(first + offset, build).lower(), offset)
    for name, offset in _CONTEXT_NAMES.items():
        found.setdefault(name, offset)
    return found


@dataclass
class FormulaNames:
    """The tables of W15-A6.

    ``attributes`` maps attribute names to ids, ``powers`` power names to handles (or pass a
    ``CoreToc``), ``power_tags`` power tag display names to tag ids (None: the library tag
    names), ``extra`` more power name to handle pairs (powers that left the build's CoreTOC).
    ``unknown_power_tag`` chooses what a ``PowerTag`` name with an unknown power or tag gives:
    ``"fallthrough"`` (later rules, normally rule 7) or ``"error"``.
    """

    attributes: Mapping[str, int] | None = None
    powers: object = None
    power_tags: Mapping[str, int] | None = None
    extra: Mapping[str, int] | None = None
    unknown_power_tag: str = "fallthrough"
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def _tables(self) -> dict:
        if not self._cache:
            powers = self.powers
            if powers is not None and hasattr(powers, "entries"):
                powers = power_handles(powers)
            tags = library_power_tags() if self.power_tags is None else self.power_tags
            self._cache.update(
                attributes=_lower_map(self.attributes), powers=_lower_map(powers),
                extra=_lower_map(self.extra), tags=_lower_map(tags),
                attribute_names=_reverse_map(self.attributes), power_names=_reverse_map(powers),
                extra_names=_reverse_map(self.extra), tag_names=_reverse_map(tags),
            )
        return self._cache

    def power(self, name: str) -> int | None:
        tables = self._tables()
        key = name.lower()
        found = tables["powers"].get(key)
        return found if found is not None else tables["extra"].get(key)

    def power_tag(self, name: str) -> int | None:
        key = name.lower()
        found = self._tables()["tags"].get(key)
        if found is not None:
            return found
        match = _SCRIPT_FORMULA.match(key)
        if match:
            n = int(match.group(1))
            return 0x41100 + (n // 10) * 0x100 + (n % 10) * 0x10
        return None

    def resolve(self, name: str, build: str) -> tuple[int, int, int, int]:
        """The PUSHID operands (four u32 words) of a name, by W15-A6."""
        key = name.lower()
        # 1. PowerTag.<power>."<tag name>"
        if key.startswith("powertag."):
            rest = name[len("powertag."):]
            split = rest.find('."')
            if split >= 0:
                power_name = rest[:split]
                tag_name = rest[split + 2:]
                if tag_name.endswith('"'):
                    tag_name = tag_name[:-1]
                handle = self.power(power_name)
                tag = self.power_tag(tag_name)
                if handle is not None and tag is not None:
                    return (CLASS_OTHER_POWER, handle, tag, 0)
                if self.unknown_power_tag == "error":
                    raise FormatError(f"{name!r}: unknown power or power tag")
        # 2. levels
        if key in _LEVEL_NAMES:
            return (_LEVEL_NAMES[key], 0, 0, 0)
        # 3. SF_n
        match = _SLOT_NAME.match(key)
        if match:
            return (23 + int(match.group(1)), 0, 0, 0)
        # 4. mDamageMin#T, mDamageDelta#T
        match = _DAMAGE_NAME.match(key)
        if match and match.group(2) in _DAMAGE_INDEX:
            index = _DAMAGE_INDEX[match.group(2)]
            return (6 + 2 * index + (match.group(1) == "delta"), 0, 0, 0)
        # 5. table and context names
        tables = _table_names(build)
        if key in tables:
            return (first_table_class(build) + tables[key], 0, 0, 0)
        # 6. attributes
        base, hash_mark, parameter = key.partition("#")
        attribute = self._tables()["attributes"].get(base)
        if attribute is not None:
            if not hash_mark:
                b = NO_PARAMETER
            elif parameter in _DAMAGE_INDEX:
                b = _DAMAGE_INDEX[parameter]
            else:
                handle = self.power(parameter)
                b = OTHER_PARAMETER if handle is None else handle
            return (0, attribute, b, 0)
        # 7. anything else
        return (0, 0, NO_PARAMETER, 0)

    def spell(self, operands: tuple[int, int, int, int], build: str) -> str | None:
        """A name that ``resolve`` maps back to these operands, or None."""
        cls, a, b, c = (_u32(x) for x in operands)
        tables = self._tables()
        first = first_table_class(build)
        candidates: list[str] = []
        if cls == CLASS_OTHER_POWER and c == 0:
            power = tables["power_names"].get(a) or tables["extra_names"].get(a)
            tag = tables["tag_names"].get(b)
            if tag is None and b >= 0x41100 and (b - 0x41100) % 0x10 == 0:
                hundreds, rest = divmod(b - 0x41100, 0x100)
                if rest // 0x10 < 10:
                    tag = f"Script Formula {hundreds * 10 + rest // 0x10}"
            if power is not None and tag is not None and '"' not in tag:
                candidates.append(f'PowerTag.{power}."{tag}"')
        elif (a, b, c) == (0, 0, 0):
            if cls in _LEVEL_SPELLING:
                candidates.append(_LEVEL_SPELLING[cls])
            elif 6 <= cls <= 21:
                index, part = divmod(cls - 6, 2)
                candidates.append(f"mDamage{'Delta' if part else 'Min'}#{_DAMAGE_TYPES[index]}")
            elif 23 <= cls < first:
                candidates.append(f"SF_{cls - 23}")
            elif first <= cls <= first + _LAST_CLASS_OFFSET:
                offset = cls - first
                spelled = {v: k for k, v in _CONTEXT_NAMES.items()}
                if offset in spelled:
                    candidates.append({"mhealthmin": "mHealthMin", "mareadamagemin": "mAreaDamageMin",
                                       "mareadamagedelta": "mAreaDamageDelta",
                                       "btraitactive": "bTraitActive"}[spelled[offset]])
                candidates.append(class_name(cls, build))
        if cls == 0 and c == 0:
            attribute = tables["attribute_names"].get(a)
            if attribute is not None and "#" not in attribute:
                if b == NO_PARAMETER:
                    candidates.append(attribute)
                elif b < len(_DAMAGE_TYPES):
                    candidates.append(f"{attribute}#{_DAMAGE_TYPES[b]}")
                if b != NO_PARAMETER:
                    power = tables["power_names"].get(b) or tables["extra_names"].get(b)
                    if power is not None:
                        candidates.append(f"{attribute}#{power}")
                if b == OTHER_PARAMETER:
                    candidates.append(f"{attribute}#None")
        for candidate in candidates:
            if _is_name_token(candidate) and self.resolve(candidate, build) == (cls, a, b, c):
                return candidate
        return None


def _is_name_token(text: str) -> bool:
    tokens = _tokenize(text, "switch")
    ps3 = _tokenize(text, "ps3")
    return tokens == ps3 == [("name", text)]


def resolve(
    name: str, build: str, attributes: Mapping[str, int] | None = None, powers=None,
    power_tags: Mapping[str, int] | None = None, extra: Mapping[str, int] | None = None,
) -> tuple[int, int, int, int]:
    """The PUSHID operands (class, a, b, c) of a formula name on a build, as u32 words.

    ``powers`` is a power name to handle mapping or a ``CoreToc``; ``power_tags`` None uses
    the library's power tag names; ``extra`` is a further power name to handle table.
    """
    return FormulaNames(attributes, powers, power_tags, extra).resolve(name, build)


# --- W15-A3: tokens ------------------------------------------------------------------------

def _digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _letter(ch: str) -> bool:
    return "a" <= ch <= "z" or "A" <= ch <= "Z"


_OPERATORS = "+-*/?:(),<>"
_TWO_CHAR = ("==", "!=")


def _tokenize(text: str, build: str) -> list[tuple[str, str]]:
    """Tokens as (kind, text), kind ``num``, ``name`` or ``op``. Unknown characters are skipped."""
    tokens = []
    i = 0
    n = len(text)
    dot_number = build != "ps3"
    while i < n:
        ch = text[i]
        if ch in " \t\n\r\v\f":
            i += 1
        elif _digit(ch) or (dot_number and ch == "." and i + 1 < n and _digit(text[i + 1])):
            j = i
            while j < n and _digit(text[j]):
                j += 1
            if j < n and text[j] == ".":
                j += 1
                while j < n and _digit(text[j]):
                    j += 1
            tokens.append(("num", text[i:j]))
            i = j
        elif _letter(ch) or ch in "_.":
            j = i + 1
            while j < n:
                c = text[j]
                if _letter(c) or _digit(c) or c in "_.#":
                    j += 1
                elif c == '"':
                    close = text.find('"', j + 1)
                    j = n if close < 0 else close + 1
                else:
                    break
            tokens.append(("name", text[i:j]))
            i = j
        elif text[i:i + 2] in _TWO_CHAR:
            tokens.append(("op", text[i:i + 2]))
            i += 2
        elif ch in _OPERATORS:
            tokens.append(("op", ch))
            i += 1
        else:
            i += 1
    return tokens


# --- W15-A4: grammar and emission ----------------------------------------------------------

_COMPARISONS = {"<": OP_LT, ">": OP_GT, "==": OP_EQ, "!=": OP_NE}


class _Parser:
    def __init__(self, tokens, build: str, names, escapes: bool, check_arity: bool):
        self.tokens = tokens
        self.pos = 0
        self.build = build
        self.names = names
        self.escapes = escapes
        self.check_arity = check_arity
        self.out: list[tuple] = []  # ('f', bits) ('id', ops) ('call', fn) ('op', opcode)
        self.has_name = False

    def peek(self, value: str | None = None):
        if self.pos >= len(self.tokens):
            return None
        token = self.tokens[self.pos]
        if value is not None:
            return token if token == ("op", value) else None
        return token

    def take(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def comparison(self) -> None:
        self.additive()
        while True:
            token = self.peek()
            if token and token[0] == "op" and token[1] in _COMPARISONS:
                self.take()
                self.additive()
                self.out.append(("op", _COMPARISONS[token[1]]))
            else:
                return

    def additive(self) -> None:
        self.ternary()
        while True:
            token = self.peek()
            if token in (("op", "+"), ("op", "-")):
                self.take()
                self.ternary()
                self.out.append(("op", OP_ADD if token[1] == "+" else OP_SUB))
            else:
                return

    def ternary(self) -> None:
        self.multiplicative()
        while self.peek("?"):
            self.take()
            self.multiplicative()
            if not self.peek(":"):
                raise FormatError(f"':' expected at token {self.pos}")
            self.take()
            self.multiplicative()
            self.out.append(("op", OP_SELECT))

    def multiplicative(self) -> None:
        self.unary()
        while True:
            token = self.peek()
            if token in (("op", "*"), ("op", "/")):
                self.take()
                self.unary()
                self.out.append(("op", OP_MUL if token[1] == "*" else OP_DIV))
            else:
                return

    def unary(self) -> None:
        token = self.peek()
        if token == ("op", "-"):
            self.take()
            self.unary()
            self.out.append(("op", OP_NEG))
        elif token == ("op", "+"):
            self.take()
            self.unary()
        else:
            self.primary()

    def close(self) -> None:
        """A ``)``, or the end of the text (a missing final ``)`` is accepted)."""
        if self.peek(")"):
            self.take()
        elif self.peek() is not None:
            raise FormatError(f"')' expected at token {self.pos}")

    def primary(self) -> None:
        token = self.peek()
        if token is None:
            raise FormatError("expression expected at the end of the text")
        kind, text = token
        if kind == "num":
            self.take()
            self.out.append(("f", literal_bits(text)))
        elif kind == "name":
            self.take()
            if self.peek("("):
                self.call(text)
            else:
                self.name(text)
        elif token == ("op", "("):
            self.take()
            self.comparison()
            self.close()
        else:
            raise FormatError(f"unexpected {text!r} at token {self.pos}")

    def call(self, name: str) -> None:
        fn = _FUNCTION_IDS.get(name.lower())
        if fn is None:
            raise FormatError(f"unknown function {name!r}")
        self.take()  # (
        count = 0
        if self.peek(")"):
            self.take()
        else:
            self.comparison()
            count = 1
            while self.peek(","):
                self.take()
                self.comparison()
                count += 1
            self.close()
        if self.check_arity and count != FUNCTIONS[fn][1]:
            raise FormatError(f"{FUNCTIONS[fn][0]} takes {FUNCTIONS[fn][1]} arguments, got {count}")
        self.out.append(("call", fn))
        self.has_name = True

    def name(self, text: str) -> None:
        if self.escapes:
            key = text.lower()
            match = _ESCAPE_F32.match(key)
            if match:
                self.out.append(("f", int(match.group(1), 16)))
                return
            match = _ESCAPE_ID.match(key)
            if match:
                self.out.append(("id", tuple(int(g, 16) for g in match.groups())))
                self.has_name = True
                return
        self.out.append(("id", tuple(_u32(x) for x in self.names(text, self.build))))
        self.has_name = True


def _emit(items, build: str) -> bytes:
    order = build_order(build)
    out = bytearray()
    for kind, value in items:
        if kind == "f":
            out += _opcode_word(OP_PUSHF) + struct.pack(order + "I", value)
        elif kind == "id":
            out += _opcode_word(OP_PUSHID) + struct.pack(order + "4I", *value)
        elif kind == "call":
            out += _opcode_word(OP_CALL) + bytes((value, 0, 0, 0))
        else:
            out += _opcode_word(value)
    return bytes(out + _opcode_word(OP_END))


def _name_resolver(names, build: str) -> Callable[[str, str], tuple]:
    if names is None:
        return FormulaNames().resolve
    if isinstance(names, FormulaNames):
        return names.resolve
    if callable(names):
        return names
    raise FormatError("names must be a FormulaNames, a callable or None")


def compile_formula(
    text: str, build: str, names=None, *, escapes: bool = True, check_arity: bool = False,
) -> bytes:
    """Formula text to code (W15-A3 to A6). Empty or blank text gives empty code.

    ``names`` is a ``FormulaNames``, a callable ``(name, build) -> (class, a, b, c)``, or None
    (no attribute or power tables). Constant-only text folds to one PUSHF.
    """
    build_order(build)
    tokens = _tokenize(text, build)
    if not tokens:
        return b""
    parser = _Parser(tokens, build, _name_resolver(names, build), escapes, check_arity)
    while parser.pos < len(tokens):
        parser.comparison()
    if parser.has_name:
        return _emit(parser.out, build)
    if len(parser.out) == 1:
        return _emit(parser.out, build)
    value = evaluate(_emit(parser.out, build), build)
    return _emit([("f", _f32_bits(value))], build)


# --- W15-A7: decompile ---------------------------------------------------------------------

_BINARY_TEXT = {OP_LT: "<", OP_GT: ">", OP_EQ: "==", OP_NE: "!=", OP_ADD: "+", OP_SUB: "-",
                OP_MUL: "*", OP_DIV: "/"}


def _constant_text(bits: int, alone: bool) -> str:
    token = literal_token(bits)
    if token is not None:
        return token
    if alone and bits >> 31:
        # A lone constant folds: (-x) gives the bits when x has a token.
        positive = literal_token(bits & 0x7FFFFFFF)
        if positive is not None:
            return f"(-{positive})"
    return f"__f32_{bits:08X}"


def decompile(code: bytes, build: str, names: FormulaNames | None = None) -> str:
    """Fully parenthesised text that ``compile_formula`` turns back into ``code``.

    Names are spelled by W15-A6 from ``names``; operands no name reaches become ``__id_``
    escapes, constants no decimal token reaches ``__f32_`` escapes. Code that leaves two
    values gives two parenthesised expressions. Empty code gives empty text.
    """
    order = build_order(build)
    names = names or FormulaNames()
    if not code:
        return ""
    stack: list[str] = []
    pos = 0
    size = len(code)
    lone = size == 12 and code[0] == OP_PUSHF and code[8] == OP_END
    while True:
        if pos + 4 > size:
            raise FormatError("code ends without END", pos)
        word = code[pos:pos + 4]
        op = word[0]
        if any(word[1:]):
            raise FormatError("opcode word has non-zero bytes", pos)
        at = pos
        pos += 4
        if op == OP_END:
            break
        if op == OP_PUSHF:
            if pos + 4 > size:
                raise FormatError("operand runs past the code", at)
            (bits,) = struct.unpack_from(order + "I", code, pos)
            pos += 4
            stack.append(_constant_text(bits, lone))
        elif op == OP_PUSHID:
            if pos + 16 > size:
                raise FormatError("operand runs past the code", at)
            operands = struct.unpack_from(order + "4I", code, pos)
            pos += 16
            spelled = names.spell(operands, build)
            stack.append(spelled or "__id_{:X}_{:X}_{:X}_{:X}".format(*operands))
        elif op == OP_CALL:
            if pos + 4 > size:
                raise FormatError("operand runs past the code", at)
            fn = code[pos]
            if any(code[pos + 1:pos + 4]) or fn not in FUNCTIONS:
                raise FormatError(f"function {fn} cannot be written as text", at)
            pos += 4
            name, arity = FUNCTIONS[fn]
            if len(stack) < arity:
                raise FormatError("stack underflow", at)
            args = stack[len(stack) - arity:]
            del stack[len(stack) - arity:]
            stack.append(f"{name}({', '.join(args)})")
        elif op == OP_NEG:
            if not stack:
                raise FormatError("stack underflow", at)
            stack.append(f"(-{stack.pop()})")
        elif op == OP_SELECT:
            if len(stack) < 3:
                raise FormatError("stack underflow", at)
            when_false, when_true, cond = stack.pop(), stack.pop(), stack.pop()
            stack.append(f"({cond} ? {when_true} : {when_false})")
        elif op in _BINARY_TEXT:
            if len(stack) < 2:
                raise FormatError("stack underflow", at)
            right, left = stack.pop(), stack.pop()
            stack.append(f"({left} {_BINARY_TEXT[op]} {right})")
        else:
            raise FormatError(f"opcode {op} cannot be written as text", at)
    if pos != size:
        raise FormatError("bytes follow END", pos)
    if not stack:
        raise FormatError("END on an empty stack", pos - 4)
    if len(stack) == 1:
        return stack[0]
    return " ".join(item if item.startswith("(") else f"({item})" for item in stack)


__all__ = [
    "FormulaNames", "assemble", "attribute_table", "compile_formula", "decompile",
    "disassemble", "formula_literal", "formula_record", "library_power_tags", "literal_bits",
    "literal_token", "power_handles", "resolve",
]
