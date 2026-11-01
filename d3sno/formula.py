"""Formula bytecode: decoder, disassembler and evaluator with an injected context."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Mapping

from .binary import FormatError, build_order, to_f32
from .power import (
    TAG_FLOAT, TAG_FORMULA, TAG_INT, PowerFile, TagRecord, formula_tag,
)

STACK_LIMIT = 64
RECURSION_LIMIT = 32
EPSILON = 1e-6
NO_PARAMETER = 0xFFFFFFFE
CURRENT_POWER = 0xFFFFFFFF
_DIMINISHING_LN4 = to_f32(1.3862944)
_DIMINISHING_LOG4_7 = to_f32(1.4036775)
_RANDOM_SCALE = to_f32(2.0 ** -32)

OP_NAMES = {
    0: "END", 1: "CALL", 2: "OP2", 3: "OP3", 4: "PUSHID1", 5: "PUSHID", 6: "PUSHF", 7: "LT",
    8: "GT", 9: "EQ", 10: "NE", 11: "ADD", 12: "SUB", 13: "MUL", 14: "DIV", 15: "OP15",
    16: "NEG", 17: "SELECT",
}
_OPERAND_WORDS = {1: 1, 4: 1, 5: 4, 6: 1}

FUNCTIONS = {
    0: ("Min", 2), 1: ("Max", 2), 2: ("Pin", 3), 3: ("RangeInt", 2), 4: ("RandomInt", 2),
    5: ("Floor", 1), 6: ("Diminishing", 3), 7: ("Pow", 2), 8: ("Log", 1),
    9: ("RangeFloat", 2), 10: ("RandomFloat", 2), 11: ("Table", 2),
}

CLASS_SKILL_LEVEL = 1
CLASS_ITEM_LEVEL = 2
CLASS_PREVIOUS_SKILL_LEVEL = 4
CLASS_MONSTER_DAMAGE_TYPE = 5
CLASS_OTHER_POWER = 22
CLASS_FIRST_SLOT = 23
_FIRST_TABLE = {"ps3": 63, "switch": 87}
_LAST_CLASS_OFFSET = 26  # class 113 on switch
_CLASS_NAMES = {
    0: "Attribute", 1: "SkillLevel", 2: "ItemLevel", 3: "GizmoOrMonsterLevel",
    4: "PreviousSkillLevel", 5: "MonsterDamageType", 22: "OtherPowerTag",
}
_DAMAGE_TYPES = ("Physical", "Fire", "Lightning", "Cold", "Poison", "Arcane")
_NAMED_TABLES = {
    **{i: f"DmgTier{i + 1}" for i in range(7)},
    7: "Healing", 8: "WDCost", 9: "RuneDamageBonus", 14: "HealthMin", 15: "LegendaryProcDmg",
    16: "OffhandDmgMinMin", 24: "AreaDamageMin", 25: "AreaDamageDelta", 26: "TraitActive",
}
# Formula table row per class offset. 14 is health minimum, 24 to 26 are context values.
_CLASS_ROWS = {**{i: i for i in range(14)}, **{i: i - 1 for i in range(15, 24)}}


def first_table_class(build: str) -> int:
    return _FIRST_TABLE[build]


def table_row_of_class(cls: int, build: str) -> int | None:
    """The formula table row an identifier class is, or None."""
    return _CLASS_ROWS.get(cls - _FIRST_TABLE[build])


def class_name(cls: int, build: str) -> str:
    """Readable name of an identifier class. Build-specific above class 22."""
    if cls in _CLASS_NAMES:
        return _CLASS_NAMES[cls]
    if 6 <= cls <= 21:
        index, part = divmod(cls - 6, 2)
        kind = _DAMAGE_TYPES[index] if index < len(_DAMAGE_TYPES) else f"Type{index}"
        return f"MonsterDamage{kind}{'Delta' if part else 'Min'}"
    first = _FIRST_TABLE[build]
    if cls < first:
        return f"ScriptSlot{cls - CLASS_FIRST_SLOT}"
    index = cls - first
    return _NAMED_TABLES.get(index, f"Table{index}")


def slot_of_class(cls: int, build: str) -> int | None:
    """Return n when ``cls`` is class 23 + n on this build."""
    if CLASS_FIRST_SLOT <= cls < _FIRST_TABLE[build]:
        return cls - CLASS_FIRST_SLOT
    return None


@dataclass(frozen=True)
class Instruction:
    offset: int
    opcode: int
    name: str
    operands: tuple
    truncated: bool = False


def decode(code: bytes, build: str) -> list[Instruction]:
    """Decode up to END. Only the first byte of an opcode word is read.

    Nothing past END is looked at. An unknown opcode, or an operand running past the code,
    ends the list, as it would end a run.
    """
    order = build_order(build)
    result = []
    pos = 0
    while pos + 4 <= len(code):
        opcode = code[pos]
        name = OP_NAMES.get(opcode, f"OP{opcode}")
        words = _OPERAND_WORDS.get(opcode, 0)
        args_at = pos + 4
        if args_at + words * 4 > len(code):
            result.append(Instruction(pos, opcode, name, (), True))
            break
        if opcode == 1:
            operands = (code[args_at],)
        elif opcode == 6:
            operands = struct.unpack_from(order + "f", code, args_at)
        elif words:
            operands = struct.unpack_from(f"{order}{words}I", code, args_at)
        else:
            operands = ()
        result.append(Instruction(pos, opcode, name, operands))
        if opcode == 0 or opcode not in OP_NAMES:
            break
        pos = args_at + words * 4
    return result


def _signed(word: int) -> int:
    return word - 0x100000000 if word >= 0x80000000 else word


def disassemble(code: bytes, build: str) -> str:
    lines = []
    for ins in decode(code, build):
        text = ins.name
        if ins.truncated:
            text += " <operands run past the code>"
        elif ins.opcode == 1:
            fn = FUNCTIONS.get(ins.operands[0])
            text += f" {fn[0] if fn else 'fn#' + str(ins.operands[0])}"
        elif ins.opcode == 5:
            cls, a, b, c = ins.operands
            text += (
                f" class={cls} a={_signed(a)} b={_signed(b)} c={_signed(c)}"
                f" ; {class_name(cls, build)}"
            )
        elif ins.opcode == 4:
            text += f" class={ins.operands[0]} ; {class_name(ins.operands[0], build)}"
        elif ins.opcode == 6:
            text += f" {ins.operands[0]!r}"
        lines.append(f"{ins.offset:04X}  {text}")
    return "\n".join(lines)


class EvalContext:
    """Everything the bytecode cannot compute by itself. Override what a formula needs.

    ``random_mode`` is None, or one of ``maximum`` and ``minimum`` (no draw) and
    ``draw_maximum`` and ``draw_minimum`` (one word drawn first).
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.random_mode: str | None = None
        self.pvp = False

    def identifier(self, cls: int, a: int, b: int, c: int) -> float:
        raise FormatError(f"no value for identifier class {cls} a={a} b={b} c={c}")

    def power(self, handle: int) -> PowerFile | None:
        """Return another power for class 22, or None to use ``identifier``."""
        return None

    def random_word(self) -> int:
        """One 32-bit random word."""
        raise FormatError("context has no random source")

    def table(self, a: float, b: float) -> float:
        raise FormatError("context has no Table")

    def tag_default(self, tag: int) -> float:
        """The value of a tag missing from its map. Script slots default to 0."""
        return 0.0

    def nested(self, power: PowerFile) -> "EvalContext":
        """The context of a nested run. Skill, item and trait state should not carry over."""
        return self


@dataclass
class ValueContext(EvalContext):
    """A context backed by a dictionary of ``(class, a, b, c)`` or plain class keys.

    A nested run drops the plain keys for skill and item levels, and reads the skill level
    from the key ``("skill", power handle)``.
    """

    values: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__()

    def identifier(self, cls: int, a: int, b: int, c: int) -> float:
        for key in ((cls, a, b, c), cls):
            if key in self.values:
                return float(self.values[key])
        return super().identifier(cls, a, b, c)

    def nested(self, power: PowerFile) -> "ValueContext":
        values = {
            key: value for key, value in self.values.items()
            if key not in (CLASS_SKILL_LEVEL, CLASS_ITEM_LEVEL, CLASS_PREVIOUS_SKILL_LEVEL)
        }
        skill = self.values.get(("skill", power.header.handle))
        if skill is not None:
            values[CLASS_SKILL_LEVEL] = skill
        inner = type(self).__new__(type(self))
        inner.__dict__.update(self.__dict__)
        inner.values = values
        return inner


class _Failed(Exception):
    """Ends a run: the formula is logged and its value is 0."""

    def __init__(self, message: str, offset: int):
        super().__init__(message)
        self.message = message
        self.offset = offset


class _Fatal(FormatError):
    """Opcode 15, which no formula survives."""


class _Run:
    __slots__ = ("items", "ctx", "strict")

    def __init__(self, ctx: EvalContext, strict: bool) -> None:
        self.items: list[float] = []
        self.ctx = ctx
        self.strict = strict

    def note(self, message: str, pos: int) -> None:
        """Log a problem that does not end the run."""
        if self.strict:
            raise FormatError(message, pos)
        self.ctx.warnings.append(f"{message} (offset 0x{pos:X})")

    def push(self, value: float, pos: int) -> None:
        if len(self.items) >= STACK_LIMIT:
            self.note("formula stack overflow, value dropped", pos)
            return
        self.items.append(to_f32(value))

    def pop(self, pos: int) -> float:
        if not self.items:
            self.note("formula stack underflow, 0 used", pos)
            return 0.0
        return self.items.pop()


def _floor(x: float) -> float:
    return float(math.floor(x)) if math.isfinite(x) else x


def _pow(a: float, b: float) -> float:
    try:
        return math.pow(a, b)
    except OverflowError:
        return math.inf
    except ValueError:
        return math.nan if a < 0 else math.inf


def _log(x: float) -> float:
    if x > 0:
        return math.log(x)
    return -math.inf if x == 0 else math.nan


def _trunc(x: float) -> int:
    """Truncate to a wrapping 32-bit integer."""
    if not math.isfinite(x):
        return -0x80000000
    return ((int(x) + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def _diminishing(a: float, b: float, c: float) -> float:
    s = to_f32(a + 6)
    if s < EPSILON:
        return 0.0
    ratio = to_f32(to_f32(_log(s)) / _DIMINISHING_LN4)
    return b + to_f32(c - b) * to_f32(ratio - _DIMINISHING_LOG4_7)


def _draw_unit(ctx: EvalContext) -> float:
    return to_f32(to_f32(ctx.random_word() & 0xFFFFFFFF) * _RANDOM_SCALE)


def _forced(ctx: EvalContext, minimum: float, maximum: float) -> float | None:
    """Apply the context's random flags. None means make a normal draw."""
    mode = ctx.random_mode
    if mode is None:
        return None
    if mode.startswith("draw_"):
        ctx.random_word()
    return maximum if mode.endswith("maximum") else minimum


def _range_int(low: int, span: int, ctx: EvalContext) -> float:
    forced = _forced(ctx, float(low), float(low + span))
    if forced is not None:
        return forced
    word = ctx.random_word() & 0xFFFFFFFF
    modulus = (span + 1) & 0xFFFFFFFF
    # Computed in 32 bits and converted as unsigned, so a negative sum becomes huge.
    return float((low + (word % modulus if modulus else word)) & 0xFFFFFFFF)


def _range_float(a: float, b: float, ctx: EvalContext) -> float:
    forced = _forced(ctx, a, to_f32(a + b))
    return forced if forced is not None else a + b * _draw_unit(ctx)


def _call(fn: int, args: list[float], ctx: EvalContext, run: _Run, pos: int) -> float:
    if fn == 0:
        return args[0] if args[0] < args[1] else args[1]
    if fn == 1:
        return args[0] if args[0] > args[1] else args[1]
    if fn == 2:
        x, low, high = args
        if x < low:
            return low
        return high if x > high else x
    if fn == 3:
        return _range_int(_trunc(args[0]), _trunc(args[1]), ctx)
    if fn == 4:
        if args[1] < args[0]:
            run.note("RandomInt with hi below lo", pos)
            return args[0]
        low = _trunc(args[0])
        return _range_int(low, _trunc(args[1]) - low, ctx)
    if fn == 5:
        return _floor(args[0])
    if fn == 6:
        return _diminishing(*args)
    if fn == 7:
        return _pow(*args)
    if fn == 8:
        return _log(args[0])
    if fn == 9:
        return _range_float(args[0], args[1], ctx)
    if fn == 10:
        if args[1] < args[0]:
            run.note("RandomFloat with hi below lo", pos)
            return args[0]
        return _range_float(args[0], to_f32(args[1] - args[0]), ctx)
    return ctx.table(args[0], args[1])


def _find_record(power: PowerFile, tag: int, ctx: EvalContext) -> TagRecord | None:
    """General, else the first contact map. In a PvP game the PvP map wins if it has the tag."""
    if ctx.pvp:
        for name in ("PvpGeneral", "PvpContact0"):
            record = power.maps[name].get(tag)
            if record is not None:
                return record
    return power.general.get(tag) or power.maps["Contact0"].get(tag)


def _nested_value(
    record: TagRecord | None, tag: int, power: PowerFile, ctx: EvalContext, depth: int,
    strict: bool,
) -> float:
    if record is None:
        return ctx.tag_default(tag)
    if record.type == TAG_INT or record.type == TAG_FLOAT:
        return float(record.value)
    if record.type == TAG_FORMULA:
        return _run(record.value.code, power.build, ctx.nested(power), power, depth, strict)
    ctx.warnings.append(f"tag 0x{record.tag:X} has type {record.type}, value taken as 0")
    return 0.0


def _identifier(
    cls: int, a: int, b: int, c: int, build: str, ctx: EvalContext, power: PowerFile | None,
    depth: int, strict: bool, run: _Run, pos: int,
) -> float:
    if cls == CLASS_MONSTER_DAMAGE_TYPE or cls > _FIRST_TABLE[build] + _LAST_CLASS_OFFSET:
        raise _Failed(f"identifier class {cls} is not handled", pos)
    row = table_row_of_class(cls, build)
    if row is not None:
        return float(row)
    target = None
    tag = 0
    if cls == CLASS_OTHER_POWER:
        target = power if a == CURRENT_POWER else ctx.power(a)
        tag = b
    else:
        slot = slot_of_class(cls, build)
        if slot is not None and power is not None:
            target = power
            tag = formula_tag(slot, build)
    if target is None:
        try:
            return ctx.identifier(cls, a, b, c)
        except FormatError as exc:
            raise _Failed(exc.message, pos) from None
    if depth >= RECURSION_LIMIT:
        run.note(f"formula recursion reached {RECURSION_LIMIT}, read gives 0", pos)
        return 0.0
    return _nested_value(_find_record(target, tag, ctx), tag, target, ctx, depth + 1, strict)


def _execute(
    code: bytes, build: str, ctx: EvalContext, power: PowerFile | None, depth: int, strict: bool,
) -> float:
    order = build_order(build)
    run = _Run(ctx, strict)
    size = len(code)
    pos = 0
    while True:
        if pos + 4 > size:
            raise _Failed("formula ran off the end without END", pos)
        op = code[pos]
        at = pos
        pos += 4 + _OPERAND_WORDS.get(op, 0) * 4
        if pos > size and op in _OPERAND_WORDS:
            raise _Failed("operand runs past the code", at)
        if op == 0:
            if not run.items:
                raise _Failed("END on an empty stack", at)
            return run.items.pop()
        if op == 15:
            raise _Fatal("opcode 15 is fatal", at)
        if op in (2, 3) or op > 17:
            raise _Failed(f"opcode {op} is invalid", at)
        if op == 1:
            fn = code[at + 4]
            if fn not in FUNCTIONS:
                run.note(f"unknown function {fn}", at)
                run.push(0.0, at)
                continue
            args = [run.pop(at) for _ in range(FUNCTIONS[fn][1])]
            args.reverse()
            try:
                value = _call(fn, args, ctx, run, at)
            except FormatError as exc:
                raise _Failed(exc.message, at) from None
            run.push(value, at)
        elif op == 4:
            cls = struct.unpack_from(order + "I", code, at + 4)[0]
            run.push(_identifier(cls, 0, 0, 0, build, ctx, power, depth, strict, run, at), at)
        elif op == 5:
            cls, a, b, c = struct.unpack_from(order + "4I", code, at + 4)
            run.push(_identifier(cls, a, b, c, build, ctx, power, depth, strict, run, at), at)
        elif op == 6:
            run.push(struct.unpack_from(order + "f", code, at + 4)[0], at)
        elif op == 16:
            run.push(-run.pop(at), at)
        elif op == 17:
            when_false = run.pop(at)
            when_true = run.pop(at)
            cond = run.pop(at)
            run.push(when_true if cond > EPSILON else when_false, at)
        else:
            b_value = run.pop(at)
            a_value = run.pop(at)
            if op == 7:
                value = float(a_value < b_value)
            elif op == 8:
                value = float(a_value > b_value)
            elif op == 9:
                value = float(abs(a_value - b_value) < EPSILON)
            elif op == 10:
                value = float(abs(a_value - b_value) >= EPSILON)
            elif op == 11:
                value = a_value + b_value
            elif op == 12:
                value = a_value - b_value
            elif op == 13:
                value = a_value * b_value
            else:
                value = a_value / b_value if b_value != 0 else 0.0
            run.push(value, at)


def _run(
    code: bytes, build: str, ctx: EvalContext, power: PowerFile | None, depth: int, strict: bool,
) -> float:
    """Run one formula. A failure is logged and gives 0; the caller carries on."""
    try:
        return _execute(code, build, ctx, power, depth, strict)
    except _Failed as exc:
        if strict:
            raise FormatError(exc.message, exc.offset) from None
        ctx.warnings.append(f"{exc.message} (offset 0x{exc.offset:X})")
        return 0.0


def evaluate(
    code: bytes, build: str, ctx: EvalContext | None = None, power: PowerFile | None = None,
    strict: bool = False,
) -> float:
    """Run code. ``power`` lets identifiers reach its slots and class 22 with a = -1.

    A failed formula gives 0 and a warning in ``ctx.warnings``. Only opcode 15 raises, unless
    ``strict``, which raises on every logged problem.
    """
    return _run(code, build, ctx or EvalContext(), power, 0, strict)


def evaluate_tag(
    power: PowerFile, tag: int, ctx: EvalContext | None = None, strict: bool = False,
) -> float:
    ctx = ctx or EvalContext()
    record = _find_record(power, tag, ctx)
    if record is None:
        return ctx.tag_default(tag)
    if record.type == TAG_FORMULA:
        return _run(record.value.code, power.build, ctx, power, 1, strict)
    return _nested_value(record, tag, power, ctx, 1, strict)


def evaluate_slot(
    power: PowerFile, slot: int, ctx: EvalContext | None = None, strict: bool = False,
) -> float:
    """Value of ``formula(slot)``. A missing tag gives 0, and so does a slot out of range."""
    ctx = ctx or EvalContext()
    try:
        tag = formula_tag(slot, power.build)
    except FormatError as exc:
        if strict:
            raise
        ctx.warnings.append(exc.message)
        return 0.0
    return evaluate_tag(power, tag, ctx, strict)
