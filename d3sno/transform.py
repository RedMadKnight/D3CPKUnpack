"""Transforms of quaternion, translation and uniform scale."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .binary import ByteView

Quaternion = tuple[float, float, float, float]
Vector = tuple[float, float, float]


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Hamilton product, components in x, y, z, w order."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_normalise(q: Quaternion) -> Quaternion:
    """Scale to unit length. A zero quaternion becomes the identity."""
    length = math.sqrt(sum(c * c for c in q))
    if not length or not math.isfinite(length):
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / length, q[1] / length, q[2] / length, q[3] / length)


def rotate(q: Quaternion, v: Vector) -> Vector:
    """Rotate as ``q v q*``."""
    conjugate = (-q[0], -q[1], -q[2], q[3])
    x, y, z, _ = quat_multiply(quat_multiply(q, (v[0], v[1], v[2], 0.0)), conjugate)
    return (x, y, z)


@dataclass(frozen=True)
class Transform:
    q: Quaternion = (0.0, 0.0, 0.0, 1.0)
    t: Vector = (0.0, 0.0, 0.0)
    s: float = 1.0

    def compose(self, local: "Transform") -> "Transform":
        """This transform as the parent of ``local``."""
        moved = rotate(self.q, local.t)
        return Transform(
            quat_multiply(self.q, local.q),
            (
                self.t[0] + self.s * moved[0],
                self.t[1] + self.s * moved[1],
                self.t[2] + self.s * moved[2],
            ),
            self.s * local.s,
        )

    def apply(self, p: Vector) -> Vector:
        moved = rotate(self.q, p)
        return (
            self.t[0] + self.s * moved[0],
            self.t[1] + self.s * moved[1],
            self.t[2] + self.s * moved[2],
        )

    def matrix(self) -> tuple[float, ...]:
        """A 4x4 matrix in column order, as glTF wants it."""
        x, y, z, w = quat_normalise(self.q)
        s = self.s
        return (
            s * (1 - 2 * (y * y + z * z)), s * 2 * (x * y + z * w), s * 2 * (x * z - y * w), 0.0,
            s * 2 * (x * y - z * w), s * (1 - 2 * (x * x + z * z)), s * 2 * (y * z + x * w), 0.0,
            s * 2 * (x * z + y * w), s * 2 * (y * z - x * w), s * (1 - 2 * (x * x + y * y)), 0.0,
            self.t[0], self.t[1], self.t[2], 1.0,
        )


def read_transform(view: ByteView, pos: int) -> Transform:
    values = view.f32s(pos, 8, "transform")
    return Transform(values[:4], values[4:7], values[7])


def transforms_match(a: Transform, b: Transform, tolerance: float = 1e-3) -> bool:
    """Compare within a relative tolerance. A quaternion and its negative are equal."""
    reach = tolerance * (1 + max(abs(c) for c in (*b.t, b.s)))
    if max(abs(p - q) for p, q in zip(a.t, b.t)) > reach or abs(a.s - b.s) > reach:
        return False
    same = max(abs(p - q) for p, q in zip(a.q, b.q))
    opposite = max(abs(p + q) for p, q in zip(a.q, b.q))
    return min(same, opposite) <= tolerance
