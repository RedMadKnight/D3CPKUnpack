"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.22.4"
STAGE = 22  # description files up to 53_corrections33.md (with 17_x360.md)
