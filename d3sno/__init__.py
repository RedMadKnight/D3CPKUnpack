"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.9.0"
STAGE = 9  # description files up to 19_corrections8.md (with 17_x360.md)
