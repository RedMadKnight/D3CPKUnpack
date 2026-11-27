"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.25.1"
STAGE = 25  # description files up to 61_corrections38.md (with 17_x360.md)
