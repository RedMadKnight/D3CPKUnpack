"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.14.2"
STAGE = 14  # description files up to 29_corrections17.md (with 17_x360.md)
