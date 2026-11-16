"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.11.2"
STAGE = 11  # description files up to 24_corrections13.md (with 17_x360.md)
