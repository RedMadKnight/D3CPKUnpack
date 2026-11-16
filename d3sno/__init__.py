"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.12.1"
STAGE = 12  # description files up to 25_corrections14.md (with 17_x360.md)
