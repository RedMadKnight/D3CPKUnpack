"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.17.2"
STAGE = 17  # description files up to 39_corrections24.md (with 17_x360.md)
