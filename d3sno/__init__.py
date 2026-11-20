"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.16.2"
STAGE = 16  # description files up to 36_corrections22.md (with 17_x360.md)
