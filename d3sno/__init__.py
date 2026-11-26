"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.23.2"
STAGE = 23  # description files up to 57_corrections36.md (with 17_x360.md)
