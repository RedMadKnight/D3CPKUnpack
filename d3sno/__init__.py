"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.15.3"
STAGE = 15  # description files up to 33_corrections20.md (with 17_x360.md)
