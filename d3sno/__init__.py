"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.13.2"
STAGE = 13  # description files up to 27_corrections16.md (with 17_x360.md)
