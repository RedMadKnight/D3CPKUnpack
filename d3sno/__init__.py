"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.10.1"
STAGE = 10  # description files up to 21_corrections10.md (with 17_x360.md); version 0.10.1
