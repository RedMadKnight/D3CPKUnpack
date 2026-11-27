"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.24.1"
STAGE = 24  # description files up to 59_corrections37.md (with 17_x360.md)
