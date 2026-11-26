"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.23.1"
STAGE = 23  # description files up to 56_corrections35.md (with 17_x360.md)
