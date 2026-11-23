"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.18.2"
STAGE = 18  # description files up to 42_corrections26.md (with 17_x360.md)
