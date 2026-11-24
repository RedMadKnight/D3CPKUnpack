"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.21.0"
STAGE = 21  # description files up to 46_stage21_apply.md (with 17_x360.md)
