"""Readers and tools for CPK archives and SNO asset files."""

from .binary import FormatError, Pair

__all__ = ["FormatError", "Pair"]
__version__ = "0.19.0"
STAGE = 19  # description files up to 43_settled_questions.md (with 17_x360.md)
