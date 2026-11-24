"""Stage 14 marker: ``28_derived_values.md`` (W14-1 to W14-23). The rules are tested in
test_stage14_index, _texture, _collision, _navigation and _misc."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import unittest

import d3sno
from d3sno import collision_write, derived, index_write, navigation_derive, sno_layout, texture_write


class Marker(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(x) for x in d3sno.__version__.split(".")), (0, 14, 0))
        self.assertGreaterEqual(d3sno.STAGE, 14)

    def test_api(self):
        for module, names in (
                (index_write, ("new_handle", "sno_identity", "write_core_toc", "add_toc_entry", "write_x360_tocs",
                               "write_prefetch", "add_prefetch_parent", "write_manifest", "write_sprite_table",
                               "write_descriptor_sizes", "add_file", "replace_file")),
                (texture_write, ("encode_texture",)),
                (collision_write, ("build_collision_mesh", "write_collision")),
                (navigation_derive, ("derive_navigation", "cells_from_grid")),
                (derived, ("special_indices", "update_bank_row")),
                (sno_layout, ("write_tree", "set_elements"))):
            for name in names:
                self.assertTrue(callable(getattr(module, name)), f"{module.__name__}.{name}")


if __name__ == "__main__":
    unittest.main()
