# Original test assets

Small "programmer art" assets, made from scratch by [`generate.py`](generate.py), so the engine and
CI can be tested without any game data. Nothing here is read, copied or derived from game files or
executables. Only the sources are committed; the console files are built from them by the
d3sno writers, in the tests and on demand.

## Sources (`sources/`)

| File | Content |
|---|---|
| `checker_64.png` | 64 x 64 RGBA checker: magenta and dark grey cells of 8 px, a white 4 px grid band every 32 px. Every 4 x 4 block is one colour, and the colours are exact in RGB565, so DXT1 keeps the top level lossless |
| `box.obj` | A 2 x 2 x 2 cube centred on the origin, z up: 8 positions, 4 UVs, 6 normals, 12 triangles wound outward |
| `box_collision.obj` | The same cube as positions and triangles only, for the collision builder |
| `material.json` | One material: ambient, diffuse, specular and emissive colours, power, sort value, no shader map (-1), the checker texture in the diffuse slot |
| `strings.tsv` | A five-row text table (label, text), UTF-8, one empty text and one with Polish letters |
| `anim_box_bob.json` | A one-bone animation of 31 key frames at 30 fps: the bone rises 0.5 and turns 90° about z |

The sources are written by `python -I -B test_assets/generate.py sources`, and
`python -I -B test_assets/generate.py check` reports any that differ from what the generator
makes (PNG files are compared by pixels, as zlib output may differ between zlib builds).

## Console files

`python -I -B test_assets/generate.py build OUT_DIR` writes, for each build (ps3, switch, x360),
the SNO files at their archive paths, a `CoreTOC.dat` that lists them and a `test_assets.cpk`
holding them (v6 big-endian for ps3, v7 little-endian for Switch, v7 big-endian for x360). Use
a folder outside the repository, or `test_assets/out/`, which is ignored. `*.cpk` is ignored
everywhere.

| Source | File | ps3 | switch | x360 | Writer |
|---|---|:-:|:-:|:-:|---|
| `checker_64.png` | `Textures\TA_Checker64.tex` | V | V | V | `texture_write.encode_texture`, DXT1 (format 9), full mip chain |
| `material.json` | `Material\TA_CheckerMaterial.mat` | V | V | V | `material_write.new_material` |
| `box_collision.obj` | `PhysMesh\TA_BoxCollision.phm` | V | V | V | `collision_write.build_collision_mesh` |
| `anim_box_bob.json` | `Anim\TA_BoxBob.ani` | X | V | X | `anim_write.encode_animation`, `write_ani` |
| `strings.tsv` | `.stl` | X | X | X | none (see below) |
| `box.obj` | `.app` | X | X | X | none (see below) |
| all of the above | `CoreTOC.dat` | V | V | V | `index_write.write_core_toc` |
| all of the above | `test_assets.cpk` | V | V | V | `archive.write_archive` |

The SNO handles are fixed (`HANDLES` in `generate.py`, 0x7A550001 to 0x7A550004) so every build
is reproducible; they are our own choice. The bone of the animation is named by `hash33` of our
own name `TA_Root`; which hash an engine expects for track names is not asserted here.

## Writers d3sno does not have yet

These are not implemented here: d3sno is developed in a clean-room process elsewhere, and this
folder only uses it.

- **`.stl` from scratch.** `write.write_stl` writes a `StringList` that was read from a file: it
  keeps the file's fixed 0x38-byte head (`StringList.fixed`). There is no writer that makes that
  head for a new list (README "Supported formats": `.stl` Create is X). `strings.tsv` is kept as
  the source; the tests check its labels and that their `hash33` values are unique.
- **`.app` from scratch.** Mesh import works only onto an existing appearance (`.app` Create is
  X), so `box.obj` has no console build yet.
- **`.ani` for ps3 and x360.** `anim_write.write_ani` creates Switch files only. On ps3 and x360
  the curve data lives in the `.can` child; `children.write_can` writes the child, but there is no
  writer for a new parent `.ani` record table (`update_ani_records` changes an existing parent
  with a fixed permutation count).
- **x360 texture decoding.** The x360 `.tex` is built and round-trips byte for byte, but d3sno
  does not decode its pixel layout, so only ps3 and Switch are compared with the source pixels.
- **x360 index.** The x360 build keeps its index in six TOC files (`index_write.write_x360_tocs`);
  the single `CoreTOC.dat` written here is enough for the material's link check, not a full x360
  file set.

## Tests

[`tests_d3sno/test_original_assets.py`](../tests_d3sno/test_original_assets.py), run by CI with the
rest of `tests_d3sno`:

- the committed sources equal the generator's output, and the generator is deterministic;
- every build is built twice with equal bytes;
- every file is read back with d3sno (strict mode where the reader has one: `.tex`, `.phm`,
  `.ani`, `.cpk`; `.mat` is checked as a canonical tree with no stray bytes) and written back
  byte for byte; the `.phm` also rebuilds unchanged with every value derived;
- the texture decodes to the source pixels (ps3, Switch), the animation decodes to the source
  keys, the material's texture link resolves in the built `CoreTOC.dat`, and the archive holds
  exactly the built files and rebuilds byte for byte.

```bash
python -I -B -m unittest discover -s tests_d3sno -p "test_original_assets.py" -v
```
