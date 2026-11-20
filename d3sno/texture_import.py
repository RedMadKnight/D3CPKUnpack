"""Texture import onto a template ``.tex`` (``34_modding_tools.md`` W16-2 to W16-4).

``import_texture(template, images, *, levels=None, format=None)`` builds a ``.tex`` the way
``texture_write.encode_texture`` does, keeping what did not change:

- W16-2: build, format (unless given), faces, depth, frames, image ids and the other header
  values come from the template. The level count is ``levels``, else the given chain's length,
  else the template's count when W14-8 allows it for the new size, else W14-8's full chain.
- W16-3 rule 1: a level the caller did not give is the template's decoded level when the given
  top level equals the template's decoded top level (per face) and the size is unchanged;
  otherwise it is made with ``downsample``.
- W16-3 rule 2: with the template's size and format, each element (DXT block or texel) of a level
  the template has is the template's stored element when decoding it gives exactly the image's
  pixels there; otherwise it is encoded from the image.
- When format, size, level count, faces and depth are all the template's, the merged elements
  are written into a copy of the template's payload at their stored addresses and the template's
  header is written back as stored (``write.write_tex``). Otherwise the payload is built by
  W14-9 to W14-11 and the header by ``assemble_texture`` with the template.

``split_strip(image, count)`` (W16-4) cuts a vertical strip into ``count`` equal images.
"""

from __future__ import annotations

import dataclasses

from . import texture as T
from . import texture_write as W
from . import x360_texture as XT
from .binary import FormatError
from .texture import FORMATS, Image, TextureFile


# --- W16-4 ----------------------------------------------------------------------------------

def split_strip(image: Image, count: int) -> list[Image]:
    """Cut a vertical strip (as ``texture.vertical_strip`` writes it) into ``count`` images.

    A strip whose height is not ``count`` times its width is refused."""
    _check_image(image, "strip")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"strip count {count!r}: a positive integer is needed")
    if image.height != count * image.width:
        raise FormatError(
            f"strip of {image.width}x{image.height} is not {count} square images "
            f"(height {count * image.width} needed)")
    size = image.width * image.width * 4
    return [Image(image.width, image.width, image.rgba[i * size:(i + 1) * size])
            for i in range(count)]


# --- Helpers --------------------------------------------------------------------------------

def _check_image(image, what: str) -> None:
    if not isinstance(image, Image):
        raise ValueError(f"{what}: expected a texture.Image, got {type(image).__name__}")
    if image.width < 1 or image.height < 1 or len(image.rgba) != image.width * image.height * 4:
        raise ValueError(f"{what}: {len(image.rgba)} bytes of RGBA for {image.width}x{image.height}")


def decode_template(tex: TextureFile, level: int = 0, face: int = 0, z: int = 0) -> Image:
    """The template's decoded surface, as the exporter writes it (x360 with its decoder)."""
    if tex.build == "x360":
        return XT.decode_image(tex, level, face, z)
    return T.decode_image(tex, level, face, z)


def decode_form(kind: str, build: str, raw: bytes, width: int, height: int) -> bytes:
    """Decode elements in a build's element form (``texture_write.element_form``) to RGBA."""
    if build == "x360":
        if kind.startswith("dxt"):
            return T._decode_elements(kind, "switch", raw, width, height)
        if kind == "argb1555":  # standard form is little-endian A1 R5 G5 B5
            return T._decode_elements(kind, "ps3", XT.swap16(raw), width, height)
        return T._decode_elements(kind, "ps3", raw, width, height)
    return T._decode_elements(kind, build, raw, width, height)


def merge_elements(tex_format: int, build: str, stored: bytes, image: Image) -> bytes:
    """W16-3 rule 2 for one surface: per element, the stored element when it decodes to exactly
    the image's pixels there, else the element encoded from the image. ``stored`` and the
    result are in the build's element form, row order."""
    kind, bpp = FORMATS[tex_format]
    width, height = image.width, image.height
    old = decode_form(kind, build, stored, width, height)
    if old == image.rgba:
        return bytes(stored)
    fresh = W.element_form(image, tex_format, build)
    if len(fresh) != len(stored):
        raise FormatError(f"{len(stored)} bytes of stored elements, {len(fresh)} expected")
    out = bytearray(fresh)
    new = image.rgba
    if kind in W.DXT_KINDS:
        wb = (width + 3) // 4
        for by in range((height + 3) // 4):
            for bx in range(wb):
                same = True
                left, right = bx * 16, min(bx * 4 + 4, width) * 4
                for y in range(by * 4, min(by * 4 + 4, height)):
                    row = y * width * 4
                    if old[row + left:row + right] != new[row + left:row + right]:
                        same = False
                        break
                if same:
                    at = (by * wb + bx) * bpp
                    out[at:at + bpp] = stored[at:at + bpp]
    else:
        for i in range(width * height):
            if old[i * 4:i * 4 + 4] == new[i * 4:i * 4 + 4]:
                out[i * bpp:(i + 1) * bpp] = stored[i * bpp:(i + 1) * bpp]
    return bytes(out)


def _stored_offsets(tex: TextureFile, level: int, face: int, z: int = 0):
    if tex.build == "x360":
        return XT.element_offsets(tex, level, face, z)
    return T.element_offsets(tex, level, face, z)


def _patch_payload(tex: TextureFile, elements: dict[tuple[int, int], bytes]) -> bytes:
    """Write element-form surfaces into a copy of the template's payload at stored addresses."""
    kind, bpp = FORMATS[tex.format]
    swap = tex.build == "x360" and kind in XT.SWAPPED_KINDS
    payload = bytearray(tex.payload)

    def put(offsets, raw: bytes) -> None:
        if swap:
            raw = XT.swap16(raw)
        if isinstance(offsets, range):
            payload[offsets.start:offsets.stop] = raw
            return
        for i, at in enumerate(offsets):
            payload[at:at + bpp] = raw[i * bpp:(i + 1) * bpp]

    if tex.depth != 1:
        raw = elements[(0, 0)]
        size = len(raw) // tex.depth
        for z in range(tex.depth):
            put(_stored_offsets(tex, 0, 0, z), raw[z * size:(z + 1) * size])
    else:
        for (level, face), raw in elements.items():
            put(_stored_offsets(tex, level, face), raw)
    return bytes(payload)


def _chains(images, faces: int, depth: int) -> list[list[Image]]:
    """Normalise ``images`` to one chain per face (or per volume slice)."""
    if isinstance(images, Image):
        images = [images]
    if not isinstance(images, (list, tuple)):
        raise ValueError(f"images: expected an Image or a list, got {type(images).__name__}")
    images = list(images)
    if faces == 1 and depth == 1:
        if len(images) == 1 and isinstance(images[0], (list, tuple)):
            images = list(images[0])
        chains = [images]
    else:
        wanted = depth if depth != 1 else faces
        what = "slices" if depth != 1 else "faces"
        if len(images) != wanted:
            raise ValueError(f"{len(images)} images for a template with {wanted} {what}")
        chains = [list(s) if isinstance(s, (list, tuple)) else [s] for s in images]
        if depth != 1 and any(len(c) != 1 for c in chains):
            raise ValueError("a volume slice is one image (a volume has one level)")
    for f, chain in enumerate(chains):
        if not chain:
            raise ValueError(f"surface {f}: no image given")
        for m, image in enumerate(chain):
            _check_image(image, f"surface {f} level {m}")
    return chains


def _level_count(tex_format: int, width: int, height: int, depth: int, levels, chains,
                 template: TextureFile) -> int:
    if levels is not None:
        if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
            raise ValueError(f"levels {levels!r}: a positive integer is needed")
        count = levels
    else:
        given = max(len(chain) for chain in chains)
        limit = W.max_levels(tex_format, width, height, depth)
        if given > 1:
            count = given
        elif template.levels <= limit:
            count = template.levels
        else:
            count = limit
    W.check_levels(tex_format, width, height, count, depth)
    for f, chain in enumerate(chains):
        if len(chain) > count:
            raise ValueError(f"surface {f}: {len(chain)} levels given for a {count}-level texture")
    return count


def _same_image(a: Image, b: Image) -> bool:
    return (a.width, a.height, a.rgba) == (b.width, b.height, b.rgba)


# --- W16-2 / W16-3 --------------------------------------------------------------------------

def import_texture(template: TextureFile, images, *, levels: int | None = None,
                   format: int | None = None) -> bytes:
    """Build a ``.tex`` from RGBA images on a parsed template (W16-2 to W16-4)."""
    if not isinstance(template, TextureFile):
        raise ValueError("template: expected a parsed .tex (texture.TextureFile)")
    build = template.build
    if build not in W.BUILDS:
        raise FormatError(f"unknown build {build!r}")
    tex_format = template.format if format is None else format
    if isinstance(tex_format, bool) or not isinstance(tex_format, int) or tex_format not in FORMATS:
        raise ValueError(f"format {tex_format!r} is not a known texture format")
    kind = FORMATS[tex_format][0]
    faces, depth = template.faces, template.depth
    chains = _chains(images, faces, depth)
    width, height = chains[0][0].width, chains[0][0].height
    for f, chain in enumerate(chains):
        if (chain[0].width, chain[0].height) != (width, height):
            raise FormatError(f"surface {f} is {chain[0].width}x{chain[0].height}, "
                              f"the first is {width}x{height}")
    if depth != 1 and (kind, width, height) != ("argb8888", W.VOLUME_SIDE, W.VOLUME_SIDE):
        raise FormatError("a volume is 64 x 64 x 64 ARGB8888 with one level")
    count = _level_count(tex_format, width, height, depth, levels, chains, template)
    sizes = W.level_sizes(width, height, count)

    has_pixels = bool(template.payload) and template.format in FORMATS
    same_size = has_pixels and (width, height) == (template.width, template.height)
    keep = same_size and tex_format == template.format  # rule 2 applies
    stored = W.texture_elements(template) if keep else {}

    elements: dict[tuple[int, int], bytes] = {}
    if depth != 1:
        parts = []
        for z, chain in enumerate(chains):
            image = chain[0]
            if keep:
                size = len(stored[(0, 0)]) // template.depth
                parts.append(merge_elements(tex_format, build, stored[(0, 0)][z * size:(z + 1) * size],
                                            image))
            else:
                parts.append(W.element_form(image, tex_format, build))
        elements[(0, 0)] = b"".join(parts)
    else:
        for face, chain in enumerate(chains):
            chain = list(chain)
            if len(chain) < count:
                from_template = same_size and _same_image(chain[0], decode_template(template, 0, face))
                while len(chain) < count:
                    level = len(chain)
                    if from_template and level < template.levels:
                        chain.append(decode_template(template, level, face))
                    else:
                        chain.append(W.downsample(chain[-1]))
            for level in range(count):
                image = chain[level]
                if (image.width, image.height) != sizes[level]:
                    raise FormatError(
                        f"face {face} level {level} is {image.width}x{image.height}, "
                        f"{sizes[level][0]}x{sizes[level][1]} needed")
                if keep and level < template.levels:
                    elements[(level, face)] = merge_elements(
                        tex_format, build, stored[(level, face)], image)
                else:
                    elements[(level, face)] = W.element_form(image, tex_format, build)

    if keep and count == template.levels:
        from .write import write_tex

        payload = _patch_payload(template, elements)
        return write_tex(dataclasses.replace(template, payload=payload))
    payload = W.build_payload(build, tex_format, width, height, count, elements, faces, depth)
    return W.assemble_texture(build, tex_format, width, height, payload, count, faces, depth,
                              None, template)


__all__ = ["decode_form", "decode_template", "import_texture", "merge_elements", "split_strip"]
