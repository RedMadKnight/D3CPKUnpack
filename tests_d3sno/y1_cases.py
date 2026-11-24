"""``47`` Y1 / Y10: deterministic synthetic archives and changes for the byte-equality tests of the
streaming archive writer. Each case gives the SHA-256 of what a byte-returning function writes;
``GOLDEN_0_21_0`` holds the values version 0.21.0 gave for them (recorded before Y1 was made, with the
zlib named there: other deflaters give other level-9 bytes, so the check against 0.21.0 then skips)."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import hashlib
import random

from d3sno import archive as A
from d3sno import index_write as IW
from d3sno import x360_archive as XA


def data(seed: int, size: int) -> bytes:
    """Blocks of random bytes and of a repeated short pattern (both packing paths get used)."""
    rnd = random.Random(seed)
    out = bytearray()
    while len(out) < size:
        if rnd.random() < 0.5:
            out += rnd.randbytes(rnd.randrange(1, 40000))
        else:
            out += rnd.randbytes(rnd.randrange(1, 50)) * rnd.randrange(1, 3000)
    return bytes(out[:size])


def files(seed: int, big: int = 2_500_000) -> dict:
    return {
        "Base\\a.dat": data(seed, 70_000),
        "Base\\big.dat": data(seed + 1, big),
        "Base\\empty.dat": b"",
        "Sound\\music.fsb": data(seed + 2, 90_000),
        "Base\\small.txt": b"hello world\n" * 20,
        "Base\\b.dat|enUS": data(seed + 3, 3_000),
    }


def ps3_source() -> bytes:
    f = files(1)
    inputs = [A.InputFile(n, [d, d] if n == "Base\\a.dat" else [d]) for n, d in f.items()]
    return A.write_archive(inputs, version=6, order=">", flags=2)


def switch_source(flags: int = 2) -> bytes:
    f = files(2)
    inputs = [A.InputFile(n, [d, d] if n == "Base\\small.txt" else [d]) for n, d in f.items()]
    return A.write_archive(inputs, version=7, order="<", flags=flags, read_sector=0x800 if flags == 10 else None,
                           encrypt=lambda n: n % 5 == 1)


def x360_source(flags: int = 2) -> bytes:
    f = files(3)
    pieces = []
    for name, content in f.items():
        pieces.append(XA.Piece(name, 0, content))
    pieces.insert(2, XA.Piece("Base\\a.dat", 1, f["Base\\a.dat"]))  # a second copy, not adjacent
    return XA.write_x360_archive(pieces, flags=flags, read_sector=0x800, encrypted=[(0, 0), (3, 0)]).data


SOURCES = {
    "ps3": ps3_source,
    "switch": switch_source,
    "switch_compact": lambda: switch_source(10),
    "switch_raw": lambda: switch_source(0),
    "x360": x360_source,
    "x360_compact": lambda: x360_source(10),
    "x360_raw": lambda: x360_source(0),
}

CHANGES = {
    "replace_small": lambda: ({"Base\\small.txt": b"changed text\n" * 7}, {}),
    "replace_big": lambda: ({"Base\\big.dat": data(99, 1_300_000)}, {}),
    "replace_copies": lambda: ({"Base\\a.dat": data(98, 50_000), "Base\\empty.dat": b"now some bytes"}, {}),
    "add": lambda: ({}, {"New\\added.dat": data(97, 200_000)}),
    "both": lambda: ({"Sound\\music.fsb": data(96, 33_000)}, {"New\\x.dat|frFR": b"x" * 5000}),
}


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def outputs():
    """(case name, bytes-returning call) pairs; every call reads its source afresh."""
    for source_name, make in SOURCES.items():
        for change_name, change in CHANGES.items():
            def call(make=make, change=change):
                replace, add = change()
                return IW.replace_files(make(), replace, add)
            yield f"{source_name}/{change_name}", call
    yield "ps3/rebuild_remove", lambda: A.rebuild(ps3_source(), remove=["Base\\small.txt"]).data
    yield "ps3/rebuild_single", lambda: A.rebuild(ps3_source(), single_copy=True).data
    yield "switch/replace_file", lambda: IW.replace_file(switch_source(), "Base\\big.dat", b"short")
    yield "x360/add_file", lambda: IW.add_file(x360_source(), "New\\one.dat", data(95, 70_000))


# Version 0.21.0, recorded with the zlib named below (``x360_archive.zlib_implementation()``).
GOLDEN_ZLIB = 'zlib-ng 2.2.4 (zlib API 1.3.1.zlib-ng)'
GOLDEN_0_21_0 = {
    'apply_three_edits/Act1.cpk': '86e910f4e0dce5fe6ef8ff4d91abdf0547ea152835181f7c5c5ac4dbb743c4c0',
    'apply_three_edits/Common.cpk': '49265bc03f7e235b35cc1814091a2e047ed814dfe283c4b8fa1c6802659e4c40',
    'ps3/add': 'dccd06566010b115aa9d07f5cc67cb975100acb080f22a7e016b38dcde1284f0',
    'ps3/both': 'c1ebf7531ed7693a1913129a9956ce8333918a5ae10082ff42d8cf3bef18730d',
    'ps3/rebuild_remove': '75dd06ff6ba4edbd5b2fe0e9ae0364c07809b657df8cc9bb006c093f3bfc3511',
    'ps3/rebuild_single': 'd0afff432205e845b2c022f246e3bcdaaad0aa036c28773df325adaab2626988',
    'ps3/replace_big': '698ee9af088298604409ad027fb06c57e51fa3f717c32094c5e720eda2cee5d7',
    'ps3/replace_copies': '4d6d1b819d9b3a6adeb6d6ec9e6c9a3d1478a0bfea16443aa8aca329459d2a8b',
    'ps3/replace_small': '9bca7a9c5098676ad065abe58f5984ebef9d20e7fb6afd45a3890251a3e63bb6',
    'switch/add': '8d1636ab9a14a72224fd072c92631e5f33087ce83d6813a17aeb36994b6bfe14',
    'switch/both': '2f72c0d978062a72f6fd6d3d8c473ef1be2429cd517f3a871a470ae424b4afbf',
    'switch/replace_big': '1cef3474acec41e3f129ce95456b1e7210b84effce2dc678a88f4dbc2c284c47',
    'switch/replace_copies': '8254890ebd9c23967669b2e49a50766f1b161a09f40f5ae2e5f10d9c481bf34a',
    'switch/replace_file': '2e06fe80e633c55b1880af998a28d5781cb28d80f49dd89eb83baf59383db209',
    'switch/replace_small': 'fbb01d1445c29863b39601b6eb09b91765f5714bdbc2da868b1ab889614838ae',
    'switch_compact/add': '9828a5d9a8cde8b1fd90404f30def176ab6594f32c92ff61b2d6e928fe8267ce',
    'switch_compact/both': 'db1f6dbfb6ccb84e8471af175683a32f0e2e3fbfd6ff9f9d945e576ded179dc4',
    'switch_compact/replace_big': '6b8ce5bba4998698fcf8e8fde473c6a7b1cbd0b6f93e77dda84f49a15070ea25',
    'switch_compact/replace_copies': '953eec535d58dce8adca527d1267c6acab44fdddcb80055743b104d71086d26b',
    'switch_compact/replace_small': '06a121431a4b2f6e9d10b8361d117ed1f10a1cd1b774019ec1598d0220987e39',
    'switch_raw/add': '78a81ce2d633be0e2a3abf2de21103b41964691439c7f527d4d5ff9c576bfc3a',
    'switch_raw/both': '82782e8ddf13f8e041edb70f8c80709953c672fc78b07780ece265b368447e02',
    'switch_raw/replace_big': 'c0dbe90a7dbf8a6da4e9e86d9aa0482a37c548910cbe7ec451323f1f5b35607a',
    'switch_raw/replace_copies': '64029c7b8ccc9895fd57392f685cc3809baefed96ac75cf81923e773cbd1ea0e',
    'switch_raw/replace_small': '9b17d962395ac10eb5b48440e36b27a0761e2659fa8b08883aa13a26196a6c49',
    'x360/add': '32b39a7a5f587fc02c984a7cb1f4bbf9cb2eafa84ea46377795359154576349e',
    'x360/add_file': '47d573d4bee904a2755179284f8cec9b471f9d19ef1c54921651fbec2c4baffd',
    'x360/both': '9c9262485c946548adaa86ed861fadfb0f99c88792f224b847c1cba5ea628798',
    'x360/replace_big': '7784265aa9503fa9f5b9d188d49082df765d18a6ac05651d5f60068ba2162c44',
    'x360/replace_copies': 'c4b7f8b4df4b2e1786216b6e87dd532b3d058e5dabdaccb7f65b8f3dcfb53e1c',
    'x360/replace_small': '7b13bfe2eb1d5670d4f35a4f0648995e5aadf2959454abe1d752de613231bc5b',
    'x360_compact/add': '9a040dff152d97321c5dcad1b2b3b4bc4bec8c7fc771ce9b5a340caa122ed83e',
    'x360_compact/both': '5ce94400bade2b2f29b49ae71a4c7c123124fa47380e98a235ad1874007eff8e',
    'x360_compact/replace_big': '0764fef42e8903c71c49536c94cab9446927dde3e4e6ff6de6b65b0717ad7737',
    'x360_compact/replace_copies': '21f16dc4f71f9da34292cf2087bcad8241441b1127a5b19c681366036256b280',
    'x360_compact/replace_small': '13c761015bee5015eca460683283498c53953b91813a95e7bf63378334128b94',
    'x360_raw/add': 'cf1c05549bc87418a3e96eb9b9e91d1e0d6e945d592428d6ef5b1839a77c0cd9',
    'x360_raw/both': '2663e422de97e7bded7139ea721ac35a4ba0294a7b2f0d5669b4deeafc4a7a0e',
    'x360_raw/replace_big': 'ac19b9abd3e48ab46e4dc39339c0bef5fdf6afa99eb562a0ccdbdd3cc153bdab',
    'x360_raw/replace_copies': '259dcaef0be6b241e62a57154b882c9fb348cdd18c90fee00c0124ade8d0c9bc',
    'x360_raw/replace_small': 'da4fbb6265cee683b9efcbf9187feaef66a28e0f6e1624f7cbc10483a61724e0',
}
