"""Corrections 36 (``57_corrections36.md``): AG1 (x360 rows) and AG2 (the record header), with the tables
``f57_*``. Synthetic files built from the table's rows only; nothing is written to disk."""

import _libpath  # noqa: F401  (python -I: the library folder on sys.path)
import copy
import struct
import unittest

import test_stage23 as T
import d3sno
from d3sno import effects as E
from d3sno import json_import as J
from d3sno import sno_layout as L
from d3sno.binary import build_order
from d3sno.datafiles import DATA


def row(build, ext, record, name):
    return next(r for r in E.leaf_table(build, ext) if r.record == record and r.name == name)


class Tables(unittest.TestCase):
    def test_f57_in_use(self):
        self.assertEqual((E.LAYOUT_TABLE, E.ENUM_TABLE), ("f57_layouts.tsv", "f57_enums.tsv"))
        for name in ("f55_layouts.tsv", "f56_layouts.tsv", "f57_layouts.tsv", "f57_enums.tsv"):
            self.assertTrue((DATA / name).is_file(), name)  # the older tables stay


class X360Rows(unittest.TestCase):
    """AG1."""

    def test_no_conflict_but_ctr_and_dfn(self):
        for ext in E.GROUPS:
            found = E.decode_full(L.write_tree(T.blank("x360", ext, 9), derive=True), ext, "x360")
            if ext in ("ctr", "dfn"):
                continue
            self.assertEqual(found.conflicts, [], ext)

    def test_switch_layout_records(self):
        """Six records whose x360 layout is the Switch layout."""
        for ext in ("efg", "mus", "rop", "an2", "scg", "ant"):
            top = E._top_table_record("switch", ext)
            switch = {r.name: (r.offset, r.pair_offset) for r in E.leaf_table("switch", ext) if r.record == top}
            x360 = {r.name: (r.offset, r.pair_offset) for r in E.leaf_table("x360", ext) if r.record == top}
            self.assertEqual(x360, {k: v for k, v in switch.items() if k in x360}, ext)
        effects = row("x360", "efg", E._top_table_record("x360", "efg"), "ptEffects")
        self.assertEqual((effects.offset, effects.pair_offset), (32, 16))
        tree = T.blank("x360", "efg", 9)
        slot = next(s for s in L.pair_slots("x360", "efg", L.top_record("x360", "efg")) if s.offset == 16)
        size = L.record_layout("x360", "efg", slot.field.element).size
        tree.top.pairs[16] = L.PairValue(L.Block(elements=[L.Record(slot.field.element, bytearray(size))]))
        data = L.write_tree(tree, derive=True)
        found = E.decode_full(data, "efg", "x360")
        self.assertEqual(found.conflicts, [])
        self.assertEqual(len(found.json["record"]["ptEffects"]), 1)

    def test_observer_four_lower(self):
        top = E._top_table_record("x360", "obs")
        names = {r.name for r in E.leaf_table("switch", "obs") if r.record == top}
        self.assertIn("flOrthographicHeight", names)
        self.assertNotIn("flOrthographicHeight", {r.name for r in E.leaf_table("x360", "obs") if r.record == top})
        cut = row("switch", "obs", top, "flOrthographicHeight").offset
        later = [r for r in E.leaf_table("switch", "obs") if r.record == top and r.offset is not None and r.offset > cut]
        self.assertTrue(later)
        for leaf in later:
            self.assertEqual(row("x360", "obs", top, leaf.name).offset, leaf.offset - 4, leaf.name)

    def test_particle_sixteen_lower(self):
        top = E._top_table_record("x360", "prt")
        events = row("x360", "prt", top, "ptMsgTriggeredEvents")
        self.assertEqual((events.pair_offset, events.offset), (668, 680))
        self.assertFalse([r for r in E.leaf_table("x360", "prt") if r.name == "ptParticleQuadStructure"])

    def test_encounter_spawn_option(self):
        rows = [r for r in E.leaf_table("x360", "enc") if r.record == "EncounterSpawnOption"]
        self.assertNotIn("snoCondition", {r.name for r in rows})
        self.assertLessEqual(max(r.offset + r.size for r in rows), 12)


class RecordHeader(unittest.TestCase):
    """AG2: ``tHeader.*`` named on every build."""

    def test_every_group_and_build(self):
        for build in T.BUILDS:
            for ext in E.GROUPS:
                with self.subTest(build=build, ext=ext):
                    tree = T.blank(build, ext, 4321)
                    data = L.write_tree(tree, derive=True)
                    record = T.doc(data, ext, build)["record"]
                    self.assertEqual(record[E.OWN_HANDLE], 4321)
                    self.assertEqual(record["tHeader.dwFlags"], 0)
                    top = E._top_table_record(build, ext)
                    lock = row(build, ext, top, "tHeader.nLockCount")
                    if lock.offset is None:  # an open PS3 row (``wth``, ``obs``: another row of f57 holds word 4)
                        self.assertEqual(lock.status, "open")
                        self.assertNotIn("tHeader.nLockCount", record)
                    else:
                        self.assertEqual(record["tHeader.nLockCount"], 0)
                        self.assertNotIn("0x0004", record.get("unnamed", {}))
                    self.assertFalse({"0x0000", "0x0008"} & set(record.get("unnamed", {})))

    def test_handle_equals_the_file_header(self):
        for build in T.BUILDS:
            data = T.vib_file(build, handle=777)
            found = T.doc(data, "vib", build)
            self.assertEqual(found["record"][E.OWN_HANDLE], found["header"]["handle"])

    def test_handle_refused(self):
        for build in T.BUILDS:
            data = T.vib_file(build)
            edited = T.doc(data, "vib", build)
            edited["record"][E.OWN_HANDLE] = 5
            with self.assertRaises(J.Refused) as caught:
                E.encode(data, "vib", build, edited)
            self.assertEqual(caught.exception.reason, "field_fixed")

    def test_lock_count_and_flags(self):
        for build in T.BUILDS:
            data = T.vib_file(build)
            original = T.doc(data, "vib", build)
            edited = copy.deepcopy(original)
            edited["record"]["tHeader.nLockCount"] = 3
            edited["record"]["tHeader.dwFlags"] = 0x40
            first = E.encode(data, "vib", build, edited)
            record = T.doc(first, "vib", build)["record"]
            self.assertEqual((record["tHeader.nLockCount"], record["tHeader.dwFlags"]), (3, 0x40))
            self.assertEqual(struct.unpack_from(build_order(build) + "i", first, 0x14)[0], 3)
            self.assertEqual(E.encode(first, "vib", build, original), data)


class WorkerCounts(unittest.TestCase):
    """AG3: after ``close`` the worker counts are final."""

    def test_dropped_work_200_times(self):
        import test_stage22_corrections32 as C32

        case = C32.Jobs("test_dropped_work_is_let_go")
        for _ in range(200):
            result = unittest.TestResult()
            case.run(result)
            self.assertEqual((result.failures, result.errors), ([], []))

    def test_queued_work_cancelled_at_close(self):
        """Work still queued when ``close`` is called is cancelled and counted; nothing runs after ``close``."""
        import threading
        import time
        from d3sno import archive as A

        workers = A.Workers(2)
        gate = threading.Event()
        busy = [workers.submit(gate.wait, 5) for _ in range(2)]
        time.sleep(0.05)
        late = workers.submit(lambda: 4)  # both threads busy: queued
        threading.Timer(0.2, gate.set).start()
        workers.close()
        stats = workers.stats()
        self.assertTrue(late.cancelled())
        self.assertTrue(all(f.done() for f in busy))
        self.assertEqual(sum(row["done"] for row in stats["per_worker"]), 2)
        self.assertEqual((stats["submitted"], stats["cancelled"]), (3, 1))
        time.sleep(0.05)
        self.assertEqual(workers.stats(), stats)  # final

    def test_dropped_before_it_ran(self):
        import threading
        from d3sno import archive as A

        workers = A.Workers(2)
        gate = threading.Event()
        try:
            busy = [workers.submit(gate.wait, 5) for _ in range(2)]
            queued = workers.submit(lambda: 1)
            workers.drop(queued)
            self.assertTrue(queued.cancelled())
        finally:
            gate.set()
            workers.close()
        stats = workers.stats()
        self.assertEqual(stats["cancelled"], 1)
        self.assertEqual(sum(row["done"] for row in stats["per_worker"]) + stats["cancelled"], stats["submitted"])


class Version(unittest.TestCase):
    def test_version(self):
        self.assertGreaterEqual(tuple(int(p) for p in d3sno.__version__.split(".")), (0, 23, 2))
        self.assertEqual(d3sno.STAGE, 23)


if __name__ == "__main__":
    unittest.main()
