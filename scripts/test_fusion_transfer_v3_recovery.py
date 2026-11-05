"""CPU-only orchestration regression tests; never imports a model or starts training."""
import ctypes
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_fusion_transfer_v3 as q


class AtomicTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/"state.json"
        self.path.write_text('{"old": true}', encoding="utf-8")

    @unittest.skipUnless(os.name == "nt", "Actual Windows sharing test")
    def test_real_windows_reader_temporarily_blocks_replace(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        # GENERIC_READ, FILE_SHARE_READ|WRITE, deliberately no FILE_SHARE_DELETE.
        handle = kernel.CreateFileW(str(self.path), 0x80000000, 3, None, 3, 0x80, None)
        self.assertNotIn(handle, (None, ctypes.c_void_p(-1).value))
        probe = self.path.with_suffix(".probe")
        probe.write_text("probe", encoding="utf-8")
        try:
            with self.assertRaises(PermissionError):
                os.replace(probe, self.path)
        finally:
            probe.unlink(missing_ok=True)
        errors = []
        def writer():
            try:
                q.atomic(self.path, {"new": True})
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=writer)
        thread.start()
        try:
            time.sleep(.15)
            self.assertEqual(json.loads(self.path.read_text()), {"old": True})
            self.assertTrue(thread.is_alive())
        finally:
            kernel.CloseHandle(handle)
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(json.loads(self.path.read_text()), {"new": True})
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_persistent_denial_preserves_original_and_cleans_temp(self):
        before = self.path.read_bytes()
        with patch.object(q.os, "replace", side_effect=PermissionError("denied")) as replace:
            with self.assertRaises(PermissionError):
                q.atomic(self.path, {"new": True}, retry_seconds=.06)
        self.assertGreater(replace.call_count, 1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_non_sharing_error_is_not_retried(self):
        with patch.object(q.os, "replace", side_effect=OSError("disk failure")) as replace:
            with self.assertRaises(OSError):
                q.atomic(self.path, {"new": True})
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(json.loads(self.path.read_text()), {"old": True})

    def test_concurrent_writers_use_unique_temps_and_valid_json(self):
        errors = []
        def writer(number):
            try:
                for index in range(12):
                    q.atomic(self.path, {"writer": number, "index": index})
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                q.read(self.path)
        finally:
            for thread in threads:
                thread.join(10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_read_retries_temporary_denial(self):
        with patch.object(q, "read_json", side_effect=[PermissionError("busy"), {"ok": True}]):
            self.assertEqual(q.read(self.path), {"ok": True})

    def test_read_persistent_denial_is_bounded(self):
        with patch.object(q, "read_json", side_effect=PermissionError("busy")):
            with self.assertRaises(PermissionError):
                q.read(self.path, retry_seconds=.03)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root/"completed").mkdir()
        self.manifest = {"jobs": [{"id": f"C{i:02d}_seed6", "seed": 6, "group": f"C{i:02d}",
            **{k: str(self.root/f"{i}.{k}") for k in q.ARTIFACTS}} for i in (1, 2, 3)]}
        self.addCleanup(patch.stopall)
        patch.object(q, "DEST", self.root).start()
        self.audit = patch.object(q, "audit_job").start()

    def marker(self, index):
        job = self.manifest["jobs"][index]
        q.write(self.root/"completed"/(job["id"]+".json"), {"id": job["id"]})

    def test_verified_prefix_skips_only_c01(self):
        self.marker(0)
        prefix = q.completed_prefix(self.manifest)
        self.assertEqual(prefix, 1)
        self.assertEqual(self.manifest["jobs"][prefix]["id"], "C02_seed6")
        self.audit.assert_called_once()

    def test_invalid_completed_evidence_aborts(self):
        self.marker(0)
        self.audit.side_effect = AssertionError("hash mismatch")
        with self.assertRaises(AssertionError):
            q.completed_prefix(self.manifest)

    def test_gap_aborts(self):
        self.marker(1)
        with self.assertRaisesRegex(RuntimeError, "Non-contiguous"):
            q.completed_prefix(self.manifest)

    def test_unsealed_output_aborts_instead_of_retraining(self):
        self.marker(0)
        Path(self.manifest["jobs"][1]["stdout"]).write_text("partial")
        with self.assertRaisesRegex(RuntimeError, "Unsealed artifacts"):
            q.completed_prefix(self.manifest)

    def test_unknown_marker_aborts(self):
        q.write(self.root/"completed/unknown.json", {})
        with self.assertRaisesRegex(RuntimeError, "Unknown completed"):
            q.completed_prefix(self.manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
