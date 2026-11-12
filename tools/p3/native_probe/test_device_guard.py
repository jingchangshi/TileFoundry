#!/usr/bin/env python3
"""Device identity gate tests: visible list + allow set + container presence (CPU only)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from device_guard import GuardError, check_device_identity  # noqa: E402

BOTH = [0, 1]
ONLY0 = [0]


class DeviceIdentityTests(unittest.TestCase):
    def _code(self, env, container_ids=BOTH):
        with self.assertRaises(GuardError) as cm:
            check_device_identity(env, container_ids=container_ids)
        return cm.exception.code

    def test_empty_visible_is_missing(self):
        self.assertEqual(self._code({"ASCEND_RT_VISIBLE_DEVICES": ""}), "DEVICE_ID_MISSING")

    def test_missing_visible_defaults_to_both(self):
        ident = check_device_identity({}, container_ids=BOTH)
        self.assertEqual(ident["visible_ids"], [0, 1])
        self.assertEqual(ident["effective_device"], 0)

    def test_non_integer_is_invalid(self):
        self.assertEqual(self._code({"ASCEND_RT_VISIBLE_DEVICES": "a,b"}), "DEVICE_ID_INVALID")

    def test_outside_allow_is_not_allowed(self):
        self.assertEqual(self._code({"ASCEND_RT_VISIBLE_DEVICES": "2"}), "DEVICE_ID_NOT_ALLOWED")

    def test_duplicate_is_refused(self):
        self.assertEqual(self._code({"ASCEND_RT_VISIBLE_DEVICES": "0,0"}), "DEVICE_ID_DUPLICATE")

    def test_single_zero_passes_effective_0(self):
        ident = check_device_identity({"ASCEND_RT_VISIBLE_DEVICES": "0"}, container_ids=BOTH)
        self.assertEqual(ident["effective_device"], 0)
        self.assertEqual(ident["container_visible_ids"], [0, 1])

    def test_single_one_passes_effective_1(self):
        ident = check_device_identity({"ASCEND_RT_VISIBLE_DEVICES": "1"}, container_ids=BOTH)
        self.assertEqual(ident["effective_device"], 1)

    def test_zero_one_passes_effective_0(self):
        ident = check_device_identity({"ASCEND_RT_VISIBLE_DEVICES": "0,1"}, container_ids=BOTH)
        self.assertEqual(ident["effective_device"], 0)

    def test_one_zero_passes_effective_1(self):
        ident = check_device_identity({"ASCEND_RT_VISIBLE_DEVICES": "1,0"}, container_ids=BOTH)
        self.assertEqual(ident["effective_device"], 1)

    def test_allow_set_can_reject_one(self):
        self.assertEqual(self._code({"ASCEND_RT_VISIBLE_DEVICES": "1",
                                     "TF_DEVICE_ALLOW": "0"}), "DEVICE_ID_NOT_ALLOWED")

    def test_container_missing_one_is_not_in_container(self):
        self.assertEqual(self._code({"ASCEND_RT_VISIBLE_DEVICES": "1"},
                                    container_ids=ONLY0), "DEVICE_NOT_IN_CONTAINER")

    def test_container_missing_list_is_not_in_container(self):
        self.assertEqual(self._code({"ASCEND_RT_VISIBLE_DEVICES": "0,1"},
                                    container_ids=ONLY0), "DEVICE_NOT_IN_CONTAINER")

    def test_container_zero_still_passes(self):
        ident = check_device_identity({"ASCEND_RT_VISIBLE_DEVICES": "0"}, container_ids=ONLY0)
        self.assertEqual(ident["effective_device"], 0)


if __name__ == "__main__":
    unittest.main()
