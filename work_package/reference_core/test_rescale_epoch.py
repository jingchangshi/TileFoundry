"""Logical protocol/number tests; no device, hardware event, or FP32 proof."""
from dataclasses import FrozenInstanceError
from math import exp
import unittest

from rescale_epoch_reference import Epoch, EpochProtocol, Producer, ProtocolError


def epoch(ordinal, anchors, *producers):
    return Epoch(ordinal, tuple(anchors), frozenset(producers))


def activate(protocol):
    for consumer in sorted(protocol.consumers):
        protocol.retire_metadata_consumer(consumer)
    protocol.activate_next_epoch()


class RescaleEpochTests(unittest.TestCase):
    def test_two_block_late_pv_counterexample_and_correct_result(self):
        # Scores 0/10, V=1: both the exact attention and correct protocol give 1.
        p0, p1 = Producer(0, 0), Producer(1, 0)
        protocol = EpochProtocol(epoch(0, (0.0,), p0), 1)
        protocol.add_mass(0, 0, (1.0,))
        protocol.publish_boundary(epoch(1, (10.0,), p1), predicate=True, factors=(exp(-10),))
        with self.assertRaisesRegex(ProtocolError, "all old-anchor PV"):
            protocol.drain_and_clear()
        # An illegal early drain followed by unscaled old PV produces ~2.
        illegal_result = (1.0 + 1.0) / (exp(-10) + 1.0)
        self.assertGreater(illegal_result, 1.99)
        protocol.complete_pv(0, p0, (1.0,))
        protocol.drain_and_clear()
        activate(protocol)
        protocol.add_mass(1, 1, (1.0,))
        protocol.complete_pv(1, p1, (1.0,))
        self.assertAlmostEqual(protocol.output()[0][0], 1.0, places=14)

    def test_all_output_columns_and_hi_lo_must_complete(self):
        producers = tuple(Producer(0, col, part) for col in range(2) for part in ("hi", "lo"))
        protocol = EpochProtocol(epoch(0, (0.0,), *producers), 2)
        protocol.add_mass(0, 0, (1.0,))
        protocol.publish_boundary(epoch(1, (2.0,)), predicate=True, factors=(exp(-2),))
        for producer in producers[:-1]:
            protocol.complete_pv(0, producer, (0.75 if producer.component == "hi" else 0.25,))
        with self.assertRaisesRegex(ProtocolError, "all old-anchor PV"):
            protocol.drain_and_clear()  # Missing the last column's low component.
        protocol.complete_pv(0, producers[-1], (0.25,))
        protocol.drain_and_clear()
        activate(protocol)
        self.assertEqual(protocol.output(), ((1.0, 1.0),))

    def test_stale_epoch_and_writes_after_drain_are_rejected(self):
        p0, p1 = Producer(0, 0), Producer(1, 0)
        protocol = EpochProtocol(epoch(0, (0.0,), p0), 1)
        protocol.add_mass(0, 0, (1.0,))
        protocol.complete_pv(0, p0, (1.0,))
        protocol.publish_boundary(epoch(1, (1.0,), p1), predicate=True, factors=(exp(-1),))
        with self.assertRaisesRegex(ProtocolError, "not-yet-active"):
            protocol.add_mass(1, 1, (1.0,))
        protocol.drain_and_clear()
        with self.assertRaisesRegex(ProtocolError, "after drain"):
            protocol.complete_pv(0, p0, (1.0,))
        activate(protocol)
        with self.assertRaisesRegex(ProtocolError, "stale"):
            protocol.complete_pv(0, p0, (1.0,))

    def test_predicate_and_factor_cannot_be_overwritten_before_last_use(self):
        protocol = EpochProtocol(epoch(0, (0.0,)), 1)
        next_epoch = epoch(1, (1.0,))
        protocol.publish_boundary(next_epoch, predicate=True, factors=(exp(-1),))
        with self.assertRaises(FrozenInstanceError):
            protocol.boundary.predicate = False
        protocol.drain_and_clear()
        protocol.retire_metadata_consumer("aic")
        protocol.retire_metadata_consumer("aiv0")
        with self.assertRaisesRegex(ProtocolError, "metadata uses"):
            protocol.activate_next_epoch()
        with self.assertRaisesRegex(ProtocolError, "payload still"):
            protocol.publish_boundary(next_epoch, predicate=False, factors=(1.0,))
        protocol.retire_metadata_consumer("aiv1")
        protocol.activate_next_epoch()
        protocol.publish_boundary(epoch(2, (2.0,)), predicate=True, factors=(exp(-1),))

    def test_group_drains_but_unchanged_row_uses_factor_one(self):
        p0, p1 = Producer(0, 0), Producer(1, 0)
        protocol = EpochProtocol(epoch(0, (0.0, 3.0), p0), 1)
        protocol.add_mass(0, 0, (1.0, 1.0))
        protocol.complete_pv(0, p0, (2.0, 5.0))
        next_epoch = epoch(1, (2.0, 3.0), p1)
        with self.assertRaisesRegex(ProtocolError, "unchanged rows"):
            protocol.publish_boundary(next_epoch, predicate=True, factors=(exp(-2), exp(-2)))
        protocol.publish_boundary(next_epoch, predicate=True, factors=(exp(-2), 1.0))
        protocol.drain_and_clear()
        activate(protocol)
        protocol.add_mass(1, 1, (1.0, 1.0))
        protocol.complete_pv(1, p1, (4.0, 7.0))
        actual = protocol.output()
        self.assertAlmostEqual(actual[0][0], (2 * exp(-2) + 4) / (exp(-2) + 1))
        self.assertEqual(actual[1][0], 6.0)

    def test_wrong_predicate_factor_and_missing_final_producer_are_rejected(self):
        p = Producer(0, 0)
        protocol = EpochProtocol(epoch(0, (0.0,), p), 1)
        with self.assertRaisesRegex(ProtocolError, "predicate/anchor"):
            protocol.publish_boundary(epoch(1, (1.0,)), predicate=False, factors=(exp(-1),))
        with self.assertRaisesRegex(ProtocolError, "factor must"):
            protocol.publish_boundary(epoch(1, (1.0,)), predicate=True, factors=(1.0,))
        with self.assertRaisesRegex(ProtocolError, "all old-anchor PV"):
            protocol.output()
        protocol.add_mass(0, 0, (1.0,))
        protocol.complete_pv(0, p, (2.0,))
        with self.assertRaisesRegex(ProtocolError, "duplicate PV"):
            protocol.complete_pv(0, p, (2.0,))
        self.assertEqual(protocol.output(), ((2.0,),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
