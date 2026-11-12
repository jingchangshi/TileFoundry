"""Meaningful CPU tests for the reference algorithms; no hardware claims."""
import math
import unittest
from copy import deepcopy

from online_softmax import State, update, update_rows, merge, attention_residual_row, mla_rows
from pipeline_reference import Loop, Symbol, Task, Schedule, PipelineTemplate
from completion_reference import Buffer, Event, RingProtocol
from reference_ops import attention_residual, mla_dense


def dense(logits, values, mask=None):
    mask = [True] * len(logits) if mask is None else mask
    indices = [i for i, allowed in enumerate(mask) if allowed and logits[i] != -math.inf]
    if not indices:
        return tuple(0.0 for _ in values[0])
    maximum = max(logits[i] for i in indices)
    weights = [math.exp(logits[i] - maximum) if i in indices else 0.0 for i in range(len(logits))]
    total = sum(weights)
    return tuple(sum(p * v[d] for p, v in zip(weights, values)) / total
                 for d in range(len(values[0])))


class OnlineSoftmaxTests(unittest.TestCase):
    def close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertTrue(math.isclose(a, e, rel_tol=2e-12, abs_tol=2e-12), (actual, expected))

    def test_extreme_scores_and_partition_merge(self):
        scores = [-1000.0, 1000.0, 998.0, -3.0, 999.0]
        values = [(1.0, -1.0), (2.0, 3.0), (-4.0, 2.0), (5.0, 6.0), (-2.0, 9.0)]
        expected = dense(scores, values)
        for tile in (1, 2, 3, 8):
            state = State.empty(2)
            for begin in range(0, len(scores), tile):
                state = update(state, scores[begin:begin + tile], values[begin:begin + tile])
            self.close(state.normalized(), expected)
        left, right = update(State.empty(2), scores[:2], values[:2]), update(State.empty(2), scores[2:], values[2:])
        self.close(merge(left, right).normalized(), expected)
        self.close(merge(right, left).normalized(), expected)

    def test_empty_mask_does_not_produce_nan(self):
        empty = State.empty(2)
        self.assertEqual(update(empty, [-math.inf], [(3.0, 4.0)]), empty)
        self.assertEqual(update(empty, [math.nan], [(math.nan, math.nan)], [False]), empty)
        self.assertEqual(empty.normalized(), (0.0, 0.0))
        live = update(empty, [1000.0], [(3.0, 4.0)])
        self.assertEqual(update(live, [math.inf], [(0.0, 0.0)], [False]), live)
        self.assertEqual(merge(empty, live), live)
        self.assertEqual(merge(live, empty), live)

    def test_multirow_and_mask(self):
        values = [(1.0, 2.0), (3.0, 4.0), (-3.0, 5.0)]
        logits = [[1.0, 2.0, 8.0], [-1.0, 5.0, 2.0]]
        masks = [[True, True, False], [False, False, False]]
        states = update_rows([State.empty(2), State.empty(2)], logits, [values, values], masks)
        self.close(states[0].normalized(), dense(logits[0], values, masks[0]))
        self.assertEqual(states[1].normalized(), (0.0, 0.0))

    def test_attention_uses_depth_axis_original_values_and_real_width(self):
        v = [[0.2, -0.3, 0.4], [0.5, 0.9, -0.8], [0.0, 0.0, 0.0]]
        x, w, gamma = [0.4, 0.1, -0.3], [1.3, -0.5, 0.6], [0.8, 1.2, 0.7]
        scores = [sum((row[d] / math.sqrt(sum(t*t for t in row)/3 + 1e-6)) * gamma[d] * w[d]
                      for d in range(3)) / math.sqrt(3) for row in v]
        aggregate = dense(scores, v)
        expected = tuple(0.65*a + 0.35*b for a, b in zip(x, aggregate))
        self.close(attention_residual_row(v, x, w, gamma, 0.35), expected)

    def test_mla_right_aligned_causal_and_rope_once(self):
        qn, qr = [[1., 2., -1.], [.2, .4, 1.]], [[.5], [2.]]
        kn = [[.1, .3, .5], [2., -1., .2], [1., 1., 1.], [-2., .3, .9], [.5, .6, .7]]
        kr = [[1.], [2.], [-.5], [.3], [3.]]
        values = [row[:] for row in kn]
        for causal in (False, True):
            expected = []
            for r in range(2):
                scores = [(sum(a*b for a, b in zip(qn[r], k)) + qr[r][0]*kr[j][0])/2
                          for j, k in enumerate(kn)]
                expected.append(dense(scores, values, [not causal or j <= r+3 for j in range(5)]))
            for block in (1, 2, 4, 8):
                actual = mla_rows(qn, qr, kn, kr, values, query_positions=[0, 1], query_length=2,
                                  causal=causal, block_kv=block)
                for row, ref in zip(actual, expected):
                    self.close(row, ref)
        with self.assertRaises(ValueError):
            mla_rows(qn, qr, kn, kr, values, query_positions=[0, 1], query_length=6, causal=True)


class FullOperatorTests(unittest.TestCase):
    def close_nested(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            if isinstance(e, (list, tuple)):
                self.close_nested(a, e)
            else:
                self.assertTrue(math.isclose(a, e, rel_tol=2e-12, abs_tol=2e-12), (a, e))

    def test_attention_batch_token_mapping(self):
        v = [[[[0.2*n + 0.1*b - 0.3*t + 0.4*d for d in range(3)]
               for t in range(2)] for b in range(2)] for n in range(3)]
        x = [[[0.1*(1+b+t+d) for d in range(3)] for t in range(2)] for b in range(2)]
        w, gamma, alpha = [0.3, -0.7, 1.2], [1.0, 0.8, 1.3], 0.65
        expected = []
        for b in range(2):
            tokens = []
            for t in range(2):
                values = [v[n][b][t] for n in range(3)]
                scores = [sum(row[d]*gamma[d]*w[d] for d in range(3)) /
                          math.sqrt(sum(a*a for a in row)/3 + 1e-6)/math.sqrt(3) for row in values]
                aggregate = dense(scores, values)
                tokens.append([(1-alpha)*x[b][t][d] + alpha*aggregate[d] for d in range(3)])
            expected.append(tokens)
        self.close_nested(attention_residual(v, x, w, gamma, alpha), expected)

    def test_mla_layout_gqa_causal_and_scale(self):
        qn = [[[[0.2+b+0.3*i+0.4*h, -0.5+0.2*h] for h in range(4)]
               for i in range(2)] for b in range(2)]
        qr = [[[[0.1*(1+i+h)] for h in range(4)] for i in range(2)] for _ in range(2)]
        kn = [[[[0.2*j+0.6*h, -0.3+0.4*h+0.1*b] for h in range(2)]
               for j in range(5)] for b in range(2)]
        kr = [[[[0.3-0.1*j+0.2*h] for h in range(2)] for j in range(5)] for _ in range(2)]
        def bnsd(t):
            return [[[t[b][s][h] for s in range(len(t[b]))]
                     for h in range(len(t[b][0]))] for b in range(len(t))]
        for causal in (False, True):
            for requested_scale in (-1.0, 0.125):
                scale = 1/math.sqrt(3) if requested_scale <= 0 else requested_scale
                expected = []
                for b in range(2):
                    sequences = []
                    for i in range(2):
                        heads = []
                        for hq in range(4):
                            hk = hq//2
                            scores = [scale*(sum(qn[b][i][hq][d]*kn[b][j][hk][d] for d in range(2))
                                             + qr[b][i][hq][0]*kr[b][j][hk][0]) for j in range(5)]
                            heads.append(dense(scores, [kn[b][j][hk] for j in range(5)],
                                               [not causal or j <= i+3 for j in range(5)]))
                        sequences.append(heads)
                    expected.append(sequences)
                for layout in ("BSND", "BNSD"):
                    inputs = [qn, qr, kn, kr, deepcopy(kn)]
                    if layout == "BNSD":
                        inputs = [bnsd(t) for t in inputs]
                    result = mla_dense(*inputs, numKVHeads=2, scaleValue=requested_scale,
                                       inputLayout=layout, is_causal=causal, block_kv=2)
                    self.close_nested(result, expected if layout == "BSND" else bnsd(expected))
        wrong_v = deepcopy(kn)
        wrong_v[0][0][0][0] += 1
        with self.assertRaisesRegex(ValueError, "equal K_nope by value"):
            mla_dense(qn, qr, kn, kr, wrong_v, numKVHeads=2)


class PipelineTests(unittest.TestCase):
    def test_equal_phase_manual_pipe_order_beats_default_stage_priority(self):
        schedule = Schedule.resolve([
            Task("first", "AIV:VECTOR", 0, authored_stage=0),
            Task("second", "AIV:VECTOR", 10, authored_stage=1),
        ], 10)
        self.assertEqual([r.task.name for r in schedule.tasks], ["first", "second"])
        self.assertEqual([r.stage for r in schedule.tasks], [0, 1])

    @staticmethod
    def schedule():
        return Schedule.resolve([
            Task("LX", "MTE2", 0, ("X",), 0), Task("LZ", "MTE2", 40, ("Z",), 0),
            Task("VF", "VECTOR", 80, ("X", "Z", "Y"), 1), Task("ST", "MTE3", 160, ("Y",), 2),
        ], 80)

    def test_phase_order_and_bounded_symbolic_body(self):
        schedule = self.schedule()
        self.assertEqual([r.task.name for r in schedule.tasks], ["ST", "VF", "LX", "LZ"])
        self.assertEqual([r.phase for r in schedule.tasks], [0, 0, 0, 40])
        template = PipelineTemplate(Loop(0, Symbol("N")), schedule, (("X", 2), ("Z", 2), ("Y", 2)))
        self.assertEqual(len(template.schedule.tasks), 4)
        self.assertIn("ceil_div(N - (0), 1)", template.render())
        for count in (0, 1, 2, 3, 7):
            instances = list(template.trace({"N": count}))
            self.assertEqual(len(instances), 4 * count)
            self.assertEqual({(i.task, i.ordinal) for i in instances},
                             {(name, k) for name in ("LX", "LZ", "VF", "ST") for k in range(count)})
            for instance in instances:
                for _, slot, generation in instance.slots:
                    self.assertEqual((slot, generation), (instance.ordinal % 2, instance.ordinal))

    def test_nonzero_lower_and_step(self):
        template = PipelineTemplate(Loop(Symbol("A"), 12, 2), self.schedule(), (("X", 2), ("Z", 2), ("Y", 2)))
        trace = list(template.trace({"A": 5}))
        self.assertEqual([i.source_index for i in trace if i.task == "ST"], [5, 7, 9, 11])

    def test_invalid_assignments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "stage violated"):
            Schedule.resolve([Task("A", "V", 80, authored_stage=0)], 80)
        with self.assertRaisesRegex(ValueError, "offset=False"):
            Schedule.resolve([Task("A", "V", 80)], 80, enable_offset=False)
        with self.assertRaisesRegex(ValueError, "same-pipe"):
            Schedule.resolve([Task("A", "V", 40, authored_stage=0), Task("B", "V", 80, authored_stage=1)], 80)
        with self.assertRaisesRegex(ValueError, "missing resolved"):
            PipelineTemplate(Loop(0, 2), self.schedule(), (("X", 2), ("Z", 2)))


class CompletionTests(unittest.TestCase):
    def test_valid_generation_reuse_and_exit(self):
        protocol = RingProtocol([Buffer("V", 2, ("score", "acc"))])
        for g in range(5):
            for kind, who in (("write_begin", ""), ("ready", ""), ("read_begin", "score"),
                              ("read_done", "score"), ("read_begin", "acc"), ("read_done", "acc"), ("free", "")):
                protocol.accept(Event(kind, "V", g % 2, g, who))
        protocol.finish()
        with self.assertRaisesRegex(ValueError, "stale"):
            protocol.accept(Event("write_begin", "V", 0, 0))
        RingProtocol([Buffer("empty_loop", 2, ("compute",))]).finish()

    def test_score_completion_does_not_release_value(self):
        protocol = RingProtocol([Buffer("V", 2, ("score", "acc"))])
        for kind, who in (("write_begin", ""), ("ready", ""), ("read_begin", "score"), ("read_done", "score")):
            protocol.accept(Event(kind, "V", 0, 0, who))
        with self.assertRaisesRegex(ValueError, "last consumer"):
            protocol.accept(Event("free", "V", 0, 0))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            protocol.accept(Event("write_begin", "V", 0, 2))
        with self.assertRaisesRegex(ValueError, "live asynchronous"):
            protocol.finish()

    def test_wrong_generation_and_nonexistent_wait(self):
        protocol = RingProtocol([Buffer("P", 2, ("PV",))])
        with self.assertRaisesRegex(ValueError, "nonexistent"):
            protocol.accept(Event("read_begin", "P", 0, 0, "PV"))
        protocol.accept(Event("write_begin", "P", 0, 0))
        with self.assertRaisesRegex(ValueError, "wrong or nonexistent"):
            protocol.accept(Event("ready", "P", 0, 2))
        with self.assertRaisesRegex(ValueError, "slot/generation"):
            protocol.accept(Event("ready", "P", 1, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
