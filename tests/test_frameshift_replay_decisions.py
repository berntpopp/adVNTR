"""Replayed policy changes use the original candidate traversal and native gates."""
import unittest

from advntr.frameshift_traversal import freeze_traversal
from advntr.frameshift_replay_policy import decode_replay_policy
from tests.test_frameshift_replay_policy import replay_policy


class TestReplayDecisions(unittest.TestCase):
    def test_support_change_reaches_a_previously_unvisited_second_flank_branch(self):
        from advntr.frameshift_replay_decisions import evaluate_visits
        traversal = freeze_traversal({}, {'I5_prefix_suffix': 2})
        geometry = {'1': {'sequence': 'ACGT', 'reference_copies': 1, 'ru_bp_coverage': 8}}
        original = evaluate_visits(traversal, ['L', '1', 'R'], (0, 10), geometry, {},
                                   decode_replay_policy(replay_policy()), None)
        changed = evaluate_visits(traversal, ['L', '1', 'R'], (0, 10), geometry, {},
                                  decode_replay_policy(replay_policy(support=2)), None)
        self.assertEqual(['insufficient-read-support'], [visit.disposition for visit in original])
        self.assertEqual(['suffix', 'prefix'], [visit.plan.site for visit in changed])
        self.assertEqual(['called', 'called'], [visit.disposition for visit in changed])
        self.assertEqual([1.0, 1.0], [visit.mean_coverage for visit in changed])

    def test_empty_traversal_and_boundary_suppression_do_not_score_evidence_rows(self):
        from advntr.frameshift_replay_decisions import evaluate_visits
        policy = decode_replay_policy(replay_policy())
        self.assertEqual((), evaluate_visits(freeze_traversal({}, {}), [], (0, 0), {}, {'unused': {}}, policy, None))
        visits = evaluate_visits(freeze_traversal({}, {'I1_prefix_LEN1': 4}), [], (0, 0), {}, {}, policy, None)
        self.assertEqual('outside-boundary', visits[0].disposition)
        self.assertIsNone(visits[0].statistic)

    def test_background_presence_cannot_silently_select_another_caller(self):
        from advntr.frameshift_replay_decisions import evaluate_visits
        for mode, background in (('exact', None), ('legacy', object())):
            with self.assertRaises(ValueError):
                evaluate_visits(freeze_traversal({}, {}), [], (0, 0), {}, {},
                                decode_replay_policy(replay_policy(mode)), background)
