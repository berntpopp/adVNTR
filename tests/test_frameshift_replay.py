"""Pure replay verifies original receipts before considering a different policy."""
import json
import unittest

from advntr.frameshift_background import BackgroundModel
from tests.test_frameshift_capture_record import capture_document
from tests.test_frameshift_replay_policy import replay_policy


class TestFrameshiftReplay(unittest.TestCase):
    def test_baseline_and_changed_support_use_the_same_frozen_source_capture(self):
        from advntr.frameshift_replay import replay_capture
        record = capture_document()
        baseline = replay_capture(record, replay_policy())
        changed = replay_capture(record, replay_policy(support=5))
        self.assertTrue(baseline['baseline_parity'])
        self.assertEqual(17, baseline['vntr_id'])
        self.assertEqual([{'state': 'D2_1', 'read_support': 4, 'mean_coverage': 1.0, 'pvalue': 0.0}], baseline['calls'])
        self.assertEqual([], changed['calls'])
        self.assertEqual('insufficient-read-support', changed['decision_visits'][0]['disposition'])
        self.assertNotEqual(baseline['policy_sha256'], changed['policy_sha256'])
        self.assertEqual(baseline['capture_record_sha256'], changed['capture_record_sha256'])
        self.assertEqual(['minimum_read_match_ratio'], baseline['capture_audit']['calibrated_policy_domain_errors'])
        json.dumps(changed, allow_nan=False)

    def test_candidate_exact_background_does_not_replace_the_original_baseline(self):
        from advntr.frameshift_replay import replay_capture
        background = BackgroundModel(1, 'invented only', 0.5, {}, 'never-read')
        result = replay_capture(capture_document(), replay_policy('exact'), background)
        self.assertTrue(result['baseline_parity'])
        self.assertEqual([], result['calls'])
        self.assertEqual(0.875, result['decision_visits'][0]['statistic']['pvalue'])
        self.assertIsNotNone(result['loaded_background_sha256'])

    def test_replay_refuses_tampered_baseline_capture_changes_and_background_mismatch(self):
        from advntr.frameshift_replay import replay_capture
        bad = capture_document()
        bad['decision_visits'][0]['statistic']['called'] = False
        with self.assertRaises(ValueError):
            replay_capture(bad, replay_policy(support=5))
        changed = replay_policy()
        changed['capture_policy']['parameters']['prune_reverse'] = True
        with self.assertRaisesRegexp(ValueError, 'recapture'):
            replay_capture(capture_document(), changed)
        for mode, background in (('exact', None), ('legacy', object()), ('exact', object())):
            with self.assertRaises(ValueError):
                replay_capture(capture_document(), replay_policy(mode), background)
