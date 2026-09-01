"""The Tier 2 baseline is the pristine case, and says so.

`tests/golden/tier3_manifest.json` used to be `{"count": ..., "digest": ...}` and
nothing else, so nothing in the tree said which kernel produced it -- a reader could
mistake it for a pristine baseline (advntr_harness/tier3.py:101-119; FORK.md's 2.0.2/
2.0.3 entries are that mistake's fallout). Tier 3 now states plainly that it is NOT
pristine. This is the opposite case: `tests/golden/tier2_manifest.json` genuinely was
captured from this fork's pristine kernel (05fd98a), and these tests pin that claim down
instead of leaving it to be re-derived from kernel_provenance digests alone.

All of this reads a committed JSON file -- no corpus, no decode, seconds to run.
"""
import json
import os
import unittest

from advntr_harness.capture import (BASELINE_KIND, CAPTURE_MANIFEST_KEYS, TIER2_FILES)

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
MANIFEST = os.path.join(GOLDEN, 'tier2_manifest.json')

#: hmm/hmm.pyx at 05fd98a. Matching this is what makes the baseline pristine rather than
#: a self-comparison -- see tests/test_tier3_occurrence.py's own copy of this constant.
PRISTINE_KERNEL_DIGEST = 'e87fabf5e8633235'

has_baseline = unittest.skipUnless(os.path.isfile(MANIFEST),
                                   'Tier 2 baseline not captured yet')


class TestFixtureFileExists(unittest.TestCase):
    """Not `@has_baseline` -- deliberately unconditional (mirrors
    tests/test_orientation.py::TestFixtureFileExists). Every `@has_baseline` class
    below skips without complaint if `tests/golden/tier2_manifest.json` is
    missing -- the right idiom for a corpus artefact that may not have been
    captured yet, but this manifest is COMMITTED (c1981d4), so `skipUnless` alone
    would let its deletion pass silently across all 5 tests below instead of
    failing anywhere. This test has no skip decorator, so that absence fails
    here.
    """

    def test_the_committed_baseline_manifest_is_present(self):
        self.assertTrue(
            os.path.isfile(MANIFEST),
            '%s is missing. This is a COMMITTED fixture (c1981d4), not an '
            'external artefact yet to be captured -- its absence means it was '
            'deleted. Regenerate with `make tier2` (or `python -m '
            'advntr_harness.capture --tier 2 --out tests/golden --verify '
            'tests/golden`) and `git add` it back; every `@has_baseline` class '
            'in this module silently skips without it.' % MANIFEST)


@has_baseline
class TestTheBaselineSaysWhatItIs(unittest.TestCase):
    def setUp(self):
        with open(MANIFEST) as handle:
            self.baseline = json.load(handle)

    def test_the_baseline_declares_itself_pristine(self):
        self.assertEqual(self.baseline['baseline_kind'], BASELINE_KIND)
        self.assertIn('pristine', self.baseline['note'])
        self.assertIn('05fd98a', self.baseline['note'])

    def test_the_baseline_records_the_pristine_kernel_digest(self):
        """Unlike Tier 3's baseline, this one really does match 05fd98a -- see
        tests/test_tier3_occurrence.py's test_tier_one_by_contrast_does_record_the_
        pristine_kernel for the mirror image of this assertion."""
        self.assertEqual(self.baseline['kernel_provenance']['hmm/hmm.pyx'],
                         PRISTINE_KERNEL_DIGEST)

    def test_the_shipped_manifest_carries_exactly_what_its_producer_writes(self):
        """Key-set equality against CAPTURE_MANIFEST_KEYS so the artefact and
        advntr_harness.capture.capture() cannot silently drift apart."""
        self.assertEqual(sorted(self.baseline), sorted(CAPTURE_MANIFEST_KEYS))

    def test_every_source_file_is_one_of_the_named_public_corpus_files(self):
        """The data rule is absolute: nothing but the eight public example_* BAMs may
        enter the repo. This is the cheap, permanent guard against that regressing."""
        names = set(entry['source_file'] for entry in self.baseline['files'])
        self.assertTrue(names)
        self.assertTrue(names.issubset(set(TIER2_FILES)),
                        'unexpected source file(s): %s' % sorted(names - set(TIER2_FILES)))

    def test_the_corpus_is_substantial(self):
        """Guards against a baseline that goes green by covering almost nothing."""
        total_attempts = sum(entry.get('attempt_count', 0) for entry in self.baseline['files'])
        self.assertEqual(len(self.baseline['files']), 8)
        self.assertGreater(total_attempts, 250000)


if __name__ == '__main__':
    unittest.main()
