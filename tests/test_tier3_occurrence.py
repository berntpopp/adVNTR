"""Tier 3: the gate for any change to the read loop.

Runs the real `select_illumina_reads` and compares what it returned, in order. This is
the only tier that can see occurrence loss, reordering, or metadata mis-association --
the failure modes threading introduces. It is deliberately slow and is skipped unless
VNtyper's corpus is present.
"""
import json
import os
import unittest

from advntr_harness.capture import build_finder
from advntr_harness.tier3 import selection_digest, selection_evidence

DATA = '/home/bernt-popp/development/VNtyper/tests/data'
GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
DB = os.path.join(GOLDEN, 'models', 'hg19_muc1.db')
BAM = os.path.join(DATA, 'example_7a61_hg19_subset.bam')
MANIFEST = os.path.join(GOLDEN, 'tier3_manifest.json')

#: Thread counts every read-loop change must reproduce. 1 is the reference.
THREAD_COUNTS = (1, 2, 4, 8, 16)

has_corpus = unittest.skipUnless(os.path.isfile(BAM),
                                 'VNtyper corpus not available')
has_baseline = unittest.skipUnless(os.path.isfile(MANIFEST),
                                   'Tier 3 baseline not captured yet')


@has_corpus
class TestTier3Occurrence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        finder, _reference = build_finder(DB)
        cls.evidence = selection_evidence(finder, BAM)

    def test_selection_is_not_empty(self):
        """Guards against a gate that passes by selecting nothing."""
        self.assertGreater(self.evidence['count'], 500)

    def test_digest_is_not_the_empty_sentinel(self):
        self.assertNotEqual(self.evidence['digest'], 'EMPTY')

    def test_query_names_are_unique_per_occurrence_slot(self):
        """Duplicate query names are legal (read pairs); the ORDER is what must hold."""
        self.assertEqual(len(self.evidence['query_names']), self.evidence['count'])

    def test_model_is_the_one_production_rebuilt(self):
        """select_illumina_reads overwrites its hmm argument, so this is 2565 (read
        length 151 derived from the BAM), not 2559."""
        self.assertEqual(self.evidence['model_states'], 2565)

    @has_baseline
    def test_matches_the_captured_baseline(self):
        with open(MANIFEST) as handle:
            baseline = json.load(handle)
        self.assertEqual(self.evidence['count'], baseline['count'])
        self.assertEqual(self.evidence['digest'], baseline['digest'])


@has_corpus
@has_baseline
class TestTier3ThreadInvariance(unittest.TestCase):
    """Every thread count must return the identical ordered selection.

    Skipped while `settings.CORES` is still inert on this path -- running five identical
    single-threaded passes would burn ~15 minutes proving nothing. It becomes meaningful
    the moment the read loop honours CORES, and `test_threading_is_actually_wired`
    below is what flips it on.
    """

    @unittest.skipUnless(os.environ.get('ADVNTR_THREADED') == '1',
                         'read loop does not honour settings.CORES yet; '
                         'set ADVNTR_THREADED=1 once it does')
    def test_every_thread_count_returns_the_same_ordered_selection(self):
        finder, _reference = build_finder(DB)
        reference = selection_evidence(finder, BAM, threads=1)
        for threads in THREAD_COUNTS[1:]:
            finder, _reference_vntr = build_finder(DB)
            actual = selection_evidence(finder, BAM, threads=threads)
            self.assertEqual(actual['count'], reference['count'],
                             'read count changed at -t %d' % threads)
            self.assertEqual(actual['query_names'], reference['query_names'],
                             'read ORDER changed at -t %d' % threads)
            self.assertEqual(actual['digest'], reference['digest'],
                             'selection diverged at -t %d' % threads)


if __name__ == '__main__':
    unittest.main()
