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
from advntr_harness.tier3 import BASELINE_MANIFEST_KEYS, selection_digest, selection_evidence

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

    Order matters as much as content: `hmm_alignment.generate_aln` parses an unframed
    DEBUG stream, and tied mutations are sorted by count alone, so a permuted selection
    can change the reported result without changing any score.
    """

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




TIER1_MANIFEST = os.path.join(GOLDEN, 'tier1_manifest.json')

#: hmm/hmm.pyx at 05fd98a. Tier 1's baseline records this, which is what makes it a
#: pristine gate rather than a self-comparison.
PRISTINE_KERNEL_DIGEST = 'e87fabf5e8633235'


class TestTheBaselineSaysWhatItIs(unittest.TestCase):
    """The Tier 3 manifest was `{"count": ..., "digest": ...}` and nothing else.

    It was added by 82b1c2b -- the threading commit itself -- so nothing in the tree
    established which kernel produced it, and a reader could reasonably take it for a
    pristine baseline like Tier 1's. It is not one. Recording that is the difference
    between a gate and a gate believed to be stronger than it is.

    These read files only, so they run without the corpus.
    """

    @has_baseline
    def test_the_baseline_records_the_kernel_that_produced_it(self):
        with open(MANIFEST) as handle:
            baseline = json.load(handle)

        self.assertIn('kernel_provenance', baseline)
        self.assertIn('hmm/hmm.pyx', baseline['kernel_provenance'])
        self.assertNotEqual(baseline['kernel_provenance']['hmm/hmm.pyx'], 'absent')

    @has_baseline
    def test_the_baseline_does_not_claim_to_be_pristine(self):
        with open(MANIFEST) as handle:
            baseline = json.load(handle)

        self.assertEqual(baseline['baseline_kind'], 'post-rewrite regression baseline')
        self.assertNotEqual(baseline['kernel_provenance']['hmm/hmm.pyx'],
                            PRISTINE_KERNEL_DIGEST)
        self.assertIn('Tier 1 is the pristine gate', baseline['note'])

    @has_baseline
    def test_the_baseline_names_the_file_it_was_captured_from(self):
        """A count and a digest mean nothing without the input that produced them."""
        with open(MANIFEST) as handle:
            baseline = json.load(handle)

        self.assertEqual(baseline['source_file'], os.path.basename(BAM))
        self.assertEqual(baseline['model_states'], 2565)

    @has_baseline
    def test_the_shipped_manifest_carries_exactly_what_its_producer_writes(self):
        """`baseline_manifest` is the only record of how this artefact was made, and until
        it had a caller nothing noticed if the two drifted. Key-set equality is cheap and
        needs no corpus; the values are checked by the comparison test above."""
        with open(MANIFEST) as handle:
            baseline = json.load(handle)

        self.assertEqual(sorted(baseline), sorted(BASELINE_MANIFEST_KEYS))

    @unittest.skipUnless(os.path.isfile(TIER1_MANIFEST), 'Tier 1 baseline not present')
    def test_tier_one_by_contrast_does_record_the_pristine_kernel(self):
        """The contrast is the point: this is what a pristine baseline looks like, and it
        is why the equivalence claim rests on Tier 1 and not on this tier."""
        with open(TIER1_MANIFEST) as handle:
            tier1 = json.load(handle)

        self.assertEqual(tier1['kernel_provenance']['hmm/hmm.pyx'], PRISTINE_KERNEL_DIGEST)


if __name__ == '__main__':
    unittest.main()
