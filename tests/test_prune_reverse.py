"""Task 8's `--prune-reverse`: the real-kernel evidence.

`tests/test_read_selection.py::TestPruneReverseFlag` covers `_decode_one`'s CONTROL FLOW
against a stub. This module is the other half: the compiled kernel actually does less
work under pruning, and `select_illumina_reads` returns the identical selected-read
stream with the flag on or off. Both need the public VNtyper corpus and are skipped
without it, mirroring `tests/test_tier3_occurrence.py`'s `has_corpus` idiom.

No cohort data: every number here comes from decoding one of the public `example_*`
corpus BAMs already named elsewhere in this repo (AGENTS.md, tests/test_tier3_occurrence.py).
"""
import os
import unittest

from Bio.Seq import Seq

from advntr import settings
from advntr_harness.capture import _ModelCache, build_finder
from advntr_harness.extract import eligible_reads
from advntr_harness.tier3 import selection_evidence
from advntr_harness.workload import decode_with_counters

DATA = '/home/bernt-popp/development/VNtyper/tests/data'
GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
MODELS = os.path.join(GOLDEN, 'models')
DB = os.path.join(MODELS, 'hg19_muc1.db')
BAM = os.path.join(DATA, 'example_7a61_hg19_subset.bam')

#: Sized for a fast gate run (~2s of instrumented decoding; measured on this machine),
#: not for a full-corpus figure -- see task-8-report.md for the full-file measurement.
SAMPLE_SIZE = 300

has_corpus = unittest.skipUnless(os.path.isfile(BAM), 'VNtyper corpus not available')


@has_corpus
class TestReverseDecodeRelaxationsDropUnderPruning(unittest.TestCase):
    """The claim behind `--prune-reverse`: for a read the forward decode already wins
    decisively -- the overwhelming majority of the corpus, since AGENTS.md measures the
    reverse-complement stratum at 77 of 29,998 reads -- flooring the reverse decode's DP
    threshold at the forward logp prunes away most of its work. Measured per-read rather
    than as one pooled sum: a raw sum is dominated by the minority of reads with a weak
    forward match (their full decode is enormous AND barely shrinks under pruning), which
    would hide the typical read's outcome behind a handful of outliers.
    """

    @classmethod
    def setUpClass(cls):
        cache = _ModelCache(MODELS)
        cls.model, _fingerprint, _score = cache.get('hg19', 151)
        _finder, reference = cache.finder('hg19')
        sequences = [read[2] for read in
                    eligible_reads(BAM, reference, read_length=151)]
        cls.sequences = sequences[:SAMPLE_SIZE]

    def _reverse_relaxation_ratio(self, sequence):
        fwd_logp, _fwd_vpath = self.model.viterbi(sequence)
        reverse = str(Seq(sequence).reverse_complement()).upper()
        full = decode_with_counters(self.model, reverse)
        pruned = decode_with_counters(self.model, reverse, min_threshold=fwd_logp)
        self.assertGreater(full['edge_relaxations'], 0)
        return pruned['edge_relaxations'] / float(full['edge_relaxations'])

    def test_the_median_read_sees_a_relaxation_drop_of_at_least_ninety_percent(self):
        """Median, not mean: robust to the few reads a weak forward match makes
        expensive either way. Measured on this sample: median ratio 0.044 (95.6% drop),
        comfortably inside the 0.10 (90% drop) bound asserted here."""
        ratios = sorted(self._reverse_relaxation_ratio(seq) for seq in self.sequences)
        n = len(ratios)
        median = ratios[n // 2] if n % 2 else (ratios[n // 2 - 1] + ratios[n // 2]) / 2.0
        self.assertLessEqual(
            median, 0.10,
            'median reverse-decode relaxation ratio %.4f exceeds the 90%% drop bound' % median)

    def test_pruning_never_increases_relaxations(self):
        """A tighter threshold can only prune more work, never less -- the DP threshold
        check is the same `>= threshold` comparator either way, just at a higher floor."""
        for sequence in self.sequences[:50]:
            fwd_logp, _fwd_vpath = self.model.viterbi(sequence)
            reverse = str(Seq(sequence).reverse_complement()).upper()
            full = decode_with_counters(self.model, reverse)
            pruned = decode_with_counters(self.model, reverse, min_threshold=fwd_logp)
            self.assertLessEqual(pruned['edge_relaxations'], full['edge_relaxations'])


@has_corpus
class TestSelectedReadDigestIsUnchangedByPruning(unittest.TestCase):
    """Tier B's identical-genotype-calls burden (AGENTS.md's two-tier rule), discharged
    directly against `advntr_harness.tier3.selection_evidence` -- the same machinery
    `tests/test_tier3_occurrence.py` uses for thread-invariance -- so this is exactly
    the equivalence claim the gate already trusts, just varied over the flag instead of
    the thread count.
    """

    def setUp(self):
        self._original_flag = settings.PRUNE_REVERSE_DECODE
        self.addCleanup(setattr, settings, 'PRUNE_REVERSE_DECODE', self._original_flag)

    def test_flag_on_matches_flag_off_on_a_reverse_complement_wins_source(self):
        """example_7a61 is one of the three files that together supply every one of the
        corpus's 77 reverse-complement-wins reads (advntr_harness/capture.py's
        TIER1_FILES comment) -- the population most likely to exercise the safety valve,
        not just the pruning path."""
        settings.PRUNE_REVERSE_DECODE = False
        finder_off, _reference = build_finder(DB)
        off = selection_evidence(finder_off, BAM)

        settings.PRUNE_REVERSE_DECODE = True
        finder_on, _reference = build_finder(DB)
        on = selection_evidence(finder_on, BAM)

        self.assertEqual(off['count'], on['count'])
        self.assertEqual(off['query_names'], on['query_names'])
        self.assertEqual(off['digest'], on['digest'])


if __name__ == '__main__':
    unittest.main()
