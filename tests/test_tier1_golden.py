"""Tier 1 equivalence gate.

Decodes the committed fixture sequences and requires byte-identical decode-attempt rows
against the baseline captured from pristine 05fd98a.

The expected rows are stored under *canonical* identities (source `tier1`, ordinal =
position in the fixture list, name `fixtureN`) precisely so this gate can reproduce them
from the sequences alone. Storing the original file/ordinal/query-name would make the
gate unable to regenerate row zero, let alone compare it.
"""
import gzip
import json
import os
import unittest

from advntr_harness.capture import (_ModelCache, canonical_fixture_rows,
                                    read_fixture_file)
from advntr_harness.fingerprint import comparable_fingerprint
from advntr_harness.strata import STRATUM_NAMES

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
MODELS = os.path.join(GOLDEN, 'models')
READS = os.path.join(GOLDEN, 'tier1_reads.txt.gz')
EXPECTED = os.path.join(GOLDEN, 'tier1_expected.tsv.gz')
MANIFEST = os.path.join(GOLDEN, 'tier1_manifest.json')

has_fixtures = unittest.skipUnless(
    os.path.isfile(READS) and os.path.isfile(EXPECTED) and os.path.isfile(MANIFEST),
    'Tier 1 fixtures not captured yet '
    '(python -m advntr_harness.capture --tier 1 --out tests/golden)')


@has_fixtures
class TestTier1Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = read_fixture_file(READS)
        cls.sequences = [sequence for _key, sequence in cls.fixtures]
        with gzip.open(EXPECTED) as handle:
            cls.expected = handle.read().split('\n')
        with open(MANIFEST) as handle:
            cls.manifest = json.load(handle)

    def test_fixture_set_is_substantial(self):
        self.assertGreater(len(self.sequences), 1000)

    def test_expected_rows_are_two_per_fixture(self):
        """One per orientation. A mismatch means capture and gate disagree on shape."""
        self.assertEqual(len(self.expected), 2 * len(self.sequences))

    def test_every_stratum_is_populated(self):
        """An empty stratum makes the gate vacuous for whatever it was meant to catch."""
        strata = self.manifest['strata']
        for name in STRATUM_NAMES:
            self.assertIn(name, strata)
            self.assertGreater(strata[name], 0, 'stratum %r is empty' % name)

    def test_the_reverse_complement_stratum_is_present(self):
        """The stratum upstream PR #57 would delete. Measured at 0.26% of reads, and
        exactly zero in example_66bf -- so its presence here is not automatic."""
        self.assertGreater(self.manifest['strata']['reverse_complement_wins'], 0)

    def test_every_fixture_model_matches_the_baseline(self):
        cache = _ModelCache(MODELS)
        for model_key, expected in self.manifest['fixture_models'].items():
            assembly, read_length = model_key.split('@')
            _model, fingerprint, _score = cache.get(assembly, int(read_length))
            self.assertEqual(comparable_fingerprint(fingerprint),
                             comparable_fingerprint(expected),
                             'model %s changed' % model_key)

    def test_fixtures_span_more_than_one_model_context(self):
        """Read length is derived per file and the corpus disagrees: 7a61/b178 give
        151 (2565 states), a5c1 gives 149 (2553). Binding fixtures to a single model
        would mislabel most of the reverse-complement stratum."""
        self.assertGreater(len(set(key for key, _ in self.fixtures)), 1)

    def test_decoding_is_byte_identical_to_the_baseline(self):
        cache = _ModelCache(MODELS)
        actual = canonical_fixture_rows(cache, self.fixtures)

        self.assertEqual(len(actual), len(self.expected))
        for index, (got, want) in enumerate(zip(actual, self.expected)):
            if got != want:
                # Report where it diverged, not just that it did.
                got_fields, want_fields = got.split('\t'), want.split('\t')
                differing = [name for name, a, b in zip(
                    ('source', 'ordinal', 'query', 'orientation', 'sequence',
                     'logp', 'vpath', 'status', 'decision'), got_fields, want_fields)
                    if a != b]
                self.fail('row %d diverged in %s\n  expected %s\n  actual   %s'
                          % (index, differing, want[:160], got[:160]))


if __name__ == '__main__':
    unittest.main()
