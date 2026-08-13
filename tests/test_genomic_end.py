"""The genomic end of a VNTR is a coordinate, not a sum of stored segment lengths.

`get_length()` returns the total length of the stored repeat units. That equals the
array's genomic extent only when the segments tile the array end to end exactly once.
The shipped MUC1 hg38 model does not: it stores 14 units totalling 840 bp for an array
of 3,525 bp, so the read-fetch window covered 24% of the locus.

These tests pin the separation: an explicit `ref_end` when the model records one, and
the legacy sum when it does not, so existing databases keep their current behaviour.

See berntpopp/adVNTR#1 and hassansaei/VNtyper#268.
"""
import os
import shutil
import sqlite3
import tempfile
import unittest

from advntr.models import load_unique_vntrs_data
from advntr.reference_vntr import ReferenceVNTR


LEGACY_SCHEMA = (
    'CREATE TABLE vntrs(id INTEGER PRIMARY KEY, nonoverlapping TEXT, '
    'chromosome TEXT, ref_start INTEGER, gene_name TEXT, annotation TEXT, '
    'pattern TEXT, left_flanking TEXT, right_flanking TEXT, repeats TEXT, '
    'scaled_score REAL DEFAULT 0)'
)

V2_SCHEMA = (
    'CREATE TABLE vntrs(id INTEGER PRIMARY KEY, nonoverlapping TEXT, '
    'chromosome TEXT, ref_start INTEGER, gene_name TEXT, annotation TEXT, '
    'pattern TEXT, left_flanking TEXT, right_flanking TEXT, repeats TEXT, '
    'scaled_score REAL DEFAULT 0, ref_end INTEGER)'
)

# The shipped v2 model lives in its own table so that an adVNTR which does not
# understand ref_end cannot read it at all. See FailsClosedForOldReaders.
V2_TABLE_SCHEMA = V2_SCHEMA.replace('CREATE TABLE vntrs(', 'CREATE TABLE vntrs_v2(')

SEGMENTS = ['A' * 60, 'C' * 60, 'G' * 48]
SUM_OF_SEGMENTS = 168
START = 1000


def _make_vntr(ref_end=None):
    vntr = ReferenceVNTR(25561, 'A' * 60, START, 'chr1', 'MUC1', 'Coding',
                         estimated_repeats=len(SEGMENTS))
    vntr.init_from_xml(SEGMENTS, 'L' * 500, 'R' * 500)
    if ref_end is not None:
        vntr.ref_end = ref_end
    return vntr


def _write_db(path, schema, row):
    table = schema.split('CREATE TABLE ')[1].split('(')[0]
    db = sqlite3.connect(path)
    db.execute(schema)
    db.execute('insert into %s values (%s)' % (table, ','.join(['?'] * len(row))), row)
    db.commit()
    db.close()


class GenomicEndFallback(unittest.TestCase):
    """A model without an explicit end keeps the behaviour it has today."""

    def test_falls_back_to_the_legacy_sum(self):
        vntr = _make_vntr()
        self.assertEqual(vntr.get_genomic_end(), START + SUM_OF_SEGMENTS)

    def test_legacy_sum_is_still_what_get_length_returns(self):
        # get_length() keeps its meaning: total length of the stored units. The fix
        # stops using it as a coordinate; it does not redefine it.
        self.assertEqual(_make_vntr().get_length(), SUM_OF_SEGMENTS)

    def test_explicit_end_is_used_when_recorded(self):
        vntr = _make_vntr(ref_end=START + 3525)
        self.assertEqual(vntr.get_genomic_end(), START + 3525)

    def test_explicit_end_is_independent_of_segment_lengths(self):
        # The whole point: the two quantities are allowed to disagree.
        vntr = _make_vntr(ref_end=START + 3525)
        self.assertNotEqual(vntr.get_genomic_end(), START + vntr.get_length())


class DatabaseLoading(unittest.TestCase):
    """Old databases load unchanged; new ones carry the end through."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _row(self, extra=()):
        return (25561, 'True', 'chr1', START, 'MUC1', 'Coding', 'A' * 60,
                'L' * 500, 'R' * 500, ','.join(SEGMENTS), 0.0) + tuple(extra)

    def test_database_without_ref_end_loads_and_falls_back(self):
        path = os.path.join(self.tmp, 'legacy.db')
        _write_db(path, LEGACY_SCHEMA, self._row())
        vntr = load_unique_vntrs_data(path)[0]
        self.assertIsNone(vntr.ref_end)
        self.assertEqual(vntr.get_genomic_end(), START + SUM_OF_SEGMENTS)

    def test_database_with_ref_end_uses_it(self):
        path = os.path.join(self.tmp, 'v2.db')
        _write_db(path, V2_SCHEMA, self._row(extra=(START + 3525,)))
        vntr = load_unique_vntrs_data(path)[0]
        self.assertEqual(vntr.ref_end, START + 3525)
        self.assertEqual(vntr.get_genomic_end(), START + 3525)

    def test_null_ref_end_is_treated_as_absent(self):
        # A v2 schema whose column was never populated must not become end=0.
        path = os.path.join(self.tmp, 'v2null.db')
        _write_db(path, V2_SCHEMA, self._row(extra=(None,)))
        vntr = load_unique_vntrs_data(path)[0]
        self.assertIsNone(vntr.ref_end)
        self.assertEqual(vntr.get_genomic_end(), START + SUM_OF_SEGMENTS)


class FailsClosedForOldReaders(unittest.TestCase):
    """A v2 model must not be silently misread by an adVNTR that ignores ref_end.

    2.0.3 selects the eleven legacy columns by name, so an extra `ref_end` column on
    the `vntrs` table would be dropped without a word and the old 840 bp window
    recreated -- wrong answers, no error. Putting v2 models in their own table makes
    that read fail loudly instead.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _row(self, extra=()):
        return (25561, 'True', 'chr1', START, 'MUC1', 'Coding', 'A' * 60,
                'L' * 500, 'R' * 500, ','.join(SEGMENTS), 0.0) + tuple(extra)

    def test_the_legacy_query_cannot_read_a_v2_database(self):
        path = os.path.join(self.tmp, 'v2.db')
        _write_db(path, V2_TABLE_SCHEMA, self._row(extra=(START + 3525,)))
        db = sqlite3.connect(path)
        self.assertRaises(sqlite3.OperationalError,
                          db.execute, 'SELECT id FROM vntrs')

    def test_the_new_loader_reads_a_v2_database(self):
        path = os.path.join(self.tmp, 'v2.db')
        _write_db(path, V2_TABLE_SCHEMA, self._row(extra=(START + 3525,)))
        vntr = load_unique_vntrs_data(path)[0]
        self.assertEqual(vntr.get_genomic_end(), START + 3525)

    def test_the_new_loader_still_reads_a_legacy_database(self):
        path = os.path.join(self.tmp, 'legacy.db')
        _write_db(path, LEGACY_SCHEMA, self._row())
        vntr = load_unique_vntrs_data(path)[0]
        self.assertEqual(vntr.get_genomic_end(), START + SUM_OF_SEGMENTS)

    def test_a_database_with_neither_table_says_so(self):
        path = os.path.join(self.tmp, 'empty.db')
        sqlite3.connect(path).execute('CREATE TABLE unrelated(x INTEGER)')
        self.assertRaises(ValueError, load_unique_vntrs_data, path)


class VersionHandshake(unittest.TestCase):
    """A consumer cannot require >=2.0.4 unless the binary can state its version."""

    def test_package_exposes_a_version(self):
        import advntr
        self.assertRegexpMatches(advntr.__version__, r'^\d+\.\d+\.\d+$')

    def test_version_is_at_least_the_span_aware_release(self):
        import advntr
        parts = tuple(int(p) for p in advntr.__version__.split('.'))
        self.assertGreaterEqual(parts, (2, 0, 4))


class TilingInvariant(unittest.TestCase):
    """A model whose segments do not tile the reference is worth saying out loud."""

    def test_segments_that_tile_report_no_violation(self):
        vntr = _make_vntr()
        reference = 'N' * START + ''.join(SEGMENTS) + 'N' * 100
        self.assertTrue(vntr.segments_tile_reference(reference))

    def test_segments_that_do_not_tile_are_detected(self):
        vntr = _make_vntr()
        reference = 'N' * (START + SUM_OF_SEGMENTS + 100)
        self.assertFalse(vntr.segments_tile_reference(reference))


if __name__ == '__main__':
    unittest.main()
