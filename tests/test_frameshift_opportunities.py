"""Task 7 shadow counters: integer (k, N) per frameshift candidate.

Every test here drives the real `VNTRFinder.find_frameshift_from_selected_reads` from
synthetic vpaths -- no BAM, no HMM, no decoder -- reusing the fixture idiom of
`tests/test_frameshift_context.py:17-107`. The builders are duplicated rather than
imported so neither module constrains the other's fixtures.

The reference unit is 12 bp, not that file's 8 bp, so that a 6 bp insertion and a 3 bp
deletion run both still clear the legacy read-level rejections at
`advntr/vntr_finder.py:366` and `:373` (`> pattern_length / 2`). At 8 bp those cap the
insertion at 4 bases, which is too little room to drive `N` and
`round(ru_bp_coverage / ru_length)` apart in a legible fixture.
"""
import json
import logging
import sys
import unittest
from cStringIO import StringIO

from advntr import frameshift_opportunities
from advntr import settings
from advntr.genome_analyzer import GenomeAnalyzer
from advntr.reference_vntr import ReferenceVNTR
import advntr.vntr_finder as vntr_finder_module
from advntr.vntr_finder import SelectedRead, VNTRFinder


REFERENCE_UNIT = 'ACGTACGTACGT'
UNIT_LENGTH = len(REFERENCE_UNIT)


class _FakeState(object):
    def __init__(self, name):
        self.name = name


class _FakeHMM(object):
    read_length_used_to_build_model = UNIT_LENGTH


class _CallingFinder(VNTRFinder):
    """Always call, so no candidate is hidden behind the p-value cutoff."""

    def identify_frameshift(self, *_args, **_kwargs):
        return 0.0, 1.0, 0.0


class _AlignmentFinder(object):
    """Keep the real candidate traversal while replacing only BAM read selection."""

    def __init__(self, finder, selected_reads):
        self.finder = finder
        self.selected_reads = selected_reads

    def find_frameshift_from_alignment_file(self, _alignment_file, _unmapped):
        return self.finder.find_frameshift_from_selected_reads(self.selected_reads)

    @property
    def last_frameshift_context(self):
        return self.finder.last_frameshift_context


class _StdoutScope(object):
    def __enter__(self):
        self.saved = sys.stdout
        self.stream = StringIO()
        sys.stdout = self.stream
        return self.stream

    def __exit__(self, _exc_type, _exc_value, _traceback):
        sys.stdout = self.saved


def _vpath(state_names):
    return ([(0, _FakeState('start'))] +
            [(i + 1, _FakeState(name)) for i, name in enumerate(state_names)] +
            [(len(state_names) + 1, _FakeState('end'))])


def _unit(events=None, pattern='1', first=1, last=UNIT_LENGTH, close=True):
    """One repeat occurrence. `events` maps a reference position to ordered I/D ops.

    Position 0 carries the start slot's insertions (`I0_p`), position `last` the
    right-boundary slot's (`I{L}_p`); `close=False` leaves `unit_end` unvisited, which
    is what makes an occurrence a `partial_end` one.
    """
    events = events or {}
    states = ['unit_start_%s' % pattern]
    sequence = []
    for _kind, payload in [item for item in events.get(0, []) if item[0] == 'I']:
        for inserted in payload:
            states.append('I0_%s' % pattern)
            sequence.append(inserted)
    for position in range(first, last + 1):
        operations = events.get(position, [])
        if any(kind == 'D' for kind, _payload in operations):
            states.append('D%d_%s' % (position, pattern))
        else:
            states.append('M%d_%s' % (position, pattern))
            sequence.append(REFERENCE_UNIT[position - 1])
        for kind, payload in operations:
            if kind == 'I':
                for inserted in payload:
                    states.append('I%d_%s' % (position, pattern))
                    sequence.append(inserted)
    if close:
        states.append('unit_end_%s' % pattern)
    return states, ''.join(sequence)


def _read(units, query_name='read-1'):
    """Concatenate `(states, sequence)` pieces into one SelectedRead."""
    states = []
    sequence = []
    for unit_states, unit_sequence in units:
        states.extend(unit_states)
        sequence.append(unit_sequence)
    return SelectedRead(''.join(sequence), -1.0, _vpath(states), query_name=query_name)


def _clean_read(query_name='clean'):
    return _read([_unit()], query_name)


def _deletion_read(position, query_name='deleted'):
    return _read([_unit({position: [('D', '')]})], query_name)


def _insertion_read(inserted_sequence, query_name='inserted', position=2):
    return _read([_unit({position: [('I', inserted_sequence)]})], query_name)


def _split_insertion_read(query_name='separated'):
    """One 2 bp insertion at slot 1 reached by two separate visits to `I1_1`.

    The legacy `_LEN2` suffix comes from the whole-read visit count, but
    `extract_raw_mutations` yields two records (`advntr/mutation_keys.py:138-158`), so
    this is the shape that proves both sides deduplicate on the observation identity
    rather than counting raw mutations. Mirrors
    `tests/test_frameshift_context.py:216-227`.
    """
    states = ['unit_start_1', 'M1_1', 'I1_1', 'M2_1', 'I1_1']
    states.extend(['M%d_1' % position for position in range(3, UNIT_LENGTH + 1)])
    states.append('unit_end_1')
    sequence = REFERENCE_UNIT[0] + 'T' + REFERENCE_UNIT[1] + 'G' + REFERENCE_UNIT[2:]
    return SelectedRead(sequence, -1.0, _vpath(states), query_name=query_name)


def _capture_at_level(level, action):
    """Run `action` at a fixed root log level, with output swallowed and then restored."""
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, root.handlers[:]
    root.handlers = [logging.NullHandler()]
    root.setLevel(level)
    try:
        action()
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def _flank_read(suffix_states, prefix_states, flank_sequence, query_name):
    """Left-flank states, one clean repeat occurrence, then right-flank states.

    `flank_sequence` is the bases the flank M/I states emit, in vpath order; the repeat
    unit contributes its own.
    """
    unit_states, unit_sequence = _unit()
    consumed = [state for state in suffix_states if state[0] in ('M', 'I')]
    sequence = (flank_sequence[:len(consumed)] + unit_sequence +
                flank_sequence[len(consumed):])
    return SelectedRead(sequence, -1.0,
                        _vpath(suffix_states + unit_states + prefix_states),
                        query_name=query_name)


def _same_deletion_in_two_occurrences(query_name='two-occurrences'):
    return _read([_unit({3: [('D', '')]}), _unit({3: [('D', '')]})], query_name)


class TestFrameshiftOpportunities(unittest.TestCase):
    def setUp(self):
        self.original_ref_alignment = settings.USE_REF_ALIGNMENT
        self.original_min_support = settings.MIN_SUPPORTING_READ_COUNT
        self.original_full_ru = settings.USE_ONLY_FULLY_COVERED_RU
        self.original_get_pattern_clusters = vntr_finder_module.get_pattern_clusters
        self.original_signature_supports = frameshift_opportunities._signature_supports
        settings.USE_REF_ALIGNMENT = False
        settings.USE_ONLY_FULLY_COVERED_RU = False
        settings.MIN_SUPPORTING_READ_COUNT = 1
        vntr_finder_module.get_pattern_clusters = lambda patterns: [[patterns[0], patterns[1]]]

        reference = ReferenceVNTR(1, REFERENCE_UNIT, 100, 'chr1', None, None)
        reference.init_from_xml([REFERENCE_UNIT, REFERENCE_UNIT], 'TTTTTTTT', 'GGGGGGGG')
        self.finder = _CallingFinder(reference)
        self.finder.hmm = _FakeHMM()

    def tearDown(self):
        frameshift_opportunities._signature_supports = self.original_signature_supports
        vntr_finder_module.get_pattern_clusters = self.original_get_pattern_clusters
        settings.MIN_SUPPORTING_READ_COUNT = self.original_min_support
        settings.USE_ONLY_FULLY_COVERED_RU = self.original_full_ru
        settings.USE_REF_ALIGNMENT = self.original_ref_alignment

    def _run(self, reads):
        self.results = self.finder.find_frameshift_from_selected_reads(reads)
        return self.finder.last_frameshift_opportunities

    @staticmethod
    def _pair(records, candidate):
        return (records[candidate]['support'], records[candidate]['opportunities'])

    def test_read_spanning_a_callable_state_without_an_indel_is_an_opportunity(self):
        """The zero-support inventory: a clean read leaves no trace today, because
        `advntr/vntr_finder.py:402-403` short-circuits before any evidence is recorded."""
        records = self._run([_deletion_read(3), _clean_read(), _clean_read('clean-2')])

        self.assertEqual(self._pair(records, 'D3_1'), (1, 3))
        self.assertLess(records['D3_1']['support'], records['D3_1']['opportunities'])
        self.assertEqual(records['D3_1']['legacy_support'], 1)

    def test_a_multi_base_insertion_counts_once_in_support_and_once_in_opportunity(self):
        records = self._run([_insertion_read('TC')])
        self.assertEqual(self._pair(records, 'I2_1_T_LEN2'), (1, 1))

        records = self._run([_split_insertion_read()])
        self.assertEqual(self._pair(records, 'I1_1_T_LEN2'), (1, 1))

    def test_a_deletion_is_support_and_opportunity_yet_adds_no_repeat_bp_coverage(self):
        """Pin the Q-COV divergence rather than assume it: `is_matching_state`
        (`advntr/hmm_utils.py:135-139`) is false for `D*`, so the deleted position
        contributes nothing to `ru_bp_coverage` (`advntr/hmm_utils.py:326`)."""
        deletion_only = self._run([_deletion_read(3)])
        self.assertEqual(self._pair(deletion_only, 'D3_1'), (1, 1))
        self.assertEqual(deletion_only['D3_1']['ru_bp_coverage'], UNIT_LENGTH - 1)
        self.assertEqual(deletion_only['D3_1']['ru_length'], UNIT_LENGTH)

        with_clean_read = self._run([_deletion_read(3), _clean_read()])
        self.assertEqual(with_clean_read['D3_1']['ru_bp_coverage'], 2 * UNIT_LENGTH - 1)

        # Six deleted positions in each of two occurrences: 12 emitted bases carrying two
        # observations, so the emitted-base ratio (1) cannot stand in for `N` (2).
        run = dict((position, [('D', '')]) for position in range(3, 9))
        records = self._run([_read([_unit(run), _unit(run)], 'deletion-runs')])
        candidate = 'D3_1&D4_1&D5_1&D6_1&D7_1&D8_1'
        self.assertEqual(self._pair(records, candidate), (2, 2))
        self.assertEqual(records[candidate]['ru_bp_coverage'], UNIT_LENGTH)
        self.assertEqual(records[candidate]['ru_bp_coverage_ratio'], 1)

    def test_two_occurrences_of_one_repeat_unit_type_are_two_observations(self):
        """Q-OCC: the legacy count is per read, the shadow count is per occurrence."""
        records = self._run([_same_deletion_in_two_occurrences()])

        self.assertEqual(self._pair(records, 'D3_1'), (2, 2))
        self.assertEqual(records['D3_1']['legacy_support'], 1)
        self.assertEqual(self.results[0][:2], ('D3_1', 1))

    def test_two_mates_sharing_a_query_name_are_two_observations(self):
        records = self._run([_insertion_read('TC', 'shared-name'),
                             _insertion_read('TC', 'shared-name')])

        self.assertEqual(self._pair(records, 'I2_1_T_LEN2'), (2, 2))

    def test_a_flank_candidate_never_draws_on_repeat_unit_occurrences(self):
        supporting = _flank_read(['I8_suffix'], [], 'A', 'supporting')
        crossing = _flank_read(['I3_suffix', 'I3_suffix', 'M8_suffix', 'M9_suffix'], [],
                               'AA' + REFERENCE_UNIT[7] + REFERENCE_UNIT[8], 'crossing')
        records = self._run([supporting, crossing, _deletion_read(3)])

        self.assertEqual(self._pair(records, 'I8_suffix_LEN1'), (1, 2))
        self.assertEqual(records['D3_1']['opportunities'], 3)
        self.assertEqual(records['I8_suffix_LEN1']['pattern_index'], 'suffix')
        self.assertIsNone(records['I8_suffix_LEN1']['ru_bp_coverage'])

    def test_one_read_contributes_at_most_one_observation_per_flank(self):
        twice = _flank_read(['I8_suffix', 'M9_suffix', 'I8_suffix'], [], 'AAA', 'twice')
        records = self._run([twice])

        self.assertEqual(self._pair(records, 'I8_suffix_LEN2'), (1, 1))


    def test_partial_start_has_no_start_slot_opportunity(self):
        """`partial_start` never visits `unit_start`, so the `I0` slot was never
        available to it (`advntr/hmm_utils.py:661-663`)."""
        head_states = ['M%d_1' % position for position in range(4, UNIT_LENGTH + 1)]
        head_states.append('unit_end_1')
        partial_start = SelectedRead(REFERENCE_UNIT[3:], -1.0, _vpath(head_states),
                                     query_name='partial-start')
        records = self._run([_read([_unit({0: [('I', 'A')]})], 'start-slot'), partial_start,
                             _deletion_read(5, 'reached-five')])

        self.assertEqual(self._pair(records, 'I0_1_A_LEN1'), (1, 2))
        # The partial_start span is eligible and reaches position 5, so its absence from
        # the I0 denominator is about the slot, not about the span being dropped.
        self.assertEqual(records['D5_1']['opportunities'], 3)

    def test_partial_end_stopping_mid_unit_has_no_boundary_slot_opportunity(self):
        """An occurrence that never reaches `unit_end` never crossed the `I{L}` slot
        (`advntr/hmm_utils.py:671-680`)."""
        partial_end = _read([_unit(), _unit(last=8, close=False)], 'partial-end')
        boundary = _read([_unit({UNIT_LENGTH: [('I', 'A')]})], 'boundary')
        records = self._run([boundary, partial_end, _deletion_read(5, 'reached-five')])

        self.assertEqual(self._pair(records, 'I12_1_A_LEN1'), (1, 3))
        # Its complete occurrence and its partial_end one are both eligible, so only the
        # boundary slot -- not the span -- is missing from the I12 denominator.
        self.assertEqual(records['D5_1']['opportunities'], 4)

    def test_compound_candidate_requires_every_component_in_one_occurrence(self):
        both = _read([_unit({3: [('D', ''), ('I', 'TC')]})], 'both')
        crossing = _clean_read()
        records = self._run([both, crossing])

        self.assertEqual(self._pair(records, 'D3_1&I3_1_T_LEN2'), (1, 2))

    def test_components_split_across_two_occurrences_support_the_fused_candidate_zero(self):
        """Q-OCC in its most damaging form: `legacy_mutation_candidates` fuses adjacent
        deletions across occurrences (`advntr/mutation_keys.py:189`)."""
        vntr_finder_module.get_pattern_clusters = lambda patterns: [[patterns[0]], [patterns[1]]]
        split = _read([_unit({11: [('D', '')]}, pattern='2'),
                       _unit({12: [('D', '')]}, pattern='2')], 'split')
        records = self._run([split])

        self.assertEqual(records['D11_2&D12_2']['legacy_support'], 1)
        self.assertEqual(self._pair(records, 'D11_2&D12_2'), (0, 2))
        self.assertEqual(self._pair(records, 'D11_2'), (1, 2))
        self.assertEqual(self._pair(records, 'D12_2'), (1, 2))

    def test_support_exceeding_opportunities_raises_with_the_candidate_name(self):
        """SPEC 3.3 asks for `k > N` to be a loud invariant failure, never a clamp."""
        frameshift_opportunities._signature_supports = lambda _signature, _components: False

        with self.assertRaises(ValueError) as caught:
            self._run([_insertion_read('TC')])

        self.assertIn('I2_1_T_LEN2', str(caught.exception))

    def test_opportunities_differ_from_the_emitted_base_coverage_ratio(self):
        """PLAN Task 7 Step 3: quantify the unit mismatch, do not hide it.

        A fully deleted occurrence cannot appear here: `abs(M + I - pattern_length) >
        pattern_length / 2` (`advntr/vntr_finder.py:366`) rejects the read outright, so
        a 3 bp deletion run is the strongest admissible deletion-heavy occurrence.
        """
        reads = [
            _clean_read(),
            _read([_unit({3: [('D', '')], 4: [('D', '')], 5: [('D', '')]})], 'deleted'),
            _insertion_read('TTTTTT', 'inserted'),
            _read([_unit(), _unit(last=8, close=False)], 'partial'),
            _deletion_read(10, 'deleted-10'),
        ]
        records = self._run(reads)

        self.assertEqual(records['D10_1']['ru_bp_coverage'], 70)
        self.assertEqual(records['D10_1']['ru_length'], UNIT_LENGTH)
        self.assertEqual(records['D10_1']['ru_bp_coverage_ratio'], 6)
        self.assertEqual(self._pair(records, 'D10_1'), (1, 5))
        self.assertAlmostEqual(records['D10_1']['avg_bp_coverage'], 70 / 12.0 / 2 / 2)

    def test_diagnostics_are_deterministic_versioned_and_carry_no_query_name(self):
        reads = [_insertion_read('TC', 'shared-name'), _deletion_read(3, 'shared-name'),
                 _clean_read('shared-name')]
        forward = frameshift_opportunities.encode_opportunity_diagnostics(self._run(reads))
        backward = frameshift_opportunities.encode_opportunity_diagnostics(
            self._run(list(reversed(reads))))

        self.assertEqual(forward, backward)
        self.assertTrue(forward.startswith('{"v":1,"candidates":['))
        self.assertNotIn('shared-name', forward)
        self.assertNotIn('\t', forward)
        self.assertNotIn('\n', forward)

    def test_counters_reset_at_the_start_of_each_invocation(self):
        self.assertEqual(_CallingFinder(self.finder.reference_vntr).last_frameshift_opportunities, {})

        self.assertTrue(self._run([_insertion_read('TC')]))
        self.assertEqual(self._run([_clean_read()]), {})

    def test_a_prefix_span_that_exited_one_base_in_is_not_an_opportunity_there(self):
        """`advntr/hmm_utils.py:416` gives EVERY prefix match state a 0.01 transition to
        `prefix_end_prefix`, so the end terminator does not pin the last reference
        position there. A read whose last emitted base sits AT slot 1 could not have
        emitted an insertion into it."""
        supporting = _flank_read([], ['M1_prefix', 'I1_prefix', 'M2_prefix',
                                      'prefix_end_prefix'], 'AAA', 'supporting')
        exited_early = _flank_read([], ['M1_prefix', 'prefix_end_prefix'], 'A', 'early')
        records = self._run([supporting, exited_early])

        self.assertEqual(self._pair(records, 'I1_prefix_LEN1'), (1, 1))

    def test_a_suffix_span_that_entered_deep_has_no_start_slot_opportunity(self):
        """`advntr/hmm_utils.py:459` lets the left flank be entered at ANY match state, so
        `suffix_start_suffix` in the path does not mean the path was ever at position 0.
        `I0` reaches only `M1`/`D1` (`:461-463`), so a span that entered at `M5_suffix`
        never crossed the start slot."""
        supporting = _flank_read(['suffix_start_suffix', 'I0_suffix', 'I0_suffix',
                                  'M1_suffix', 'M2_suffix'], [], 'AA' + REFERENCE_UNIT[:2],
                                 'supporting')
        entered_deep = _flank_read(['suffix_start_suffix', 'M5_suffix', 'M6_suffix',
                                    'suffix_end_suffix'], [], REFERENCE_UNIT[4:6], 'deep')
        records = self._run([supporting, entered_deep])

        self.assertEqual(self._pair(records, 'I0_suffix_LEN2'), (1, 1))

    def test_one_read_inserting_in_two_occurrences_renumbers_the_length_suffix(self):
        """`legacy_mutation_candidates` derives `_LEN%d` from its input count
        (`advntr/mutation_keys.py:167-170`), so the whole-read map names this
        `I2_1_T_LEN2` and the two per-occurrence maps each name it `I2_1_T_LEN1`.
        `legacy_states` is what stops a consumer keying off the `State` column from
        reading the legacy-named row's `support == 0` as absence of support."""
        records = self._run([_read([_unit({2: [('I', 'T')]}), _unit({2: [('I', 'T')]})],
                                   'two-insertions')])

        self.assertEqual(self.results[0][:2], ('I2_1_T_LEN2', 1))
        self.assertEqual(self._pair(records, 'I2_1_T_LEN2'), (0, 2))
        self.assertEqual(records['I2_1_T_LEN2']['legacy_support'], 1)
        self.assertEqual(records['I2_1_T_LEN2']['legacy_states'], [])
        self.assertEqual(self._pair(records, 'I2_1_T_LEN1'), (2, 2))
        self.assertEqual(records['I2_1_T_LEN1']['legacy_support'], 0)
        self.assertEqual(records['I2_1_T_LEN1']['legacy_states'], ['I2_1_T_LEN2'])

    def test_fused_deletion_rows_name_the_legacy_state_their_support_belongs_to(self):
        vntr_finder_module.get_pattern_clusters = lambda patterns: [[patterns[0]], [patterns[1]]]
        split = _read([_unit({11: [('D', '')]}, pattern='2'),
                       _unit({12: [('D', '')]}, pattern='2')], 'split')
        records = self._run([split])

        self.assertEqual(records['D11_2&D12_2']['legacy_states'], [])
        self.assertEqual(records['D11_2']['legacy_states'], ['D11_2&D12_2'])
        self.assertEqual(records['D12_2']['legacy_states'], ['D11_2&D12_2'])

    def test_flank_support_survives_the_short_circuit_that_drops_the_legacy_count(self):
        """`advntr/vntr_finder.py:402-403` skips the whole prefix/suffix block for a read
        with no repeat-unit mutation, so the legacy never counts this flank insertion;
        `observe_read` runs before that `continue`."""
        read = _flank_read([], ['M1_prefix', 'I1_prefix', 'I1_prefix'], 'AAA', 'flank-only')
        records = self._run([read])

        self.assertIsNone(self.results)
        self.assertEqual(self._pair(records, 'I1_prefix_LEN2'), (1, 1))
        self.assertEqual(records['I1_prefix_LEN2']['legacy_support'], 0)
        self.assertEqual(records['I1_prefix_LEN2']['legacy_states'], ['I1_prefix_LEN2'])

    def test_the_diagnostic_encoding_is_skipped_when_info_logging_is_disabled(self):
        """213 KB of JSON on example_66bf's 1014 candidates, discarded by the log level.
        The attribute, not the log line, is the interface the harness reads."""
        calls = []
        original = frameshift_opportunities.encode_opportunity_diagnostics

        def counting_encoder(records, **kwargs):
            calls.append(len(records))
            return original(records, **kwargs)

        frameshift_opportunities.encode_opportunity_diagnostics = counting_encoder
        try:
            _capture_at_level(logging.WARNING, lambda: self._run([_insertion_read('TC')]))
            self.assertEqual(calls, [])
            _capture_at_level(logging.INFO, lambda: self._run([_insertion_read('TC')]))
            self.assertEqual(len(calls), 1)
        finally:
            frameshift_opportunities.encode_opportunity_diagnostics = original

    def test_result_table_stays_six_columns_with_the_first_five_unchanged(self):
        alignment_finder = _AlignmentFinder(self.finder, [_insertion_read('TC')])
        analyzer = GenomeAnalyzer([], [])
        analyzer.ref_filename = 'reference.fa'
        analyzer.target_vntr_ids = [25561]
        analyzer.vntr_finder = {25561: alignment_finder}

        with _StdoutScope() as output:
            analyzer.find_frameshift_from_alignment_file('reads.bam')

        lines = output.getvalue().splitlines()
        state, count, coverage, pvalue = self.finder.find_frameshift_from_selected_reads(
            [_insertion_read('TC')])[0]
        self.assertEqual(lines[3],
                         '#VID\tState\tNumberOfSupportingReads\tMeanCoverage\tPvalue\tContext')
        fields = lines[4].split('\t')
        self.assertEqual(len(fields), 6)
        self.assertEqual(fields[:5], ['25561', state, str(count), str(coverage), str(pvalue)])
        self.assertEqual(json.loads(fields[5])['v'], 1)
        self.assertNotIn('opportunities', lines[4])


if __name__ == '__main__':
    unittest.main()
