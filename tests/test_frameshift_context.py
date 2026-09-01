import json
import sys
import unittest
from cStringIO import StringIO

from advntr import settings
from advntr.genome_analyzer import GenomeAnalyzer
from advntr.mutation_keys import extract_raw_mutations
from advntr.reference_vntr import ReferenceVNTR
import advntr.vntr_finder as vntr_finder_module
from advntr.vntr_finder import SelectedRead, VNTRFinder


REFERENCE_UNIT = 'ACGTACGT'


class _FakeState(object):
    def __init__(self, name):
        self.name = name


class _FakeHMM(object):
    read_length_used_to_build_model = len(REFERENCE_UNIT)


class _CallingFinder(VNTRFinder):
    def identify_frameshift(self, *_args, **_kwargs):
        return 0.0, 1.0, 0.0


class _PvalueRejectingFinder(_CallingFinder):
    def identify_frameshift(self, *_args, **_kwargs):
        return 1.0, 0.0, 1.0


class _OutputFinder(object):
    last_frameshift_context = {
        'I2_1_T_LEN2': '{"v":1,"contexts":[{"read_occurrence_support":1}]}'
    }

    def find_frameshift_from_alignment_file(self, _alignment_file, _unmapped):
        return [('I2_1_T_LEN2', 1, 2.5, 0.01)]


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


def _vpath(state_names):
    return ([(0, _FakeState('start'))] +
            [(i + 1, _FakeState(name)) for i, name in enumerate(state_names)] +
            [(len(state_names) + 1, _FakeState('end'))])


def _read_with_events(events, query_name='read-1'):
    """events maps a reference offset to ordered I/D operations before its match."""
    states = ['unit_start_1']
    sequence = []
    for position, base in enumerate(REFERENCE_UNIT, 1):
        deleted = any(event_type == 'D' for event_type, _sequence in events.get(position, []))
        for event_type, inserted_sequence in events.get(position, []):
            if event_type == 'D':
                states.append('D%d_1' % position)
        if not deleted:
            states.append('M%d_1' % position)
            sequence.append(base)
        for event_type, inserted_sequence in events.get(position, []):
            if event_type == 'I':
                for inserted_base in inserted_sequence:
                    states.append('I%d_1' % position)
                    sequence.append(inserted_base)
    states.append('unit_end_1')
    return SelectedRead(''.join(sequence), -1.0, _vpath(states), query_name=query_name)


def _insertion_read(inserted_sequence, query_name='read-1'):
    return _read_with_events({2: [('I', inserted_sequence)]}, query_name)


def _matching_read(query_name='match'):
    return _read_with_events({}, query_name)


def _same_deletion_in_two_occurrences(query_name='two-occurrences'):
    states = []
    sequence = []
    for _occurrence in range(2):
        states.append('unit_start_1')
        for position, base in enumerate(REFERENCE_UNIT, 1):
            if position == 3:
                states.append('D3_1')
            else:
                states.append('M%d_1' % position)
                sequence.append(base)
        states.append('unit_end_1')
    return SelectedRead(''.join(sequence), -1.0, _vpath(states), query_name=query_name)


class _StdoutScope(object):
    def __enter__(self):
        self.saved = sys.stdout
        self.stream = StringIO()
        sys.stdout = self.stream
        return self.stream

    def __exit__(self, _exc_type, _exc_value, _traceback):
        sys.stdout = self.saved


class TestFrameshiftContext(unittest.TestCase):
    def setUp(self):
        self.original_ref_alignment = settings.USE_REF_ALIGNMENT
        self.original_min_support = settings.MIN_SUPPORTING_READ_COUNT
        self.original_full_ru = settings.USE_ONLY_FULLY_COVERED_RU
        self.original_get_pattern_clusters = vntr_finder_module.get_pattern_clusters
        settings.USE_REF_ALIGNMENT = False
        settings.USE_ONLY_FULLY_COVERED_RU = False
        settings.MIN_SUPPORTING_READ_COUNT = 1
        vntr_finder_module.get_pattern_clusters = lambda patterns: [[patterns[0], patterns[1]]]

        reference = ReferenceVNTR(1, REFERENCE_UNIT, 100, 'chr1', None, None)
        reference.init_from_xml([REFERENCE_UNIT, REFERENCE_UNIT], 'TTTTTTTT', 'GGGGGGGG')
        self.finder = _CallingFinder(reference)
        self.finder.hmm = _FakeHMM()

    def tearDown(self):
        vntr_finder_module.get_pattern_clusters = self.original_get_pattern_clusters
        settings.MIN_SUPPORTING_READ_COUNT = self.original_min_support
        settings.USE_ONLY_FULLY_COVERED_RU = self.original_full_ru
        settings.USE_REF_ALIGNMENT = self.original_ref_alignment

    def test_full_insertion_context_is_compact_versioned_and_state_is_unchanged(self):
        results = self.finder.find_frameshift_from_selected_reads([_insertion_read('TC')])

        self.assertEqual(results, [('I2_1_T_LEN2', 1, 0.3125, 0.0)])
        expected = (
            '{"v":1,"contexts":[{"events":[{"inserted_sequence":"TC",'
            '"normalized_offset":1,"normalized_sequence":"CT","raw_offset":2,'
            '"type":"I"}],"observed_unit":"ACTCGTACGT","read_occurrence_support":1,'
            '"repeat_occurrence":0}]}'
        )
        self.assertEqual(self.finder.last_frameshift_context['I2_1_T_LEN2'], expected)
        self.assertNotIn('\t', expected)
        self.assertNotIn('\n', expected)

    def test_compound_state_retains_ordered_structured_events(self):
        read = _read_with_events({3: [('D', ''), ('I', 'TC')]})
        results = self.finder.find_frameshift_from_selected_reads([read])

        self.assertEqual(results[0][0], 'D3_1&I3_1_T_LEN2')
        context = json.loads(self.finder.last_frameshift_context[results[0][0]])['contexts'][0]
        self.assertEqual(context['observed_unit'], 'ACTCTACGT')
        self.assertEqual(context['events'], [
            {'inserted_sequence': '', 'raw_offset': 3, 'type': 'D'},
            {'inserted_sequence': 'TC', 'normalized_offset': 3,
             'normalized_sequence': 'TC', 'raw_offset': 3, 'type': 'I'},
        ])

    def test_identical_normalized_events_do_not_merge_distinct_contexts(self):
        earlier_context = _insertion_read('TC', 'earlier-sort')
        earlier_context.sequence = 'ACTCGTACGA'
        results = self.finder.find_frameshift_from_selected_reads([
            _insertion_read('TC', 'later-sort'),
            earlier_context,
        ])

        self.assertEqual(results[0][:2], ('I2_1_T_LEN2', 2))
        contexts = json.loads(self.finder.last_frameshift_context['I2_1_T_LEN2'])['contexts']
        self.assertEqual([context['observed_unit'] for context in contexts],
                         ['ACTCGTACGA', 'ACTCGTACGT'])
        self.assertEqual([context['events'][0]['normalized_sequence'] for context in contexts],
                         ['CT', 'CT'])
        self.assertEqual([context['read_occurrence_support'] for context in contexts], [1, 1])

    def test_same_query_name_mates_keep_distinct_immutable_evidence_by_read_ordinal(self):
        results = self.finder.find_frameshift_from_selected_reads([
            _insertion_read('TC', 'shared-name'),
            _insertion_read('TC', 'shared-name'),
        ])

        state = results[0][0]
        evidence = self.finder.last_frameshift_evidence[state]
        self.assertEqual([(item.selected_read_index, item.query_name, item.repeat_occurrence)
                          for item in evidence],
                         [(0, 'shared-name', 0), (1, 'shared-name', 0)])
        self.assertEqual(
            json.loads(self.finder.last_frameshift_context[state])['contexts'][0]['read_occurrence_support'], 2
        )
        self.assertNotIn('shared-name', self.finder.last_frameshift_context[state])
        with self.assertRaises(AttributeError):
            evidence[0].query_name = 'changed'

    def test_random_match_states_advance_the_cursor_without_entering_the_observed_unit(self):
        states = ['start_random_matches', 'unit_start_1', 'M1_1', 'I1_1',
                  'M2_1', 'unit_end_1', 'end_random_matches', 'I0_prefix']

        raw = extract_raw_mutations(states, 'XATCYZ', [REFERENCE_UNIT])

        self.assertEqual(raw[3].legacy_key, 'I1_1_T')
        self.assertEqual(raw[3].event.inserted_sequence, 'T')
        self.assertEqual(raw[3].observed_unit, 'ATC')
        self.assertEqual(raw[7].legacy_key, 'I0_prefix')
        self.assertEqual(raw[7].event.inserted_sequence, 'Z')
        self.assertEqual(raw[7].observed_unit, 'Z')

    def test_separated_visits_to_one_insertion_state_remain_ordered_events(self):
        states = ['unit_start_1', 'M1_1', 'I1_1', 'M2_1', 'I1_1']
        states.extend(['M%d_1' % position for position in range(3, 9)])
        states.append('unit_end_1')
        read = SelectedRead('ATCGGTACGT', -1.0, _vpath(states), query_name='separated')

        results = self.finder.find_frameshift_from_selected_reads([read])

        self.assertEqual(results[0][:2], ('I1_1_T_LEN2', 1))
        events = self.finder.last_frameshift_evidence['I1_1_T_LEN2'][0].events
        self.assertEqual([event.inserted_sequence for event in events], ['T', 'G'])
        self.assertEqual([event.raw_offset for event in events], [1, 1])

    def test_one_read_supporting_the_same_state_twice_keeps_two_occurrence_records(self):
        results = self.finder.find_frameshift_from_selected_reads([
            _same_deletion_in_two_occurrences()
        ])

        self.assertEqual(results[0][:2], ('D3_1', 1))
        evidence = self.finder.last_frameshift_evidence['D3_1']
        self.assertEqual([(record.selected_read_index, record.query_name,
                           record.repeat_occurrence) for record in evidence],
                         [(0, 'two-occurrences', 0), (0, 'two-occurrences', 1)])
        contexts = json.loads(self.finder.last_frameshift_context['D3_1'])['contexts']
        self.assertEqual(sum(context['read_occurrence_support'] for context in contexts), 2)

    def test_subthreshold_candidate_keeps_internal_evidence_but_emits_no_context_row(self):
        settings.MIN_SUPPORTING_READ_COUNT = 2
        alignment_finder = _AlignmentFinder(self.finder, [_insertion_read('TC', 'subthreshold')])
        analyzer = GenomeAnalyzer([], [])
        analyzer.ref_filename = 'reference.fa'
        analyzer.target_vntr_ids = [25561]
        analyzer.vntr_finder = {25561: alignment_finder}

        with _StdoutScope() as output:
            analyzer.find_frameshift_from_alignment_file('reads.bam')

        self.assertEqual(output.getvalue().splitlines(), [
            '#Input File: reads.bam',
            '#Reference file: reference.fa',
            '#P-value cutoff: 0.001',
            '#VID\tState\tNumberOfSupportingReads\tMeanCoverage\tPvalue\tContext',
        ])
        evidence = self.finder.last_frameshift_evidence['I2_1_T_LEN2']
        self.assertEqual([(record.selected_read_index, record.query_name,
                           record.repeat_occurrence) for record in evidence],
                         [(0, 'subthreshold', 0)])
        self.assertEqual(self.finder.last_frameshift_context, {})

    def test_pvalue_rejected_candidate_keeps_internal_support_evidence(self):
        rejecting = _PvalueRejectingFinder(self.finder.reference_vntr)
        rejecting.hmm = _FakeHMM()

        self.assertIsNone(rejecting.find_frameshift_from_selected_reads([_insertion_read('TC')]))
        self.assertEqual(len(rejecting.last_frameshift_evidence['I2_1_T_LEN2']), 1)
        self.assertEqual(rejecting.last_frameshift_context, {})

    def test_context_and_evidence_reset_at_the_start_of_each_invocation(self):
        self.finder.find_frameshift_from_selected_reads([_insertion_read('TC')])
        self.assertTrue(self.finder.last_frameshift_context)
        self.assertTrue(self.finder.last_frameshift_evidence)

        self.assertIsNone(self.finder.find_frameshift_from_selected_reads([_matching_read()]))
        self.assertEqual(self.finder.last_frameshift_context, {})
        self.assertEqual(self.finder.last_frameshift_evidence, {})

    def test_partial_repeat_occurrences_use_stable_string_labels(self):
        partial_start_states = ['M%d_1' % position for position in range(1, 7)]
        partial_start_states.extend(['I6_1', 'unit_end_1'])
        partial_start = SelectedRead('ACGTACT', -1.0, _vpath(partial_start_states), query_name='start')
        start_results = self.finder.find_frameshift_from_selected_reads([partial_start])
        start_context = json.loads(self.finder.last_frameshift_context[start_results[0][0]])['contexts'][0]
        self.assertEqual(start_context['repeat_occurrence'], 'partial_start')

        partial_end_states = ['M0_suffix', 'unit_start_1']
        partial_end_states.extend(['M%d_1' % position for position in range(1, 7)])
        partial_end_states.append('I6_1')
        partial_end = SelectedRead('TACGTACT', -1.0, _vpath(partial_end_states), query_name='end')
        end_results = self.finder.find_frameshift_from_selected_reads([partial_end])
        end_context = json.loads(self.finder.last_frameshift_context[end_results[0][0]])['contexts'][0]
        self.assertEqual(end_context['repeat_occurrence'], 'partial_end')

    def test_emitted_flank_calls_use_explicit_flank_labels(self):
        self.finder.is_frameshift_mode = True
        suffix_states = ['I8_suffix', 'unit_start_1', 'M1_1', 'M2_1', 'I2_1']
        suffix_states.extend(['M%d_1' % position for position in range(3, 9)])
        suffix_states.append('unit_end_1')
        suffix_read = SelectedRead('TACTGTACGT', -1.0, _vpath(suffix_states), query_name='suffix')
        suffix_results = self.finder.find_frameshift_from_selected_reads([suffix_read])
        suffix_state = [result[0] for result in suffix_results if 'suffix' in result[0]][0]
        suffix_context = json.loads(self.finder.last_frameshift_context[suffix_state])['contexts'][0]
        self.assertEqual(suffix_context['repeat_occurrence'], 'suffix_flank')

        prefix_states = ['unit_start_1', 'M1_1', 'M2_1', 'I2_1']
        prefix_states.extend(['M%d_1' % position for position in range(3, 9)])
        prefix_states.extend(['unit_end_1', 'I0_prefix'])
        prefix_read = SelectedRead('ACTGTACGTT', -1.0, _vpath(prefix_states), query_name='prefix')
        prefix_results = self.finder.find_frameshift_from_selected_reads([prefix_read])
        prefix_state = [result[0] for result in prefix_results if 'prefix' in result[0]][0]
        prefix_context = json.loads(self.finder.last_frameshift_context[prefix_state])['contexts'][0]
        self.assertEqual(prefix_context['repeat_occurrence'], 'prefix_flank')

    def test_analyzer_appends_context_without_changing_the_first_five_columns(self):
        analyzer = GenomeAnalyzer([], [])
        analyzer.ref_filename = 'reference.fa'
        analyzer.target_vntr_ids = [25561]
        analyzer.vntr_finder = {25561: _OutputFinder()}

        with _StdoutScope() as output:
            analyzer.find_frameshift_from_alignment_file('reads.bam')

        lines = output.getvalue().splitlines()
        self.assertEqual(lines[3], '#VID\tState\tNumberOfSupportingReads\tMeanCoverage\tPvalue\tContext')
        self.assertEqual(
            lines[4],
            '25561\tI2_1_T_LEN2\t1\t2.5\t0.01\t'
            '{"v":1,"contexts":[{"read_occurrence_support":1}]}'
        )
        self.assertEqual(lines[4].split('\t')[:5],
                         ['25561', 'I2_1_T_LEN2', '1', '2.5', '0.01'])


if __name__ == '__main__':
    unittest.main()
