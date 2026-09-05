"""Frameshift row ordering.

The emitted call set must stay the same, but tied rows must not depend on the order in
which the mutation dictionary first saw them.
"""
import logging
import unittest
from cStringIO import StringIO

from advntr import settings
from advntr.reference_vntr import ReferenceVNTR
import advntr.vntr_finder as vntr_finder_module
from advntr.vntr_finder import SelectedRead, VNTRFinder


class _FakeState(object):
    def __init__(self, name):
        self.name = name


class _FakeHMM(object):
    read_length_used_to_build_model = 50


class _OrderingFinder(VNTRFinder):
    def identify_frameshift(self, *_args, **_kwargs):
        return 0.0, 1.0, 0.0


class _RealignmentVisibilityFinder(_OrderingFinder):
    def get_repeat_unit_number(self, _read):
        return ['1', '3', '2'], ['A' * 60, 'C' * 60, 'G' * 60], [0, 60, 120, 180]


class _LogCapture(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _RootLoggerScope(object):
    def __init__(self, level, handlers):
        self.level = level
        self.handlers = handlers

    def __enter__(self):
        self.logger = logging.getLogger()
        self.original_level = self.logger.level
        self.original_handlers = list(self.logger.handlers)
        self.logger.handlers = list(self.handlers)
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.logger.handlers = self.original_handlers
        self.logger.setLevel(self.original_level)


def _vpath(state_names):
    return ([(0, _FakeState('start'))] +
            [(i + 1, _FakeState(name)) for i, name in enumerate(state_names)] +
            [(len(state_names) + 1, _FakeState('end'))])


def _matching_read(length, query_name):
    states = ['unit_start_1']
    for index in range(1, length + 1):
        states.append('M%d_1' % index)
    states.append('unit_end_1')
    return SelectedRead('A' * length, -1.0, _vpath(states), query_name=query_name)


def _deletion_read(position, query_name):
    states = ['unit_start_1']
    for index in range(1, 51):
        if index == position:
            states.append('D%d_1' % index)
        else:
            states.append('M%d_1' % index)
    states.append('unit_end_1')
    return SelectedRead('A' * 49, -1.0, _vpath(states), query_name=query_name)


def _insertion_read(position, query_name):
    states = ['unit_start_1']
    for index in range(1, 51):
        if index == position:
            states.append('I%d_1' % index)
        states.append('M%d_1' % index)
    states.append('unit_end_1')
    return SelectedRead('A' * 51, -1.0, _vpath(states), query_name=query_name)


class TestFrameshiftOrdering(unittest.TestCase):
    def setUp(self):
        self._original_ref_alignment = settings.USE_REF_ALIGNMENT
        self._original_min_support = settings.MIN_SUPPORTING_READ_COUNT
        self._original_get_pattern_clusters = vntr_finder_module.get_pattern_clusters
        settings.USE_REF_ALIGNMENT = False
        settings.MIN_SUPPORTING_READ_COUNT = 1
        vntr_finder_module.get_pattern_clusters = lambda patterns: [[patterns[0], patterns[1]]]

        reference = ReferenceVNTR(1, 'A' * 50, 100, 'chr1', None, None)
        reference.init_from_xml(['A' * 50, 'A' * 50], 'TTTTTTTTTT', 'GGGGGGGGGG')
        self.finder = _OrderingFinder(reference)
        self.finder.hmm = _FakeHMM()

    def tearDown(self):
        vntr_finder_module.get_pattern_clusters = self._original_get_pattern_clusters
        settings.MIN_SUPPORTING_READ_COUNT = self._original_min_support
        settings.USE_REF_ALIGNMENT = self._original_ref_alignment

    def _ordered_states(self, reads):
        frameshifts = self.finder.find_frameshift_from_selected_reads(reads)
        return [state for state, _count, _coverage, _pval in frameshifts]

    def _run_low_coverage_realignment_logging(self):
        settings.USE_REF_ALIGNMENT = True
        try:
            reference = ReferenceVNTR(2, 'A' * 60, 100, 'chr1', None, None)
            reference.init_from_xml(['A' * 60, 'C' * 60, 'G' * 60], 'TTTTTTTTTT', 'GGGGGGGGGG')
            finder = _RealignmentVisibilityFinder(reference, is_frameshift_mode=True)
            finder.hmm = type(str('_ReadLength151'), (object,), {'read_length_used_to_build_model': 151})()

            capture = _LogCapture()
            with _RootLoggerScope(logging.INFO, [capture]):
                finder.find_frameshift_from_selected_reads([
                    _matching_read(60, 'r1'),
                    _matching_read(60, 'r2'),
                ])
            return capture.messages
        finally:
            settings.USE_REF_ALIGNMENT = False

    def test_tied_rows_do_not_depend_on_insertion_history(self):
        """Changing which tied state is seen first must not reorder the final rows."""
        expected = ['D41_1', 'I42_1_A_LEN1']
        deletions_first = [
            _deletion_read(41, 'd1'),
            _insertion_read(42, 'i1'),
            _deletion_read(41, 'd2'),
            _insertion_read(42, 'i2'),
            _deletion_read(41, 'd3'),
            _insertion_read(42, 'i3'),
        ]
        insertions_first = [
            _insertion_read(42, 'i1'),
            _deletion_read(41, 'd1'),
            _insertion_read(42, 'i2'),
            _deletion_read(41, 'd2'),
            _insertion_read(42, 'i3'),
            _deletion_read(41, 'd3'),
        ]

        self.assertEqual(self._ordered_states(deletions_first), expected)
        self.assertEqual(self._ordered_states(insertions_first), expected)

    def test_non_tied_rows_keep_the_existing_ascending_support_order(self):
        states = self._ordered_states([
            _deletion_read(41, 'd1'),
            _insertion_read(42, 'i1'),
            _insertion_read(42, 'i2'),
            _insertion_read(42, 'i3'),
        ])

        self.assertEqual(states, ['D41_1', 'I42_1_A_LEN1'])

    def test_logs_once_when_repeat_unit_realignment_is_inactive_below_three_units(self):
        """Dropping the low-coverage visibility log would hide a real disabled branch."""
        inactive_messages = [message for message in self._run_low_coverage_realignment_logging()
                             if 'repeat-unit realignment is inactive' in message]
        self.assertEqual(len(inactive_messages), 1)
        self.assertIn('151', inactive_messages[0])
        self.assertIn('60', inactive_messages[0])

    def test_preexisting_root_handlers_stay_silent_while_capture_observes_the_info_log(self):
        """Appending our handler instead of isolating handlers would leak noise to stderr."""
        leaked_stream = StringIO()
        noisy_handler = logging.StreamHandler(leaked_stream)
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        root_logger.addHandler(noisy_handler)
        try:
            capture_messages = self._run_low_coverage_realignment_logging()
            self.assertEqual(leaked_stream.getvalue(), '')
            self.assertEqual(list(root_logger.handlers), original_handlers + [noisy_handler])
        finally:
            root_logger.removeHandler(noisy_handler)
        inactive_messages = [message for message in capture_messages
                             if 'repeat-unit realignment is inactive' in message]
        self.assertEqual(len(inactive_messages), 1)


if __name__ == '__main__':
    unittest.main()
