"""The frozen background artifact: it is supplied, validated, or the caller does not run.

PLAN Task 8 Step 3's *mechanism* only. Calibrating a real background happens outside
this repository and is a later sub-task; Global Constraint G5 keeps every cohort-derived
number, path and sample id out of Git, so every probability written here is synthetic
and obviously so (0.25, 0.125 -- nothing that could be mistaken for a measured indel
background). SPEC Q-RATE additionally forbids planting the public candidate-conditioned
summaries as a default, so an absent artifact must mean "do not run", never "fall back".
"""
import ast
import json
import os
import shutil
import tempfile
import unittest

from advntr import frameshift_background
from advntr.frameshift_background import BackgroundModelError, load_background_model


VALID = {
    'schema': 'advntr.frameshift.background',
    'version': 1,
    'provenance': 'SYNTHETIC FIXTURE -- not calibrated on anything',
    'default_probability': 0.25,
    'states': {'D3_1': 0.125},
}


class TestBackgroundArtifact(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix='advntr-background-test-')

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def _write(self, document, name='background.json'):
        path = os.path.join(self.tempdir, name)
        with open(path, 'w') as handle:
            if isinstance(document, str):
                handle.write(document)
            else:
                json.dump(document, handle)
        return path

    def _refusal(self, document, name='background.json'):
        path = self._write(document, name)
        with self.assertRaises(BackgroundModelError) as caught:
            load_background_model(path)
        message = str(caught.exception)
        self.assertIn(path, message)
        return message

    def test_a_valid_artifact_loads_and_reports_its_provenance(self):
        model = load_background_model(self._write(VALID))

        self.assertEqual(model.version, 1)
        self.assertEqual(model.provenance, VALID['provenance'])
        self.assertEqual(model.probability_for('D3_1'), 0.125)

    def test_an_unknown_state_falls_back_to_the_declared_default(self):
        """Every candidate must get a `p0`, and the artifact -- not this tree -- decides
        what it is."""
        model = load_background_model(self._write(VALID))

        self.assertEqual(model.probability_for('I7_2_G_LEN1'), 0.25)

    def test_an_absent_file_is_refused_by_name(self):
        missing = os.path.join(self.tempdir, 'nothing-here.json')
        with self.assertRaises(BackgroundModelError) as caught:
            load_background_model(missing)

        self.assertIn(missing, str(caught.exception))
        self.assertIn('not found', str(caught.exception))

    def test_a_file_that_is_not_json_is_refused(self):
        self.assertIn('not valid JSON', self._refusal('this is not json at all'))

    def test_a_json_document_that_is_not_an_object_is_refused(self):
        self.assertIn('JSON object', self._refusal('[1, 2, 3]'))

    def test_a_foreign_document_is_refused_by_schema(self):
        document = dict(VALID, schema='some.other.thing')
        self.assertIn('schema', self._refusal(document))

    def test_a_version_less_artifact_is_refused(self):
        document = dict(VALID)
        del document['version']
        self.assertIn('version', self._refusal(document))

    def test_an_unsupported_version_is_refused_and_names_the_version(self):
        message = self._refusal(dict(VALID, version=99))
        self.assertIn('99', message)
        self.assertIn('version', message)

    def test_a_provenance_less_artifact_is_refused(self):
        """The artifact must say what it was calibrated on. G5 forbids naming cohort
        samples, so a free-text line the operator writes is the whole record."""
        document = dict(VALID)
        del document['provenance']
        self.assertIn('provenance', self._refusal(document))

    def test_an_empty_provenance_is_refused(self):
        self.assertIn('provenance', self._refusal(dict(VALID, provenance='   ')))

    def test_a_missing_default_probability_is_refused(self):
        document = dict(VALID)
        del document['default_probability']
        self.assertIn('default_probability', self._refusal(document))

    def test_a_nan_probability_is_refused_and_names_the_field(self):
        message = self._refusal('{"schema": "advntr.frameshift.background", '
                                '"version": 1, "provenance": "SYNTHETIC", '
                                '"default_probability": NaN, "states": {}}')
        self.assertIn('NaN', message)
        self.assertIn('default_probability', message)

    def test_a_nan_probability_inside_states_is_refused_and_names_the_state(self):
        message = self._refusal('{"schema": "advntr.frameshift.background", '
                                '"version": 1, "provenance": "SYNTHETIC", '
                                '"default_probability": 0.25, '
                                '"states": {"D3_1": NaN}}')
        self.assertIn('NaN', message)
        self.assertIn('D3_1', message)

    def test_a_probability_of_zero_or_one_is_refused(self):
        """The open interval, not the closed one: `p0 = 0` makes every observation
        infinitely surprising and `p0 = 1` makes none of them surprising at all."""
        self.assertIn('0.0', self._refusal(dict(VALID, default_probability=0.0)))
        self.assertIn('1.0', self._refusal(dict(VALID, default_probability=1.0)))

    def test_a_probability_above_one_is_refused_and_names_the_state(self):
        message = self._refusal(dict(VALID, states={'D3_1': 1.5}))
        self.assertIn('D3_1', message)
        self.assertIn('1.5', message)

    def test_a_non_numeric_probability_is_refused(self):
        message = self._refusal(dict(VALID, states={'D3_1': 'small'}))
        self.assertIn('D3_1', message)
        self.assertIn('number', message)

    def test_states_must_be_a_mapping(self):
        self.assertIn('states', self._refusal(dict(VALID, states=[1, 2])))

    def test_an_unknown_top_level_field_is_refused_rather_than_ignored(self):
        """A misspelled `"state"` would otherwise load as a valid single-rate model and
        score every candidate against the default -- silently, and with a `states` table
        the operator believes is in force."""
        document = dict(VALID, state={'D3_1': 0.125})
        del document['states']
        message = self._refusal(document)

        self.assertIn("'state'", message)
        self.assertIn('unknown field', message)

    def test_a_boolean_version_is_refused(self):
        """`True in (1,)` is True in Python 2, so a membership test alone accepts it and
        the model then describes itself as `vTrue`."""
        message = self._refusal(dict(VALID, version=True))

        self.assertIn('version must be an integer', message)

    def test_the_states_table_may_be_omitted_entirely(self):
        """A single-rate background is a legitimate artifact; the per-state table is the
        refinement, not the requirement."""
        document = dict(VALID)
        del document['states']
        model = load_background_model(self._write(document))

        self.assertEqual(model.probability_for('D3_1'), 0.25)

    def test_no_probability_literal_lives_in_the_module_code(self):
        """SPEC Q-RATE: the public candidate-conditioned summaries (~1e-3, 3.0e-4,
        1.7e-4) are conditional on candidates with support >= 3 and are not plug-in
        estimates for a production null. A default that looks calibrated but is not is
        worse than no default, so the module must ship no numeric fallback at all.

        Asserted over the parsed AST, not the file text: the docstring is *required* to
        name those rejected values and to carry a synthetic example, and a substring
        search cannot tell prose from a planted constant. Every float the code itself
        may contain is an interval bound (`0.0`, `1.0`); anything strictly between them
        is a probability, and there is no legitimate reason for one to be here.
        """
        source = open(frameshift_background.__file__.rstrip('c')).read()
        inside_the_unit_interval = [node.n for node in ast.walk(ast.parse(source))
                                    if isinstance(node, ast.Num)
                                    and isinstance(node.n, float)
                                    and 0.0 < node.n < 1.0]

        self.assertEqual(inside_the_unit_interval, [])


if __name__ == '__main__':
    unittest.main()
