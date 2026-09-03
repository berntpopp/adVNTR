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

    # -- Task 8i: `probability_for` is a byte-exact dict lookup with no normalisation
    # (`advntr/frameshift_background.py:probability_for`), so a `states` key that is
    # not byte-identical to an emitted `State` string never matches anything and
    # silently scores that state against `default_probability`. These tests trip each
    # rejection deliberately -- `tests/test_ratchets.py`'s model: a gate nobody has
    # seen fail is a gate nobody knows works.

    def test_a_trailing_space_key_is_refused_rather_than_silently_scored_against_default(self):
        """The defect this task closes. Before `_validated_state_keys` existed, a
        `states` key of `"D3_1 "` loaded without error and `probability_for('D3_1')`
        silently returned `default_probability` -- the state was never looked up under
        its dirty key, and nothing said so. Now the artifact is refused outright."""
        message = self._refusal(dict(VALID, states={'D3_1 ': 0.125}))

        self.assertIn(repr('D3_1 '), message)
        self.assertIn('whitespace', message)

    def test_a_leading_space_key_is_refused(self):
        message = self._refusal(dict(VALID, states={' D3_1': 0.125}))

        self.assertIn(repr(' D3_1'), message)
        self.assertIn('whitespace', message)

    def test_internal_whitespace_around_a_compound_separator_is_refused_rather_than_silently_scored_against_default(self):
        """Fix round 1's own gap, found by adversarial review: `"D3_1 &D4_1"` has no
        LEADING or TRAILING whitespace on the *whole* key, so the whole-key `.strip()`
        check alone (`_validated_state_keys` rule 4, first half) let it through -- and
        `parse_components` never validates a component's pattern-index field's
        *contents* (rule 5), so the grammar check let it through too. The key then
        loaded and `probability_for('D3_1&D4_1')` -- the clean form a real compound
        candidate actually uses -- would have silently returned `default_probability`:
        the identical failure this task exists to close, just moved one character to
        the right of the `&`. 2,642 of the 5,500 real states this task's evidence run
        collected are compound `A&B&...` forms, so this was not an edge case."""
        message = self._refusal(dict(VALID, states={'D3_1 &D4_1': 0.125}))

        self.assertIn(repr('D3_1 &D4_1'), message)
        self.assertIn(repr('D3_1 '), message)
        self.assertIn('whitespace', message)

    def test_an_empty_states_key_is_refused(self):
        """`''.strip() == ''`, so the whitespace rule alone cannot catch this -- it
        needs its own rule."""
        message = self._refusal(dict(VALID, states={'': 0.125}))

        self.assertIn('empty', message)

    def test_two_keys_that_collide_once_stripped_are_both_named_in_the_refusal(self):
        """JSON does not catch this itself: `"D3_1"` and `"D3_1 "` are different raw
        text, so both survive `json.load` as separate `states` entries -- only a
        literally repeated key string would collapse, silently, inside `json.load`
        before this module ever sees the dict. An artifact carrying both means the
        calibration named the same state twice, which is refused rather than silently
        picking one."""
        message = self._refusal(dict(VALID, states={'D3_1': 0.1, 'D3_1 ': 0.2}))

        self.assertIn(repr('D3_1'), message)
        self.assertIn(repr('D3_1 '), message)
        self.assertIn('collide', message)

    def test_two_separate_collision_groups_are_both_reported_in_one_refusal(self):
        """`_validated_state_keys` rule 2 collects every colliding group before
        refusing, rather than raising on the first one found: a 21,000-key artifact
        should cost one edit-and-rerun cycle for its whole set of collisions, not one
        per group. Two unrelated pairs here must both be named in the single message
        this call raises."""
        message = self._refusal(dict(VALID, states={
            'D3_1': 0.1, 'D3_1 ': 0.2, 'D4_1': 0.3, ' D4_1': 0.4,
        }))

        self.assertIn(repr('D3_1'), message)
        self.assertIn(repr('D3_1 '), message)
        self.assertIn(repr('D4_1'), message)
        self.assertIn(repr(' D4_1'), message)

    def test_a_key_the_shipped_grammar_cannot_produce_is_refused(self):
        """See `_validated_state_keys` rule 5's docstring for why this rule cannot
        refuse a legitimate key for any cohort -- the structural-closure argument, and
        the public-corpus run that corroborates it. A key with no leading `I`/`D`
        submodel letter is not one of the forms that argument covers."""
        message = self._refusal(dict(VALID, states={'not_a_state': 0.125}))

        self.assertIn(repr('not_a_state'), message)
        self.assertIn('grammar', message)

    def test_a_non_string_key_is_refused(self):
        """JSON object keys are always strings -- `json.load` cannot hand
        `_validated_state_keys` anything else, so this path is unreachable through
        `load_background_model`'s public file-based API. Checked anyway because
        `raw_states` is just a `dict` with no guarantee every caller came through JSON,
        so this test calls the validator directly, the same way
        `test_no_probability_literal_lives_in_the_module_code` reaches past the public
        API to exercise something a JSON fixture cannot express."""
        with self.assertRaises(BackgroundModelError) as caught:
            frameshift_background._validated_state_keys('some/path.json', {1: 0.125})

        message = str(caught.exception)
        self.assertIn('some/path.json', message)
        self.assertIn('not a string', message)

    def test_real_emitted_state_forms_all_load_and_score_by_their_own_key(self):
        """Guards against the new validation quietly rejecting everything. Every key
        below is a real `State`/candidate string this fork's caller emitted on public
        BAMs, collected via `finder.last_frameshift_evidence.keys()` for
        `_validated_state_keys` rule 5's evidence run (that docstring has the full
        corpus and counts): a plain deletion, an insertion with its emitted base and
        length, the undecorated deletion flank form with no `_LEN` suffix (the shape
        most likely to trip a naive grammar check), an insertion flank form, and a
        multi-component deletion+insertion compound. All five must load, and each must
        score by its own state-specific rate rather than falling back to the shared
        default."""
        real_states = {
            'D3_1': 0.10,
            'I10_1_A_LEN1': 0.11,
            'D148_suffix': 0.12,
            'I0_prefix_LEN1': 0.13,
            'D10_2&I10_2_C_LEN9': 0.14,
        }
        model = load_background_model(self._write(dict(VALID, states=real_states)))

        for state, probability in real_states.items():
            self.assertEqual(model.probability_for(state), probability)
        self.assertEqual(model.probability_for('I7_2_G_LEN1'), VALID['default_probability'])

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
