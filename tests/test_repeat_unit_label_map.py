"""Tests for RepeatUnitLabelMap (Issue #7)."""
import unittest

from advntr.repeat_unit_label_map import RepeatUnitLabelMap


class TestRepeatUnitLabelMap(unittest.TestCase):

    def test_contiguous_mapping(self):
        # 0-indexed internal to 1-indexed external
        lmap = RepeatUnitLabelMap(['1', '2', '3', '4', '5', '6', '7'])
        self.assertEqual(lmap.to_external(0), '1')
        self.assertEqual(lmap.to_external(6), '7')
        self.assertEqual(lmap.to_internal('7'), 6)

    def test_non_contiguous_mapping(self):
        # Motif 4 missing in a new assembly: external labels [1, 2, 3, 5, 6, 7]
        # Internal dense indices 0, 1, 2, 3, 4, 5
        mapping = {0: '1', 1: '2', 2: '3', 3: '5', 4: '6', 5: '7'}
        lmap = RepeatUnitLabelMap(mapping, model_id='hg38_v2')
        self.assertEqual(lmap.to_external(3), '5')
        self.assertEqual(lmap.to_internal('5'), 3)
        self.assertEqual(lmap.to_external(5), '7')
        self.assertEqual(lmap.to_internal('7'), 5)

    def test_state_name_translation(self):
        # Map internal 5 -> '6', 6 -> '7'
        mapping = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7'}
        lmap = RepeatUnitLabelMap(mapping)

        # Single mutation
        self.assertEqual(lmap.translate_state_name_to_external('I10_5_A_LEN1'), 'I10_6_A_LEN1')
        self.assertEqual(lmap.translate_state_name_to_external('D20_6'), 'D20_7')

        # Compound mutation
        compound = 'D20_6&D21_6&I22_6_A_LEN9'
        expected = 'D20_7&D21_7&I22_7_A_LEN9'
        self.assertEqual(lmap.translate_state_name_to_external(compound), expected)

        # Round trip
        self.assertEqual(lmap.translate_state_name_to_internal(expected), compound)

        # Flanks unaffected
        self.assertEqual(lmap.translate_state_name_to_external('M_left_prefix_10'), 'M_left_prefix_10')

    def test_json_roundtrip(self):
        mapping = {0: 'RU1', 1: 'RU2', 2: 'RU5C'}
        lmap = RepeatUnitLabelMap(mapping, model_id='custom_muc1')
        json_data = lmap.to_json()

        restored = RepeatUnitLabelMap.from_json(json_data)
        self.assertEqual(restored.model_id, 'custom_muc1')
        self.assertEqual(restored.to_external(2), 'RU5C')
        self.assertTrue(lmap.is_compatible_with(restored))


if __name__ == '__main__':
    unittest.main()
