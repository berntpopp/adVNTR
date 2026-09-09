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
        # Unmapped keys in explicit mapping raise KeyError
        with self.assertRaises(KeyError):
            lmap.to_internal('4')
        with self.assertRaises(KeyError):
            lmap.to_external(6)

    def test_unconfigured_legacy_fallback(self):
        # Empty unconfigured map provides 1-based positional fallback
        lmap = RepeatUnitLabelMap()
        self.assertEqual(lmap.to_external(0), '1')
        self.assertEqual(lmap.to_external(5), '6')
        self.assertEqual(lmap.to_internal('6'), 5)
        # State name translation round-trip in unconfigured mode
        ext = lmap.translate_state_name_to_external('I10_5_A_LEN1')
        self.assertEqual(ext, 'I10_6_A_LEN1')
        self.assertEqual(lmap.translate_state_name_to_internal(ext), 'I10_5_A_LEN1')
        del_ext = lmap.translate_state_name_to_external('D20_5')
        self.assertEqual(del_ext, 'D20_6')
        self.assertEqual(lmap.translate_state_name_to_internal(del_ext), 'D20_5')

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

    def test_remapping_internal_id_cleans_stale_reverse(self):
        lmap = RepeatUnitLabelMap({0: 'RU1'})
        self.assertEqual(lmap.to_internal('RU1'), 0)
        # Remap 0 to 'RU7'
        lmap.add_mapping(0, 'RU7')
        self.assertEqual(lmap.to_external(0), 'RU7')
        self.assertEqual(lmap.to_internal('RU7'), 0)
        # 'RU1' should no longer point to 0 and can be assigned to internal 1
        with self.assertRaises(KeyError):
            lmap.to_internal('RU1')
        lmap.add_mapping(1, 'RU1')
        self.assertEqual(lmap.to_internal('RU1'), 1)

    def test_composite_label_with_underscores(self):
        # External label contains underscores: 'RU_5C'
        lmap = RepeatUnitLabelMap({0: 'RU_5C'})
        ext = lmap.translate_state_name_to_external('I10_0_A_LEN1')
        self.assertEqual(ext, 'I10_RU_5C_A_LEN1')
        internal = lmap.translate_state_name_to_internal(ext)
        self.assertEqual(internal, 'I10_0_A_LEN1')

        # Deletion
        del_ext = lmap.translate_state_name_to_external('D20_0')
        self.assertEqual(del_ext, 'D20_RU_5C')
        self.assertEqual(lmap.translate_state_name_to_internal(del_ext), 'D20_0')

    def test_overlapping_prefix_label_roundtrip(self):
        # Labels overlapping across underscores with non-nucleotide suffixes: 'RU' and 'RU_motif'
        lmap = RepeatUnitLabelMap({0: 'RU', 1: 'RU_motif'})
        state = 'I10_0_A_LEN1'
        ext = lmap.translate_state_name_to_external(state)
        self.assertEqual(ext, 'I10_RU_A_LEN1')
        self.assertEqual(lmap.translate_state_name_to_internal(ext), state)
        # Unit 1 roundtrip
        state1 = 'I10_1_A_LEN1'
        ext1 = lmap.translate_state_name_to_external(state1)
        self.assertEqual(ext1, 'I10_RU_motif_A_LEN1')
        self.assertEqual(lmap.translate_state_name_to_internal(ext1), state1)

    def test_explicitly_empty_map_json_roundtrip(self):
        # RepeatUnitLabelMap({}) is in explicit mode and strictly rejects unknown IDs
        m = RepeatUnitLabelMap({})
        with self.assertRaises(KeyError):
            m.to_external(0)
        # Deserialized copy must preserve explicit mode and continue raising KeyError
        restored = RepeatUnitLabelMap.from_json(m.to_json())
        with self.assertRaises(KeyError):
            restored.to_external(0)
        self.assertTrue(m.is_compatible_with(restored))

    def test_unconfigured_map_json_roundtrip(self):
        # RepeatUnitLabelMap() is in unconfigured mode and uses 1-based positional fallback
        m = RepeatUnitLabelMap()
        self.assertEqual(m.to_external(0), '1')
        restored = RepeatUnitLabelMap.from_json(m.to_json())
        self.assertEqual(restored.to_external(0), '1')
        self.assertTrue(m.is_compatible_with(restored))
        # Explicitly empty map is not compatible with unconfigured map
        explicit_empty = RepeatUnitLabelMap({})
        self.assertFalse(explicit_empty.is_compatible_with(m))

    def test_state_name_translation_unmapped_explicit_raises(self):
        lmap = RepeatUnitLabelMap({0: '1', 1: '2', 2: '3', 3: '5', 4: '6', 5: '7'})
        # D20_4 has unmapped external label '4'
        with self.assertRaises(KeyError):
            lmap.translate_state_name_to_internal('D20_4')
        with self.assertRaises(KeyError):
            lmap.translate_state_name_to_internal('I20_4_A_LEN1')
        with self.assertRaises(KeyError):
            lmap.translate_state_name_to_internal('M20_4')

    def test_ambiguous_or_reserved_labels_rejected(self):
        # Labels colliding with insertion metadata (_LEN<int>) must be rejected
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'RU_A_LEN1'})
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'RU_LEN2'})
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'LEN1'})
        # Labels colliding with insertion of bases on a prefix label must be rejected
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'RU', 1: 'RU_A'})
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'RU_A', 1: 'RU'})
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'RU', 1: 'RU_AGATCGGA'})
        # Labels containing compound delimiter '&' must be rejected
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'RU&7'})
        # Labels containing reserved flank identifiers 'prefix' or 'suffix' must be rejected
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'RU_suffix_5'})
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: 'prefix_1'})
        # Empty labels must be rejected
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({0: ''})
        # Negative internal IDs must be rejected
        with self.assertRaises(ValueError):
            RepeatUnitLabelMap({-1: 'RU1'})

    def test_composite_label_without_len_suffix_roundtrip(self):
        # Non-colliding labels like RU_A and RU_C
        lmap = RepeatUnitLabelMap({0: 'RU_A', 1: 'RU_C'})
        # Exact state translation (no insertion metadata)
        self.assertEqual(lmap.translate_state_name_to_external('I10_0'), 'I10_RU_A')
        self.assertEqual(lmap.translate_state_name_to_internal('I10_RU_A'), 'I10_0')
        # Insertion with base but no LEN
        ins_base = lmap.translate_state_name_to_external('I10_0_G')
        self.assertEqual(ins_base, 'I10_RU_A_G')
        self.assertEqual(lmap.translate_state_name_to_internal(ins_base), 'I10_0_G')
        # Insertion with emitted base and length on RU_A
        ins_ext = lmap.translate_state_name_to_external('I10_0_G_LEN2')
        self.assertEqual(ins_ext, 'I10_RU_A_G_LEN2')
        self.assertEqual(lmap.translate_state_name_to_internal(ins_ext), 'I10_0_G_LEN2')
        # Insertion on RU_C
        ins_ru_c = lmap.translate_state_name_to_external('I10_1_T_LEN1')
        self.assertEqual(ins_ru_c, 'I10_RU_C_T_LEN1')
        self.assertEqual(lmap.translate_state_name_to_internal(ins_ru_c), 'I10_1_T_LEN1')
        # Deletion and match states
        self.assertEqual(lmap.translate_state_name_to_external('D20_0'), 'D20_RU_A')
        self.assertEqual(lmap.translate_state_name_to_internal('D20_RU_A'), 'D20_0')
        self.assertEqual(lmap.translate_state_name_to_external('M20_0'), 'M20_RU_A')
        self.assertEqual(lmap.translate_state_name_to_internal('M20_RU_A'), 'M20_0')


if __name__ == '__main__':
    unittest.main()
