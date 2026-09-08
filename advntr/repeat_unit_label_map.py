"""Repeat-unit identity stability and label translation (Issue #7, Constraint 1).

In adVNTR, repeat units were historically identified by a dense positional index:
    sorted(list(set(patterns))) then i + 1

Re-segmenting or adding motifs renumbers every repeat unit. Downstream consumers
(such as VNtyper, clinical guidelines, and panel-of-normal denylists like RU7)
key rules on emitted state strings (e.g. D20_7, I10_6_A_LEN1). Renumbering silently
alters the semantic meaning of all downstream rules.

This module explicitly decouples:
  1. Internal dense cluster indices (0-indexed or 1-indexed dense integers used by
     HMM topology and DP matrices).
  2. External stable labels (strings such as "1", "2", ... "7", or arbitrary motif
     identifiers like "RU_5C"), which may be non-contiguous without requiring
     dummy HMM states.
"""
import json
import re


class RepeatUnitLabelMap(object):
    """Bidirectional mapping between internal dense cluster IDs and stable external labels."""

    def __init__(self, mapping=None, model_id=None):
        """Initialize from a dict {internal_id: external_label} or list of external labels.

        :param mapping: dict of int -> str, or list of str
        :param model_id: optional model identifier / hash for panel-of-normals validation
        """
        self.model_id = str(model_id) if model_id is not None else None
        self._internal_to_external = {}
        self._external_to_internal = {}

        if mapping is not None:
            if isinstance(mapping, (list, tuple)):
                for idx, label in enumerate(mapping):
                    self.add_mapping(idx, label)
            elif isinstance(mapping, dict):
                for internal_id, label in mapping.items():
                    self.add_mapping(int(internal_id), label)

    def add_mapping(self, internal_id, external_label):
        internal_id = int(internal_id)
        external_label = str(external_label).strip()
        if not external_label:
            raise ValueError('External label cannot be empty')
        if external_label in self._external_to_internal and self._external_to_internal[external_label] != internal_id:
            raise ValueError('Duplicate external label %s mapped to multiple internal IDs' % external_label)
        self._internal_to_external[internal_id] = external_label
        self._external_to_internal[external_label] = internal_id

    def to_external(self, internal_id):
        internal_id = int(internal_id)
        if internal_id in self._internal_to_external:
            return self._internal_to_external[internal_id]
        # Default fallback if unmapped: str(internal_id + 1) for 0-indexed internal
        return str(internal_id + 1)

    def to_internal(self, external_label):
        external_label = str(external_label).strip()
        if external_label in self._external_to_internal:
            return self._external_to_internal[external_label]
        # Default fallback: try parsing as int - 1
        try:
            return int(external_label) - 1
        except ValueError:
            raise KeyError('Unknown external label: %s' % external_label)

    def translate_state_name_to_external(self, state_name):
        """Translate state name string components from internal index to external label.

        Example: 'I22_6_A_LEN7' -> 'I22_7_A_LEN7' (when internal 6 maps to external '7').
        Example: 'D20_6&D21_6' -> 'D20_7&D21_7'.
        """
        if '&' in state_name:
            components = state_name.split('&')
            translated = [self._translate_single_component_to_external(c) for c in components]
            return '&'.join(translated)
        return self._translate_single_component_to_external(state_name)

    def _translate_single_component_to_external(self, component):
        if 'prefix' in component or 'suffix' in component:
            return component
        # Standard format: {MUT}{POS}_{RU_IDX}[_{NUC}_LEN{LEN}]
        parts = component.split('_')
        if len(parts) >= 2 and parts[0][0] in ('M', 'I', 'D', 'S'):
            try:
                internal_id = int(parts[1])
                parts[1] = self.to_external(internal_id)
                return '_'.join(parts)
            except ValueError:
                return component
        return component

    def translate_state_name_to_internal(self, state_name):
        """Translate state name string components from external label to internal index."""
        if '&' in state_name:
            components = state_name.split('&')
            translated = [self._translate_single_component_to_internal(c) for c in components]
            return '&'.join(translated)
        return self._translate_single_component_to_internal(state_name)

    def _translate_single_component_to_internal(self, component):
        if 'prefix' in component or 'suffix' in component:
            return component
        parts = component.split('_')
        if len(parts) >= 2 and parts[0][0] in ('M', 'I', 'D', 'S'):
            external_label = parts[1]
            if external_label in self._external_to_internal:
                parts[1] = str(self._external_to_internal[external_label])
                return '_'.join(parts)
        return component

    def to_dict(self):
        return {
            'model_id': self.model_id,
            'mapping': {str(k): v for k, v in self._internal_to_external.items()}
        }

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        mapping = data.get('mapping', {})
        model_id = data.get('model_id')
        label_map = cls(model_id=model_id)
        for k, v in mapping.items():
            label_map.add_mapping(int(k), v)
        return label_map

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    def is_compatible_with(self, other_map):
        """Verify that external labels and model identities match."""
        if not isinstance(other_map, RepeatUnitLabelMap):
            return False
        if self.model_id and other_map.model_id and self.model_id != other_map.model_id:
            return False
        return self._internal_to_external == other_map._internal_to_external

    def __len__(self):
        return len(self._internal_to_external)
