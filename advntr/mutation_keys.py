def left_normalise_insertion(ref_unit, offset, inserted_sequence):
    """Canonicalise an insertion without changing its implied edited repeat unit."""
    if ref_unit is None or len(ref_unit) == 0:
        raise ValueError('ref_unit must be non-empty')
    if inserted_sequence is None or len(inserted_sequence) == 0:
        raise ValueError('inserted_sequence must be non-empty')
    if offset < 0 or offset > len(ref_unit):
        raise ValueError('offset must be between 0 and len(ref_unit)')

    normalised_offset = offset
    rotated_inserted_sequence = inserted_sequence

    while normalised_offset > 0:
        previous_base = ref_unit[normalised_offset - 1]
        if rotated_inserted_sequence[-1] != previous_base:
            break
        rotated_inserted_sequence = previous_base + rotated_inserted_sequence[:-1]
        normalised_offset -= 1

    return normalised_offset, rotated_inserted_sequence
