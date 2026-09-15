"""Exact, non-duplicated genotype options that bind calibration policy/assets."""


_DESTINATIONS = frozenset((
    'pacbio', 'nanopore', 'frameshift', 'haploid', 'threads', 'min_read_length',
    'prune_reverse', 'filter_adapter_readthrough', 'min_read_match_ratio',
    'rare_unit_coverage_guard', 'noref_aln', 'fullru', 'models',
    'exact_frameshift_caller', 'frameshift_background', 'frameshift_pvalue_cutoff',
    'min_frameshift_read_support', 'frameshift_calibration_out', 'frameshift_capture_version',
))


def validate_genotype_arguments(parser, arguments):
    """Check actual argparse actions, including aliases and --name=value syntax.

    Python 2 argparse cannot disable abbreviations. Inspect its option resolution
    without interpreting values ourselves; normal argparse still owns usage errors.
    """
    seen = set()
    for token in arguments:
        if token == '--':
            break
        parsed = parser._parse_optional(token)
        if parsed is None or parsed[0] is None:
            continue
        action, _option, _value = parsed
        if action.dest not in _DESTINATIONS:
            continue
        spelling = token.split('=', 1)[0]
        if spelling not in action.option_strings:
            parser.error('calibration options require exact spelling: %s' % token)
        if action.dest in seen:
            parser.error('duplicate calibration option: %s' % spelling)
        seen.add(action.dest)
