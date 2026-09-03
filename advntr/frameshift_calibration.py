"""The default-off calibration sink: export the span inventory an offline fitter needs.

PLAN Task 8 Step 3 has to freeze a per-`State` background on a calibration partition, and
the estimator that does it needs `N` for a `State` in the samples where that `State` did
NOT fire. Nothing that leaves this tool today carries that number:

- `OpportunityCounter.finalise` emits one row per
  `sorted(set(legacy_support) | set(self._support))`
  (`advntr/frameshift_opportunities.py:554`), so a candidate with neither a legacy count
  nor occurrence-scoped support in this sample has no row at all, and therefore no
  denominator at all.
- The encoded diagnostics drop the identity and span fields outright (`UNENCODED_FIELDS`,
  `advntr/frameshift_opportunities.py:133`), so even the rows that do exist reach the run
  log without the evidence behind their two integers.
- `advntr/vntr_finder.py:429` publishes only the records. The counter is a local
  (`:238`) and goes out of scope with its span inventory inside it.

`OpportunityCounter._spans` is the one object that generates every missing denominator.
For any `State`, `N` is the number of distinct occurrences whose span signature satisfies
every one of its components -- `_signature_supports` applied to that table -- and nothing
else in the tree enumerates an occurrence that produced no mutation at all
(`occurrence_spans`, whose docstring says why: `advntr/vntr_finder.py:402-403`
short-circuits on an empty mutation map before any evidence is recorded). This module
exports that table, and the rows, when `settings.FRAMESHIFT_CALIBRATION_OUT` names a
path, and does nothing whatsoever otherwise.

The flag is independent of `--exact-frameshift-caller` on purpose: a calibration capture
runs with the caller OFF, so the capture cannot perturb the calls it is measuring.

## Why a counter method opens a file

It is a real design smell and it is not defended as good structure. It is defended as the
only place the data exists, against the one alternative, which is worse:

- The inventory is private and deliberately unexported. Publishing `_spans` on
  `VNTRFinder` would need lines in `advntr/vntr_finder.py`, which `scripts/loc_ratchet.py`
  pins at exactly 1212 and `tests/test_ratchets.py:77-93` enforces as EQUALITY rather than
  as a ceiling -- unfunded headroom is impossible there by construction, and raising a
  ceiling is forbidden.
- Publishing it would also add a second consumer of a field whose entire cost argument is
  that it stays inside: `finalise` deliberately carries span IDS on a row and resolves
  them against this table exactly once, which is what keeps it at candidates x distinct
  shapes instead of candidates x reads (`advntr/frameshift_opportunities.py`'s module
  docstring).

What that smell actually costs is bounded and worth writing down. The write is one
`open(path, 'a')`, one line, one `close`, once per `finalise` -- i.e. once per VNTR per
run, since `advntr/genome_analyzer.py:215-216` loops over `target_vntr_ids` and each
iteration reaches `find_frameshift_from_selected_reads` once. It is off by default, so the
shipped path opens no file and the counter stays a pure accumulator.

## One line

JSON Lines, append mode, one line per `finalise` invocation. Not one document per file:
`genotype -vid` takes a comma-separated list, so a single-document file would silently
overwrite on a multi-VNTR run. Every line is self-identifying (`schema`, `version`,
`vntr_id`, `read_length`, `is_haploid`), so a duplicate left by a resumed run is
detectable and removable offline rather than silently doubling a denominator.

`spans` is one entry per distinct signature, `[pattern_index, reached, inserted,
saw_start, saw_end, count]`, in the counter's own enumeration order, so an entry's index
in the list is the span id `finalise` gave it. `count` is the number of DISTINCT
identities recorded under the signature, exactly as `finalise` computes it
(`advntr/frameshift_opportunities.py:550-552`); anything else would make the offline `N`
disagree with the run's own.

`candidates` carries every field `_record` emits except `opportunity_spans` -- so
`support_identities` and `state_identities` are in, the two of `UNENCODED_FIELDS`' three
that `spans` cannot regenerate. `state_identities` is the one `aggregate_evidence` keys
`k` on; `support_identities` is the row's own evidence and is what lets a fitter re-derive
the aggregation offline instead of trusting it.

**`opportunity_spans` is excluded, and that is the one place the sink is opinionated.**
Measured through the real CLI on `example_66bf_hg19_subset.bam` -- one public hg19 BAM, a
single-sample measurement and not a corpus claim -- at 1,014 candidate rows over 1,373
distinct spans: the whole line is 443,002 bytes, of which the span table is 59,082 and the
rows are 383,790. `opportunity_spans` alone is 2,644,839 bytes: 45x the span table that
regenerates it, and 6.9x the entire line it would be added to. It is exactly derivable
offline from `spans` with the shipped `parse_components` and `_signature_supports`, so
storing it means storing a derived quantity at 45x the cost of its inputs. The sink stores
primitives and derives nothing; the `opportunities` integer each row already carries is
then a free per-row check that the offline recompute agrees with the run, which
`tests/test_frameshift_calibration.py` makes load-bearing -- and which held on that real
capture, 0 mismatches over all 1,014 rows.

Deterministic by construction: `sort_keys=True`, compact separators, the candidate list
sorted on `candidate`, one line, no trailing whitespace. Anonymous by construction: every
identity has already been through `anonymous_identities`, which drops `query_name`
(`advntr/frameshift_opportunities.py:157-171`), and this module adds no field of its own
-- the property `tests/test_frameshift_context.py:199` pins for Task 5's Context column.

## What a consumer must check, and the one thing it cannot

`advntr/exact_caller.py`'s `aggregate_evidence` docstring states the obligation this
sink's consumer inherits: `k`'s identities and `N`'s trials are different sets, only their
cardinality is guarded at runtime, and with `N` in the tens of thousands a leaked identity
is invisible there. Two halves, and they are not equally served:

- **Cardinality, offline: yes.** Recompute each row's `opportunities` from `spans` and
  assert it equals the stored integer. That is the round trip, and it fails the moment the
  span table and the rows disagree.
- **The set property, offline: NO, and saying otherwise would be an overclaim.** The
  exported table carries a COUNT per signature, not the identities behind it, so an
  offline consumer cannot ask which signature a given `(read, occurrence)` was recorded
  under. Storing those identities turns a per-signature integer into a per-occurrence
  list: on the same capture, 26,593 identity pairs, and the span table goes from 59,082
  bytes to 1,683,975 -- 28x, and 4.7x the whole line. The set property is pinned
  in-process instead, against the counter's own `_spans`, by
  `tests/test_frameshift_calibration.py`'s `TestSubsetObligation`, which also carries a
  fixture where a leak really does pass the runtime guard.

  This is a limitation of the shipped format, not a claim that the check is unnecessary.
  It stands on the measurement `advntr/exact_caller.py` cites -- 0 leaked identities over
  7,796 rows on all eight public `example_*` BAMs -- plus the runtime `support >
  opportunities` guard, which refuses the call rather than clamping. Neither is a
  substitute for the per-sample set test on a calibration corpus, and a future task that
  wants that test has to widen this format and pay the 4.7x.
"""
import json

from advntr import settings


#: Written on every line so a fitter reading a mixed directory can refuse anything else.
SCHEMA = 'advntr.frameshift.calibration'

#: Bump when a line's meaning changes, not when a `_record` field is added: the candidate
#: objects are `_record`'s own fields and a consumer reads them by name.
VERSION = 1

#: The only `_record` field the sink drops. See the module docstring for the measurement.
EXCLUDED_FIELDS = ('opportunity_spans',)


def sink_line(vntr_id, read_length, is_haploid, span_counts, records):
    """The exact bytes of one line, without the newline. Pure; nothing here touches disk.

    `span_counts` is `finalise`'s own `[(span_id, signature, count), ...]`. The id is
    dropped because it IS the list index -- the order is the counter's `OrderedDict`
    enumeration order, so preserving it keeps the row-level span ids meaningful to an
    in-process consumer, and re-sorting the list would silently break that.

    The candidate list is sorted rather than left in `records`' order. `finalise` already
    builds an `OrderedDict` over `sorted(...)`, so this changes nothing today; it means
    determinism does not depend on that staying true, which is the same reason
    `encode_opportunity_diagnostics` sorts its encoded strings
    (`advntr/frameshift_opportunities.py:632-647`).
    """
    spans = [list(signature) + [count] for _span_id, signature, count in span_counts]
    candidates = [dict((key, value) for key, value in record.items()
                       if key not in EXCLUDED_FIELDS)
                  for record in records.values()]
    candidates.sort(key=lambda row: row['candidate'])
    return json.dumps({'schema': SCHEMA, 'version': VERSION, 'vntr_id': vntr_id,
                       'read_length': read_length, 'is_haploid': bool(is_haploid),
                       'spans': spans, 'candidates': candidates},
                      sort_keys=True, separators=(',', ':'))


def write_if_configured(finder, is_haploid, span_counts, records):
    """Append one line for this invocation, or return `None` when the flag is unset.

    The guard is here and only here, so the call site in
    `OpportunityCounter.finalise` cannot lose it and a second call site cannot forget it.
    It runs before anything is read off `finder`, which is what makes the default path
    free of both the file and the attribute walk.

    `finder` is the `VNTRFinder` whose `find_frameshift_from_selected_reads` owns this
    counter. It is read for exactly two fields, both of which identify the invocation and
    neither of which the counter is given: `reference_vntr.id` and
    `hmm.read_length_used_to_build_model` -- the same attribute `advntr/vntr_finder.py:488`
    reads for the flank boundary tests, so it is set well before `finalise` runs.

    Appended, not written: a resumed run (`advntr/advntr_commands.py:111-121` supports
    `--append`) must add to the capture rather than truncate it, and each line says which
    VNTR it scored so duplicates are removable offline.
    """
    path = settings.FRAMESHIFT_CALIBRATION_OUT
    if not path:
        return None
    if finder is None:
        raise ValueError('--frameshift-calibration-out is set but the opportunity '
                         'counter was built without the finder that identifies the '
                         'invocation; a line could not say which VNTR it scored')
    line = sink_line(finder.reference_vntr.id,
                     finder.hmm.read_length_used_to_build_model,
                     is_haploid, span_counts, records)
    handle = open(path, 'a')
    try:
        handle.write(line + '\n')
    finally:
        handle.close()
    return line
