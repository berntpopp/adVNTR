# Task 1 Report: Deterministic row order

## Scope

Task 1 required a deterministic tie-break for frameshift rows without changing the emitted
call set. The binding ruling in the parent task overrode the brief's contradictory
strongest-first example: non-tied rows had to keep the existing ascending support order,
and only tied rows were allowed to reorder.

This was implemented in the `calling-quality` worktree at
`/home/bernt-popp/development/adVNTR/.worktrees/calling-quality`.

## Skill use

- `superpowers:test-driven-development`
  - Used before the production edit. It directly drove the RED/GREEN cycle below.
- `superpowers:verification-before-completion`
  - Used before closing and committing. Every passing claim below comes from a fresh run.

## Implementation

### Production change

- `advntr/vntr_finder.py`
  - Changed the frameshift row sort from `key=lambda x: x[1]` to
    `key=lambda item: (item[1], item[0])`.
  - This preserves the existing ascending order for non-tied support counts and adds a
    deterministic lexical tie-break on the state name.
  - Removed one blank line so the file shrank from 1406 to 1405 LOC, satisfying the
    repository rule that over-limit files may only shrink.

### Test

- `tests/test_frameshift_ordering.py`
  - Added a focused regression test that drives the real
    `VNTRFinder.find_frameshift_from_selected_reads()` logic with synthetic
    `SelectedRead`s.
  - The test patches only irrelevant dependencies for this behavior:
    `settings.USE_REF_ALIGNMENT` and `get_pattern_clusters`.
  - It asserts:
    - tied rows do not depend on mutation insertion history;
    - non-tied rows keep the existing ascending support order.

### Documentation / ratchet updates

- `scripts/loc_ratchet.py`
  - Lowered the grandfathered ceiling for `advntr/vntr_finder.py` from 1406 to 1405.
- `AGENTS.md`
  - Lowered the documented `advntr/vntr_finder.py` ceiling to 1405.
  - Removed the now-fixed "tied mutations sort by count alone" trap.
  - Corrected the stale `select_illumina_reads` trap to the current `genotype -fs -u`
    failure mode: `iteratively_update_model` rebuilds through the non-enhanced
    `get_read_matcher_model`, whose `Model.from_matrix` call raises `AttributeError` on
    the enhanced backend.

## RED / GREEN evidence

### RED

Command:

```bash
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
python -m unittest tests.test_frameshift_ordering
```

Result:

```text
.F
======================================================================
FAIL: test_tied_rows_do_not_depend_on_insertion_history
...
AssertionError: Lists differ: ['D41_1', 'I42_1_A_LEN1'] != ['I42_1_A_LEN1', 'D41_1']
...
Ran 2 tests in 0.002s

FAILED (failures=1)
```

Interpretation:

- The new regression test failed for the intended reason: under the old
  `key=lambda x: x[1]` sort, the same tied call set came back in different orders when
  the mutation dictionary first saw the tied states in different orders.
- The non-tied ascending-order test already passed, which matches the task ruling that
  only tied rows may reorder.

### GREEN

Command:

```bash
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
python -m unittest tests.test_frameshift_ordering
```

Result:

```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.002s

OK
```

Interpretation:

- After the production change, the tied-order regression passed and the non-tied-order
  guard remained green.

## Full verification

### Gate

Command:

```bash
set -o pipefail
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
make gate
```

Result:

```text
no upstream remote: ok
python scripts/loc_ratchet.py
loc ratchet: ok (69 files checked)
python setup_hmm.py build_ext --inplace
python -m unittest discover tests
...
wrote /tmp/tmpMZIyJM/tier2_manifest.json
VERIFIED identical against /tmp/tmpJJRYZD/tier2_manifest.json
...
Ran 226 tests in 38.081s

OK (skipped=1)
coverage run --source=advntr,hmm -m unittest discover tests
...
wrote /tmp/tmpWvaOG2/tier2_manifest.json
VERIFIED identical against /tmp/tmp9Yqsu9/tier2_manifest.json
...
Ran 226 tests in 40.248s

OK (skipped=1)
coverage 26.0% (baseline 8.0%)
gate: PASS
```

Interpretation:

- Gate passed cleanly.
- The starting gate was 224 tests; after adding `tests/test_frameshift_ordering.py`, gate
  is now 226 tests.

### Seven-file public-corpus call-set comparison

Requirement: verify the emitted call set on the seven public hg19 corpus BAMs, with a
non-empty comparison.

No repo-native oracle directly compares seven-file emitted frameshift call sets, so I
compared the real `find_frameshift_from_alignment_file()` results between:

- this modified worktree; and
- a clean `HEAD` snapshot materialized with `git archive HEAD` into a temporary tree and
  rebuilt in place with `python setup_hmm.py build_ext --inplace`.

The comparison used only the seven public hg19 BAMs already named in the repository:

- `example_6449_hg19_subset.bam`
- `example_66bf_hg19_subset.bam`
- `example_6c28_hg19_subset.bam`
- `example_7a61_hg19_subset.bam`
- `example_a5c1_hg19_subset.bam`
- `example_b178_hg19_subset.bam`
- `example_dfc3_hg19_subset.bam`

Command:

```bash
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
python - <<'PY'
import json
import os
import shutil
import subprocess
import tempfile

repo = '/home/bernt-popp/development/adVNTR/.worktrees/calling-quality'
data_root = '/home/bernt-popp/development/VNtyper/tests/data'
bams = [
    'example_6449_hg19_subset.bam',
    'example_66bf_hg19_subset.bam',
    'example_6c28_hg19_subset.bam',
    'example_7a61_hg19_subset.bam',
    'example_a5c1_hg19_subset.bam',
    'example_b178_hg19_subset.bam',
    'example_dfc3_hg19_subset.bam',
]
code = r'''
import json
import sys
from advntr_harness.capture import build_finder
bam = sys.argv[1]
db = sys.argv[2]
finder, _reference = build_finder(db)
rows = finder.find_frameshift_from_alignment_file(bam, []) or []
print(json.dumps(rows))
'''

snapshot_dir = tempfile.mkdtemp(prefix='task1-head-')
try:
    archive = subprocess.Popen(['git', 'archive', 'HEAD'], cwd=repo, stdout=subprocess.PIPE)
    untar = subprocess.Popen(['tar', '-x', '-C', snapshot_dir], stdin=archive.stdout)
    archive.stdout.close()
    untar.wait()
    archive.wait()

    subprocess.check_call(['python', 'setup_hmm.py', 'build_ext', '--inplace'], cwd=snapshot_dir)

    summary = []
    total_rows = 0
    for bam_name in bams:
        bam_path = os.path.join(data_root, bam_name)
        current_db = os.path.join(repo, 'tests', 'golden', 'models', 'hg19_muc1.db')
        head_db = os.path.join(snapshot_dir, 'tests', 'golden', 'models', 'hg19_muc1.db')
        current_rows = json.loads(subprocess.check_output(['python', '-c', code, bam_path, current_db], cwd=repo))
        head_rows = json.loads(subprocess.check_output(['python', '-c', code, bam_path, head_db], cwd=snapshot_dir))
        current_set = sorted([tuple(row) for row in current_rows])
        head_set = sorted([tuple(row) for row in head_rows])
        summary.append({
            'bam': bam_name,
            'current_count': len(current_rows),
            'head_count': len(head_rows),
            'order_equal': current_rows == head_rows,
            'set_equal': current_set == head_set,
        })
        total_rows += len(current_rows)
    print(json.dumps({'summary': summary, 'total_rows': total_rows}, indent=2, sort_keys=True))
finally:
    shutil.rmtree(snapshot_dir)
PY
```

Result:

```json
{
  "summary": [
    {
      "bam": "example_6449_hg19_subset.bam",
      "current_count": 2,
      "head_count": 2,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_66bf_hg19_subset.bam",
      "current_count": 2,
      "head_count": 2,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_6c28_hg19_subset.bam",
      "current_count": 7,
      "head_count": 7,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_7a61_hg19_subset.bam",
      "current_count": 1,
      "head_count": 1,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_a5c1_hg19_subset.bam",
      "current_count": 2,
      "head_count": 2,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_b178_hg19_subset.bam",
      "current_count": 2,
      "head_count": 2,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_dfc3_hg19_subset.bam",
      "current_count": 3,
      "head_count": 3,
      "order_equal": true,
      "set_equal": true
    }
  ],
  "total_rows": 19
}
```

Interpretation:

- The comparison was non-empty: 19 emitted rows across the seven-file public hg19 subset.
- All seven files matched `HEAD` exactly as call sets.
- On this public subset, raw row order also stayed unchanged (`order_equal: true` for all
  seven), so the deterministic tie-break is exercised by the synthetic regression test
  rather than these corpus files.

Observed stderr during this comparison:

```text
advntr/vntr_finder.py:193-194: RuntimeWarning: divide by zero encountered in double_scalars
advntr/vntr_finder.py:193-194: RuntimeWarning: overflow encountered in double_scalars
advntr/vntr_finder.py:193-194: RuntimeWarning: divide by zero encountered in log
```

These warnings appeared while comparing current vs `HEAD`; they are pre-existing in the
unchanged `identify_frameshift()` path and did not change the comparison result.

## Files changed

- `advntr/vntr_finder.py`
- `tests/test_frameshift_ordering.py`
- `scripts/loc_ratchet.py`
- `AGENTS.md`

## Self-review

- The code change is minimal and localized to the row-ordering site.
- The added test fails on the old behavior for the right reason and passes after the fix.
- The implementation follows the binding ruling: support-count order remains ascending for
  non-tied rows.
- The LOC ratchet and AGENTS documentation now match the actual 1405-line file.
- The stale AGENTS note about `-u` no longer overclaims silent non-convergence; it now
  states the observed `AttributeError` path.

## Concerns

- The seven-file public hg19 corpus did not contain a case whose emitted row order changed
  in practice, so the behavioral proof for the tie case comes from the focused regression
  test rather than the corpus comparison.
- The frameshift statistics path still emits existing runtime warnings on some public BAMs.
  This task did not change that code.

## Fix Round 1 Correction

This section corrects the original report after review.

### Retraction

The prior "seven-file public-corpus call-set comparison" evidence was invalid for the
base-vs-candidate claim it made. The script archived `git archive HEAD`, and at the time
of that run `HEAD` was the task commit itself (`dfc8f64`), so that comparison was against
the task commit rather than against the required base revision. That evidence is retracted
and replaced below.

### Test amendment

`tests/test_frameshift_ordering.py` now pins the exact required tie-break order with a
literal expectation:

- `['D41_1', 'I42_1_A_LEN1']`

That means a deterministic but non-lexical order now fails the test.

### Focused ordering test

To prove the strengthened assertion is real, I temporarily restored the old count-only
sort, ran the focused test to RED, then restored the lexical tie-break and reran it to
GREEN on the final code.

#### RED against the old count-only sort

Command:

```bash
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
python -m unittest tests.test_frameshift_ordering
```

Result:

```text
.F
======================================================================
FAIL: test_tied_rows_do_not_depend_on_insertion_history (tests.test_frameshift_ordering.TestFrameshiftOrdering)
Changing which tied state is seen first must not reorder the final rows.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests/test_frameshift_ordering.py", line 99, in test_tied_rows_do_not_depend_on_insertion_history
    self.assertEqual(self._ordered_states(insertions_first), expected)
AssertionError: Lists differ: ['I42_1_A_LEN1', 'D41_1'] != ['D41_1', 'I42_1_A_LEN1']

First differing element 0:
'I42_1_A_LEN1'
'D41_1'

- ['I42_1_A_LEN1', 'D41_1']
+ ['D41_1', 'I42_1_A_LEN1']

----------------------------------------------------------------------
Ran 2 tests in 0.002s

FAILED (failures=1)
```

Interpretation:

- The stricter test now fails specifically when the insertion-first tied case does not
  land in lexical order.

#### GREEN on the final code

Command:

```bash
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
python -m unittest tests.test_frameshift_ordering
```

Result:

```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.002s

OK
```

Interpretation:

- With the shipped lexical tie-break restored, the strengthened test passes.

### Gate on the amended tree

Command:

```bash
set -o pipefail
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
make gate
```

Result:

```text
no upstream remote: ok
python scripts/loc_ratchet.py
loc ratchet: ok (70 files checked)
python setup_hmm.py build_ext --inplace
python -m unittest discover tests
...
wrote /tmp/tmpSS7dOQ/tier2_manifest.json
VERIFIED identical against /tmp/tmpDr9gGt/tier2_manifest.json
...
Ran 226 tests in 40.098s

OK (skipped=1)
coverage run --source=advntr,hmm -m unittest discover tests
...
wrote /tmp/tmpzZo9Vr/tier2_manifest.json
VERIFIED identical against /tmp/tmp0kjbKy/tier2_manifest.json
...
Ran 226 tests in 42.436s

OK (skipped=1)
coverage 26.0% (baseline 8.0%)
gate: PASS
```

Interpretation:

- The amended tree is green end-to-end.
- `loc ratchet: ok (70 files checked)` is expected here because the tracked report file is
  now part of the tree.

### Corrected seven-file public-corpus call-set comparison

Requirement: compare the current candidate against base revision
`5a14d91488eb9f11ae14df3410f1acac0946c004`, not against task commit `dfc8f64`.

I materialized a clean archive of base revision `5a14d91488eb9f11ae14df3410f1acac0946c004`
into a temporary tree, rebuilt it with `python setup_hmm.py build_ext --inplace`, and
compared its real `find_frameshift_from_alignment_file()` results against the current
candidate worktree on the seven public hg19 BAMs.

Command:

```bash
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
python - <<'PY'
import json
import os
import shutil
import subprocess
import tempfile

base_rev = '5a14d91488eb9f11ae14df3410f1acac0946c004'
repo = '/home/bernt-popp/development/adVNTR/.worktrees/calling-quality'
data_root = '/home/bernt-popp/development/VNtyper/tests/data'
bams = [
    'example_6449_hg19_subset.bam',
    'example_66bf_hg19_subset.bam',
    'example_6c28_hg19_subset.bam',
    'example_7a61_hg19_subset.bam',
    'example_a5c1_hg19_subset.bam',
    'example_b178_hg19_subset.bam',
    'example_dfc3_hg19_subset.bam',
]
code = r'''
import json
import sys
from advntr_harness.capture import build_finder
bam = sys.argv[1]
db = sys.argv[2]
finder, _reference = build_finder(db)
rows = finder.find_frameshift_from_alignment_file(bam, []) or []
print(json.dumps(rows))
'''

snapshot_dir = tempfile.mkdtemp(prefix='task1-base-')
try:
    archive = subprocess.Popen(['git', 'archive', base_rev], cwd=repo, stdout=subprocess.PIPE)
    untar = subprocess.Popen(['tar', '-x', '-C', snapshot_dir], stdin=archive.stdout)
    archive.stdout.close()
    untar.wait()
    archive.wait()

    subprocess.check_call(['python', 'setup_hmm.py', 'build_ext', '--inplace'], cwd=snapshot_dir)

    summary = []
    total_base_rows = 0
    total_candidate_rows = 0
    for bam_name in bams:
        bam_path = os.path.join(data_root, bam_name)
        candidate_db = os.path.join(repo, 'tests', 'golden', 'models', 'hg19_muc1.db')
        base_db = os.path.join(snapshot_dir, 'tests', 'golden', 'models', 'hg19_muc1.db')
        candidate_rows = json.loads(subprocess.check_output(['python', '-c', code, bam_path, candidate_db], cwd=repo))
        base_rows = json.loads(subprocess.check_output(['python', '-c', code, bam_path, base_db], cwd=snapshot_dir))
        candidate_set = sorted([tuple(row) for row in candidate_rows])
        base_set = sorted([tuple(row) for row in base_rows])
        total_base_rows += len(base_rows)
        total_candidate_rows += len(candidate_rows)
        summary.append({
            'bam': bam_name,
            'base_count': len(base_rows),
            'candidate_count': len(candidate_rows),
            'order_equal': base_rows == candidate_rows,
            'set_equal': base_set == candidate_set,
        })
    print(json.dumps({'base_rev': base_rev,
                      'summary': summary,
                      'total_base_rows': total_base_rows,
                      'total_candidate_rows': total_candidate_rows}, indent=2, sort_keys=True))
finally:
    shutil.rmtree(snapshot_dir)
PY
```

Result:

```text
advntr/vntr_finder.py:194: RuntimeWarning: divide by zero encountered in double_scalars
  chi_square_val = -2 * numpy.log(sequencing_error_prob / frameshift_prob)
{
  "base_rev": "5a14d91488eb9f11ae14df3410f1acac0946c004",
  "summary": [
    {
      "bam": "example_6449_hg19_subset.bam",
      "base_count": 2,
      "candidate_count": 2,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_66bf_hg19_subset.bam",
      "base_count": 2,
      "candidate_count": 2,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_6c28_hg19_subset.bam",
      "base_count": 7,
      "candidate_count": 7,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_7a61_hg19_subset.bam",
      "base_count": 1,
      "candidate_count": 1,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_a5c1_hg19_subset.bam",
      "base_count": 2,
      "candidate_count": 2,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_b178_hg19_subset.bam",
      "base_count": 2,
      "candidate_count": 2,
      "order_equal": true,
      "set_equal": true
    },
    {
      "bam": "example_dfc3_hg19_subset.bam",
      "base_count": 3,
      "candidate_count": 3,
      "order_equal": true,
      "set_equal": true
    }
  ],
  "total_base_rows": 19,
  "total_candidate_rows": 19
}
```

Interpretation:

- This is the corrected, actual base-vs-candidate comparison.
- The comparison is non-empty: 19 emitted rows on both sides across the seven public hg19
  BAMs.
- All seven files matched on emitted call sets (`set_equal: true`).
- On this public subset, raw row order also matched (`order_equal: true` for all seven),
  so the strengthened synthetic regression test remains the proof that the lexical
  tie-break is enforced when a tie does occur.
