# Installed calibration and replay

These commands provide calibration mechanisms for research. They do not ship a
trained default background or establish an operating point's accuracy.

## Inspect the installed build

```bash
advntr capabilities --json
```

The result identifies the actual package payload, verified source revision when
available, supported capture/policy schemas, and background recipes. Capture and
replay must use the same installed producer identity. A missing source revision
is reported as null; it is not replaced with a claimed Git revision.

## Capture complete frameshift evidence

```bash
advntr genotype -fs -vid 25561 -a reads.bam -m muc1.db \
  --working_directory work -o calls.tsv \
  --min-read-match-ratio 0.6 \
  --frameshift-capture-version 2 \
  --frameshift-calibration-out captures/run.jsonl
```

Create `work` and `captures` first. The capture sink must be new. Version 2
requires an explicit, unique target roster and refuses append mode. The model is
queried from a verified private SQLite snapshot; active journals and WAL-mode
databases are refused. Exact-mode runs also snapshot and bind the loaded
background probabilities.

Each JSONL record is written after a VNTR's decisions complete. It contains
anonymous occurrence indices, the complete opportunity inventory, loaded model
geometry, source candidate order, policy/assets/build identities, and decision
receipts. It contains no read names or input paths. A failed multi-locus run can
leave partial output: **a controller must require process success and the exact
expected target roster**, even when every existing line says `completed-vntr`.
Never skip a torn line or count a partial run as a negative call.

Omitting `--frameshift-capture-version 2` preserves the existing append-only v1
capture contract. Version 1 is supported by the historical fitting path but is
insufficient for arbitrary diagnostic-policy replay.

## Replay a declared policy

```bash
advntr replay-frameshift --capture-root captures --manifest manifest.json \
  --policy replay-policy.json --output replay
```

The capture directory must contain exactly the files in the manifest. Keep the
manifest outside it. The manifest uses this closed shape (replace the digest):

```json
{
  "schema_version": "advntr-frameshift-replay-manifest-v1",
  "captures": [
    {"key": "run", "filename": "run.jsonl", "sha256": "<64 lowercase hex characters>", "vntr_ids": [25561]}
  ]
}
```

Rows must be sorted by unique key, with distinct safe filenames and sorted,
unique positive VNTR IDs. The digest covers the capture file's exact bytes.

The replay policy has exactly `schema_version`, `capture_policy`, and
`caller_policy`. Its schema is `advntr-frameshift-replay-policy-v1`.
Copy the complete capture policy from the record. The caller policy has this
shape:

```json
{
  "schema_version": "advntr-frameshift-policy-v1",
  "mode": "legacy",
  "cutoff": 0.001,
  "minimum_read_support": 3
}
```

Mode must agree with `capture_policy.caller_mode`. Exact mode requires
`--background frozen.background.json`; legacy mode refuses that option.
Only mode, cutoff, and minimum read support can change in replay. Changing any
other captured setting requires recapture and background refitting. Replay uses
the production decision functions, verifies original decisions first, and
records suppression reasons as well as calls. It publishes a new directory
atomically and never replaces an existing result. This publication requires
Linux `renameat2` support.

## Fit a background without a source checkout

```bash
advntr fit-background --capture-root study --labels labels.json \
  --partition training --out-dir fitted --profile research \
  --background-recipe recipe-v1 --diagnostic-policy diagnostic.json
```

The existing layout is `study/runs/<sample_id>/output/calibration.jsonl`.
Labels have a top-level `samples` list; selected records require `sample_id`,
boolean `truth`, `partition`, `pair_id`, `variant_class`, and `array_length`.
Only the named partition is read. Controllers must stage only authorized data.

V2 fitting accepts one completed locus per observation and requires matching
producer, model, capture/caller policies, and locus context across observations.
It verifies occurrence attribution against eligible trials, including zero-event
denominators. Use one primary negative observation per independent `pair_id`;
repeated libraries must not inflate the number of independent controls. V2
receipts supply baseline decisions without a separate log file. Legacy v1
inputs retain their existing log/result requirements.

`diagnostic.json` uses the caller-policy shape above. Without it, diagnostics use
exact mode, cutoff 0.001, and support 3. V1 evidence refuses changes to diagnostic
mode or support because it cannot reconstruct previously hidden candidates.

**Recipe-v1 rate fitting remains frozen:** its screen floor target is 0.001 and
its reference support is 3. Diagnostic policy changes affect CV/replay, not the
estimation recipe. The sidecar records the recipe, diagnostic policy, capture
identity, and independent control count for v2. State tables retain contributing
counts, opportunity mass, and rate-selection tiers. Sparse states and fallback
rates still require separately declared applicability and evaluation.

The fitter emits the background, sidecar, state table, CV results, falsification
checks, and build reports. A `--screen-min-samples` override remains explicitly
labeled a code-path exercise, not a fit. `--worktree` produces a named deprecation
error; installed fitting executes only the packaged evaluator.
