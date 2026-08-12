# CLAUDE.md

Project instructions for this repository live in **[AGENTS.md](AGENTS.md)**. Read it
before making any change — it is the single source of truth for setup, commands, code
style, testing, git conventions, and the repo-specific traps.

This file exists only so Claude Code picks the instructions up automatically. Keep it a
pointer: add new guidance to `AGENTS.md`, not here.

Quick reference:

- Verify with `make gate` before proposing a PR.
- Fast feedback loop: `make build && make test`, run from the repo root.
- Everything runs in the `envadvntr` conda env (Python 2.7). `muscle` must be on `$PATH`.
- This is a **hard fork**. See [FORK.md](FORK.md); do not add an `upstream` remote.
- Decoder changes must be proven byte-identical. See AGENTS.md § Testing.
