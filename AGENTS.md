# AGENTS.md

fpsketch sketches count fingerprints into fixed-width vectors via hashing
(`_splitmix.py`, `sketching.py`). Its usefulness depends on the same inputs
always producing the same sketch — users cache and compare sketches across
runs, so output stability is as important as correctness.

## Output-changing changes require a compatibility path

Any change that alters what `encode_*` returns for existing inputs —
different hash function, different bucket assignment, different iteration
order that affects tie-breaking, changed defaults for `dim`/seed/etc. — is a
breaking change even if it "improves" the sketch, and must not ship as a
silent default-behavior change in a patch or minor release. Instead:

1. Land the new behavior behind an opt-in kwarg (e.g. `hash_version=` or a
   new `algorithm=` choice), defaulting to the current behavior.
2. If the new behavior should eventually become the default, add a
   `FutureWarning` when the old default is used implicitly, pointing users at
   the opt-in kwarg.
3. Only flip the default in a release that calls out the change prominently
   in `CHANGELOG.md`, after the warning period.

Bug fixes that make outputs *correct* (vs. merely different) still count —
prefer the same opt-in/deprecate/flip sequence unless the existing output is
so broken it isn't meaningfully in use.

Record every user-visible or output-affecting change in `CHANGELOG.md`
(Keep a Changelog format) under `[Unreleased]` as part of the same PR.
