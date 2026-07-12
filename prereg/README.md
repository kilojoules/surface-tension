# Pre-registrations — timestamping status and policy

Two pre-registrations live here. Both are honest documents written before
their analyses were run, and both contain pre-registered decision rules and
limitations that later mattered. But their **timestamps are git-only and not
independently verifiable**:

| prereg | internal stamp | entered git | in commit |
|---|---|---|---|
| `strict_variants_2026-06-14.md` | 2026-06-14 | 2026-06-14 | `7cc2404` — the same commit as the Task 1+2 **results** |
| `quadrant_v4_deception_2026-07-03.md` | 2026-07-03 | 2026-07-04 | `be5edf1` — the same commit as the Stage-3 **results** |

A prereg that first appears in the same author-controlled commit as the
results it governs cannot prove, to anyone but the author, that the
predictions preceded the data. The git record is compatible with either
order. This does not retract anything in these documents — but it means
their prereg status is a claim about the author's process, not a verifiable
property of the artifact, and readers should weight it accordingly.

**Policy from 2026-07-12 onward:** any future pre-registration in this
project (in particular, the deception-pressure experiment scoped in
`src/pressure/README.md`) will be independently timestamped BEFORE data
generation begins — via an OSF registration or Zenodo deposit of the prereg
document (preferred), or at minimum a signed git tag pushed to the public
remote — and the registration link recorded in the prereg header. A
deception-positive result will need the credibility that a null did not.
