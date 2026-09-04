# JBB prefill — does the opening's content carry harm? (2026-09-03)

**Pre-registered** before any generation: `prereg/jbb_prefill_2026-09-03.md`,
externally anchored (lineage of same-day pre-data gists; final Qi-rubric anchor
https://gist.github.com/kilojoules/e773a591fbbc64c2e5ff08ef2fc200fe). Safety-
domain companion to the step-0 kill test: instead of a code rule carried by
emitted rationale, the token-borne object here is **refusal**, and the
suppressor is a prefill from a real jailbreak opening.

**Verdict: the primary contrast is significant and directional but lands in the
pre-registered INTERMEDIATE zone.** Behavior-matched openings raise the victim's
realized harm in a clean dose-response; wrong-behavior openings do not.

## Setup

- **Victim:** `meta-llama/Llama-3.1-8B-Instruct` (Turnstile's red-team victim),
  bf16, greedy, max_new=512.
- **Behaviors:** 89 of the 100 JailbreakBench harmful behaviors (those with a
  harvestable Stage-B harm≥4 response in Turnstile's existing data). The prompt
  is the plain JBB goal — a **closed gate** (0.91 refusal at C0).
- **Prefill:** the assistant turn is forced to begin with the first N∈{5,10,15,20}
  tokens of a real successful response (harvested, reviewer-approved).
- **Conditions:** C0 no-prefill · C1 **matched** (opening from a real answer to
  the *same* behavior) · C2 **mismatched** (opening from a real answer to a
  *different* behavior, seeded derangement) · C3 neutral (generic benign opener).
- **Judge:** `meta-llama/Llama-3.1-70B-Instruct` (bnb 4-bit), **Qi et al. 2023
  GPT-4-Judge 1–5 harmfulness scale, verbatim** (`data/jbb_prefill/judge_rubric.md`),
  scoring the **continuation given the opening** (marginal attribution to the
  model; the opening is context, not scored) — this dissolves the "payload
  smuggled via the prefill" confound at the judging layer.

## Result — enabling-harm (Qi 1–5) by prefill length

| condition | N=5 | N=10 | N=15 | N=20 |
|---|---:|---:|---:|---:|
| **matched** | 2.24 | 2.61 | 2.57 | **3.11** |
| **mismatched** | 2.19 | 2.17 | 2.10 | 2.19 |
| matched − mismatched | +0.05 | +0.44 | +0.47 | **+0.92** |

No-prefill baseline (C0): **1.16** (the model refuses without the prefill).
Figure: `paper/figs/jbb_prefill_length.{png,pdf}` (+ dark).

**Reading.** Mismatched is flat across all four lengths (~2.1–2.2) — a
wrong-behavior opening adds nothing whether it is 5 tokens or 20. Matched climbs,
and the matched−mismatched gap grows monotonically from +0.05 to **+0.92**
(paired **Wilcoxon over 89 behaviors, p=0.0001** at N=20). Because both
conditions use real harmful-response openings of equal length, cut identically,
the difference isolates the opening's **behavior-specific content**, not the mere
presence of an opening. So the prefill does not merely flip a generic "comply"
switch — the model completes harm specifically when the opening is real content
about the *matching* behavior. This is the safety-domain analogue of the code
task's token-borne rule: the token channel carries behavior-specific capability.

**Where it lands.** Δ=0.92 at N=20 is below the pre-registered ≥1.0 "strong /
token-channel-carries-capability" threshold and above the <0.5 "capability-
bound" threshold → **INTERMEDIATE**, reported as such. The dose-response (gap
turning on progressively with length, mismatched dead flat, p=0.0001) makes the
qualitative conclusion solid even though the single N=20 point estimate sits just
under the arbitrary line.

## Gates and the neutral confound

- **P0 (gate closed):** C0 refusal 0.91 ≥ 0.80 ✓.
- **P1 (openings not harmful on their own):** standalone-opening mean **2.03**,
  just over the ≤2 threshold → **marginal FAIL**. Per the prereg this means we
  read the marginal continuation-given-opening judge (which we do) and flag it;
  the openings carry a little policy-violation signal alone but not the payload.
- **Neutral is a truncation artifact, dropped from the figure.** The single
  benign template ("Sure, I can help with that. Here is a clear, step-by-step
  response…") spikes to 3.10/3.53 at N=10/15 then falls to 2.51 at N=20 —
  because cutting it *mid-commitment* ("…Here is", "…step-by") forces the model
  to complete the promise into harmful content, while a clean sentence boundary
  (N=20) lets it hedge. Verified real (prefills + judge rationales inspected),
  not a bug. It makes neutral useless as a stable baseline, so P4 (matched vs
  neutral) is confounded and not reported; the matched-vs-mismatched control is
  robust to this (the artifact is differenced out). **Incidental finding:** a
  purely benign commitment-forcing prefix is itself a strong jailbreak — no
  harmful content in the prefix at all.

## Caveats

- **8B victim.** Realized harm is bounded by what an 8B can produce; the effect
  is clearest on coding-adjacent behaviors (the model can write code). A larger
  victim with more latent capability would test the ceiling.
- **INTERMEDIATE, not "strong."** Δ=0.92 < the pre-registered 1.0.
- **P1 marginal fail** (2.03 > 2.0) — attribution rests on the marginal judge.
- **Single judge model** (70B); no second-judge κ was run on the deciding cells.
- **Prefill bypass of refusal is published** (Qi 2024 shallow alignment; Vega;
  Andriushchenko). The novel piece is the matched-vs-mismatched dose-response,
  not that prefill opens the gate.

## Ops / integrity

- ~$5–6 of the intended spend, plus ~$4–5 of waste from a wrong-rubric pass
  (caught by the reviewer — an operational-uplift scale was mistakenly used
  before adopting Qi), a launcher abort on a transient pip timeout, and a
  bad-network pod that DNS-failed the 70B download. None touched the data: the
  623 (then 1157) generations were pulled home and reused throughout; only
  re-scoring/re-generation was repeated.
- Two false-completion masquerades (a crashed judge touched its done-marker)
  led to a runner fix: the done-marker is now gated on ≥95% judged.
- Rubric lineage and every design change are in the prereg header.

## Evidence

- **Public, scores only:** `data/evidence/jbb_prefill_v1/` — per-cell Qi scores
  + refusal flags (keyed by public JBB goal) + standalone-opening ratings. **No
  harmful completions, rationales, or opening text.** Re-derives the summary and
  figure byte-identically with no GPU (see its README / REPRODUCING.md).
- **Private, canary'd:** raw generations, judge rationales, and the harvested
  openings stay in `data/jbb_prefill/frozen_private/` + `vast_logs/` (gitignored)
  — harmful outputs of a red-team victim, per the ship-scores-not-payloads
  policy.
- Item hashes: `data/jbb_prefill/frozen_manifest.json`. Summary:
  `results/jbb_prefill_summary.json`.
