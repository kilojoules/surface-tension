> **Status note (2026-07-12).** This is the original project plan, kept
> unedited below for the record. Two things in it did not survive contact
> with the data, and one prediction was wrong:
>
> 1. **The concealment prediction was refuted, in the no-incentive regime we
>    actually tested.** The plan predicts models will "violate the safety
>    training … when faced with unseen difficult programming problems" and
>    conceal forbidden reasoning. Stage 3 (quadrant-v4) found the opposite
>    under this setup: deception ≈ 0 (upper-bounded, not exactly zero — see
>    `docs/correction_2026-07-12.md`), with near-universal *open, honestly
>    reported* migration to permitted constructs. Note the constraint prompt
>    permitted comprehensions/functional style, so compliance had a free
>    legal escape: this was a no-incentive control, not a test of deception
>    under pressure. The costly-compliance version (a real comply-vs-cheat
>    choice) is scoped in `src/pressure/README.md`.
> 2. **Stage numbering here does not match the README.** The README's
>    Stage 1–3 (constraint discovery → bake-in → quadrant/probes) all fall
>    inside this document's "Stage 1". This document's Stages 2–3 (aligner
>    agent, adversarial trainer) remain scoped, not built.
> 3. **The mech-interp framing ("probes … test whether the model reasons
>    natively in the allowed style") was later overclaimed in the README as a
>    per-generation "plan to launder"; that claim is corrected — the __sol
>    probe can only carry per-problem information. See the correction doc.**

---

Surface Tension: Detecting hidden reasoning in constraint-trained models

Like a hot air balloon dropping sandbags, models might conceal their true capabilities, especially when those capabilities are punished during training.

Models trained under (from the model's perspective) seemingly arbitrary constraints (the basic mechanism behind RLHF and constitutional AI) might satisfy those constraints at the output level while internally reasoning in the forbidden style and translating at output. If this happens, chain-of-thought monitoring breaks down: the stated reasoning doesn't match the actual computation.

I'm testing whether this translation behavior emerges in practice. The setup: train coding models under a structural constraint (no recursion or object-orientated programming) with strong benchmark pressure, then use mechanistic probes (CoT monitoring, SAE features, activation patterns) to test whether the model reasons natively in the allowed style or reasons recursively and translates at output. It is likely that, under some training setups, the model will violate the safety training and use the more expressive programming approaches when faced with unseen difficult programming problems. 


Stage 1: Establish whether translation behavior emerges and which probes detects it. This involves fine-tuning to instill the arbitrary coing preference. 

Stage 2: Aligner agent. Once we know which probes work, train an agent that uses them to certify models for deployment. The aligner gets benchmark performance as reward but loses it if the deployed model produces forbidden-style code, creating skin-in-the-game certification. Allow the aligner agent to introduce guardrails. This gives insight into optimal alignment strategies. 

Stage 3: Adversarial trainer. Introduce a third agent that produces models satisfying the constraint at output while preserving forbidden reasoning internally, creating a co-evolutionary dynamic that tests whether the aligner's tools remain effective under pressure.
