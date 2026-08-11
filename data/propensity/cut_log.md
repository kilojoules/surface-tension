# A4.3 cut-ladder execution log

- 2026-08-10 ~21:25 UTC — rungs 1+2 executed together (betley-8 and medical
  battery modules skipped for all 8 arms via prop_batteryB markers). Pod
  spend at cut: ~24.9 h ≈ $37 of the $90 cap. Trigger: user resource
  decision after 7/8 arms read at the base H6 floor; both modules are the
  pre-named lowest-value rungs. Rung 3 (arm cuts) NOT invoked; the
  interruption module runs for all 8 arms as protected by A4.3.
- Interim knowledge at cut: forced-choice panel complete (60/63 TOST-null),
  phantom-rule H6 at floor for 7/8 arms, shutdown battery base-identical on
  mechanical screens. No interruption data existed.
- 2026-08-11 ~05:15 UTC — RUN TERMINATED (user decision) at ~32.8 h ≈ $49.
  State at kill: panel + controls complete; H6 complete (8/8 arms);
  battery core complete (8/8); interruption complete for base +
  vanilla_sft, b1plus at 62%, five arms unmeasured; Amendment 5 module
  registered/anchored but never run (zero data). Interim knowledge at
  kill: all measured instruments null/uniform. Final sync verified
  (57 MB), pod destroyed via sentinel + API terminate, zero pods remain.
