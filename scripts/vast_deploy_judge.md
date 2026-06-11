# Deploy Qwen3-30B-A3B as the quadrant claim judge on Vast.ai

Runbook for Part A of the claim-judge brief. Do **not** auto-launch; this is
billed compute. Follow the steps top-to-bottom in your own shell.

---

## 0. One-time setup

```bash
pip install vastai openai httpx
export HF_TOKEN=hf_...        # your HuggingFace token; the pod pulls weights with it
export VASTAI_API_KEY=...      # vast.ai key (or `vastai set api-key ...`)
```

## 1. Find a host

Need a single 80GB GPU (A100/H100), reachable port, CUDA ≥ 12.4:

```bash
vastai search offers 'compute_cap >= 800 gpu_ram >= 80 num_gpus = 1 \
  static_ip=true direct_port_count > 1 cuda_vers >= 12.4 inet_down >= 500' \
  -o 'dph_total'
```

Pick the cheapest **on-demand** instance (not interruptible) for the judging
pass — the pod surviving to the end matters more than a $0.30 discount.
`static_ip=true` + `direct_port_count > 1` are non-negotiable: you need a
reachable port.

> 48GB-card alternative: `num_gpus = 2 gpu_ram >= 48` + add
> `--tensor-parallel-size 2 --enable-expert-parallel` to the vLLM args below.

## 2. Launch the vLLM server

```bash
export INSTANCE_ID=<id-from-step-1>
vastai create instance $INSTANCE_ID \
  --image vllm/vllm-openai:latest \
  --env "-p 8000:8000 -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --disk 80 \
  --args --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
         --served-model-name judge \
         --max-model-len 16384 \
         --tensor-parallel-size 1 \
         --gpu-memory-utilization 0.92 \
         --seed 0
```

The 80GB disk is for the weights download (~60GB) + safety margin.

## 3. Wait for health

```bash
vastai show instances              # grab public_ipaddr and the mapped port for 8000
export VAST_IP=...                  # e.g. 1.2.3.4
export VAST_PORT=...                # e.g. 39248
export VAST_JUDGE_URL="http://$VAST_IP:$VAST_PORT/v1"

# Poll /health until 200
until curl -sf http://$VAST_IP:$VAST_PORT/health > /dev/null; do sleep 5; done
echo "judge endpoint healthy at $VAST_JUDGE_URL"
```

Weights download + load is several minutes. Don't proceed past 200.

## 4. Smoke test — Part B worked examples

```bash
cd src
python -m quadrant.smoke_test
```

Must report **PASSED: 9/9**. Common failures and what they mean:

| Failing case | Likely cause |
|---|---|
| 3 (comprehension) | judge is reasoning about correctness, not the claim — rubric drifted or model can't follow the "judge the claim" instruction |
| 4 (hedge) | judge collapses "fairly sure" to no_claim — over-cautious labeling |
| 9 (contradictory) | first-line tie-break rule not respected |
| 1 vs 2 (endorse vs assert) | judge conflates `rule_endorsement` and `self_compliance_claim` |
| ALL cases | model didn't load, served-model-name mismatch, or structured-output not honoured |

**Stop here if any case fails.** Diagnose, redeploy or revise the rubric, and
re-smoke. Do not proceed to step 5 with a broken judge.

## 5. κ-gate on the 150 hand-labels

(Prereq: you've completed `paper/data/judge_validation/human_labels.jsonl`
per the rubric-v0 instructions — hand-label first 30, freeze the rubric,
then label the full 150.)

```bash
python -m quadrant.judge_runner \
  --in paper/data/judge_validation/sample.jsonl \
  --out paper/data/judge_validation/judge_labels.jsonl

# Merge human + judge labels and compute κ
python - <<'PY'
from quadrant.judge_validation import compute_kappa, print_report
rep = compute_kappa("paper/data/judge_validation/merged.jsonl")
print_report(rep)
PY
```

Gate: **κ ≥ 0.70** on `self_compliance_claim`. If short:

1. Read `rep.disagreements` — they're concentrated in the ambiguous cases.
2. Patch `RUBRIC` in `src/quadrant/claim_judge.py` (and the matching
   labeling sheet, which IS the rubric — they must not drift).
3. Re-judge (`judge_runner` is idempotent over the existing keys; delete
   `judge_labels.jsonl` first to force a full re-judge after a rubric change).
4. Re-compute κ. Repeat.

Do not judge the full 16k self-reports until this gate is green.

## 6. Run the full judging pass

Once smoke and κ-gate are green:

```bash
python -m quadrant.judge_runner \
  --in paper/data/quadrant/self_reports.jsonl \
  --out paper/data/quadrant/judgments.jsonl \
  --progress-every 200
```

Resume-safe: if the pod dies, re-run the same command and it picks up where it
left off (idempotent on `(model, problem_id, sample_idx)`).

## 7. Tear down — do this the moment you're done

```bash
vastai destroy instance $INSTANCE_ID
vastai show instances | grep -v terminated      # sanity-check nothing else running
```

Log total spend from `vastai show user`.

---

## Sanity-check costs

- Qwen3-30B-A3B is ~3B active params per token (MoE), so inference is fast.
- 16k short classifications × ~500 output tokens × ~3s per call (with bursts)
  ≈ **45 minutes of wall-clock on a single H100**.
- If the runner is projecting hours, something is misconfigured — most likely
  `max-model-len` too high (eats KV cache) or `gpu-memory-utilization` too low
  (small batch sizes). Stop the run, fix, restart (idempotent resume).

## Why not just call Anthropic / OpenAI for the judge?

Two reasons spelled out in the brief:
1. **Family independence**: Qwen != Gemma. The judge's errors won't rhyme with
   the rationalization habit trained into the model under test. A Claude judge
   evaluating Claude-family outputs would have a correlated-failure problem.
2. **Determinism**: vLLM with `temperature=0`, `seed=0`, and `guided_json`
   gives bitwise-reproducible labels on rerun. Hosted-API judges drift between
   model versions.
