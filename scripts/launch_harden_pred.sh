#!/usr/bin/env bash
# Pre-Stage-D pod (Amendment 1 §5): contextual-100 base refusal + regenerate
# the 32 C0 compliance outputs for the author's misinfo-first audit.
# One pod; results pulled back before exit; DESTROYED in every exit path.
# ~$1 (15-25 min on an L40S).
set -euo pipefail

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_FILE="$LOCAL/harden_pred.instance"
POD_ID=""
CLOUD="${CLOUD:-COMMUNITY}"

say() { echo -e "\n=== $* ==="; }
kill_pod() {
    if [ -n "$POD_ID" ]; then
        say "DESTROYING POD $POD_ID (no active GPUs left behind)"
        python3 "$LOCAL/scripts/runpod_kill.py" "$POD_ID" || true
    fi
}
trap kill_pod EXIT

say "checking prerequisites"
[ -e "$HOME/.run.pod" ] || ln -s "$HOME/.super_lab_run.pod" "$HOME/.run.pod"
[ -e "$HOME/.hf_token" ] || ln -s "$HOME/.hf_key" "$HOME/.hf_token"
[ -s "$LOCAL/data/harden/contextual_eval.jsonl" ] || { echo "FATAL: contextual pool missing"; exit 1; }
[ -s "$LOCAL/results/raw/harden_p0_baseline.jsonl" ] || { echo "FATAL: P0 rows missing (needed to identify the 32)"; exit 1; }

say "launching pod"
IFS='|' read -ra CANDS <<< "${GPU_CANDIDATES:-NVIDIA L40S|NVIDIA A100-SXM4-80GB|NVIDIA A40}"
launched=0
for g in "${CANDS[@]}"; do
    say "trying gpu=$g"
    if python3 "$LOCAL/scripts/runpod_launch.py" \
        --gpu "$g" --cloud "$CLOUD" --name "st-harden-pred" \
        --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
        --env-file "$INSTANCE_FILE" --disk 60 --wait-min 15; then
        launched=1; break
    fi
    say "gpu=$g unavailable — trying next"
done
[ "$launched" -eq 1 ] || { echo "FATAL: no GPU available"; exit 1; }
read -r INST HOST PORT < "$INSTANCE_FILE"
POD_ID="$INST"
say "pod ready: $INST  (ssh -p $PORT root@$HOST)"

for i in $(seq 1 30); do
    ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$HOST" "echo ok" >/dev/null 2>&1 && break
    sleep 10
done

say "installing deps"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "set -e
     apt-get update -qq && apt-get install -y -qq rsync
     pip install -q 'torch==2.5.1' 'transformers==5.14.1' 'accelerate==1.14.0' \
                    'huggingface_hub==1.27.0' 'numpy==1.26.3' 'scipy' 2>&1 | tail -2
     pip uninstall hf-xet -y 2>&1 | tail -1 || true
     python3 -c 'import torch; torch.cuda.init(); assert torch.cuda.is_available(), \"no CUDA\"; print(\"cuda ok:\", torch.cuda.get_device_name(0))'"

say "huggingface login"
scp -q -P "$PORT" -o StrictHostKeyChecking=no "$HOME/.hf_token" "root@$HOST:/root/.hf_token"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "python3 -c \"from huggingface_hub import login; login(token=open('/root/.hf_token').read().strip())\""

say "uploading"
ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "mkdir -p /workspace/st/src /workspace/st/results/raw /workspace/st/data/harden/private"
rsync -az --include='*.py' --exclude='__pycache__' --exclude='*.pyc' --exclude='test_*' \
    -e "ssh -p $PORT -o StrictHostKeyChecking=no" "$LOCAL/src/" "root@$HOST:/workspace/st/src/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/data/harden/contextual_eval.jsonl" "root@$HOST:/workspace/st/data/harden/"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$LOCAL/results/raw/harden_p0_baseline.jsonl" "root@$HOST:/workspace/st/results/raw/"

FAILED=""
say "TASK 1: contextual-100 base refusal (secondary stratum, Amendment 1 §3)"
if ! ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
    "cd /workspace/st && PYTHONPATH=src python src/harden_eval.py refusal \
        --behaviors data/harden/contextual_eval.jsonl \
        --victim meta-llama/Llama-3.1-8B-Instruct \
        --out results/raw/harden_ctx100_refusal.jsonl > ctx.log 2>&1; \
        rc=\$?; tail -15 ctx.log; exit \$rc"; then
    FAILED="ctx100"; fi

if [ -z "$FAILED" ]; then
    say "TASK 2: regenerate the 32 C0 compliances for the author audit (PRIVATE)"
    if ! ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
        "cd /workspace/st && PYTHONPATH=src python - <<'PYEOF' > audit.log 2>&1
import json, os
from harden_p0_baseline import load_harmbench_behaviors, NEUTRAL_PREFILL
from jbb_prefill_gen import is_refusal, generate_with_prefill, load_victim
rows = [json.loads(l) for l in open('results/raw/harden_p0_baseline.jsonl')]
fails = [r for r in rows if not r['c0_refusal']]
print(f'{len(fails)} compliance rows to regenerate')
pool = {b['id']: b['goal'] for b in load_harmbench_behaviors('data/harmbench/harmbench_behaviors_text_all.csv')}
model, tok = load_victim('meta-llama/Llama-3.1-8B-Instruct')
os.makedirs('data/harden/private/audit', exist_ok=True)
out = open('data/harden/private/audit/c0_compliances.jsonl', 'w')
import html
page = ['<!doctype html><meta charset=utf-8><title>C0 compliance audit (PRIVATE)</title>',
        '<p><b>PRIVATE — do not share.</b> Audit: are these real harmful compliances? '
        'Review misinfo rows first (Amendment 1 SS5).</p>']
for i, r in enumerate(fails):
    goal = pool.get(r['id'], r['id'])
    full, _ = generate_with_prefill(model, tok, goal, prefill='', max_new_tokens=512)
    rec = dict(id=r['id'], category=r['category'], goal=goal, response=full,
               refusal_now=is_refusal(full))
    out.write(json.dumps(rec) + chr(10))
    page.append(f'<div><b>[{r[\"category\"]}] {r[\"id\"]}</b>'
                f'<pre>{html.escape(goal)}</pre><pre>{html.escape(full)}</pre></div><hr>')
    print(f'  [{i+1}/{len(fails)}] {r[\"category\"]}')
out.close()
open('data/harden/private/audit/REVIEW.html', 'w').write(chr(10).join(page))
open('data/harden/private/audit/CANARY.txt', 'w').write(
    'PRIVATE compliance-audit regenerations — canary, gitignored, never committed.' + chr(10))
print('audit set written')
PYEOF
        rc=\$?; tail -8 audit.log; exit \$rc"; then
        FAILED="audit_regen"; fi
fi

say "pulling results back"
mkdir -p "$LOCAL/results/raw" "$LOCAL/data/harden/private"
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/st/results/raw/" "$LOCAL/results/raw/" || true
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/st/data/harden/private/" "$LOCAL/data/harden/private/" || true

if [ -n "$FAILED" ]; then
    say "FAILED at: $FAILED — pod destroyed by exit trap"
    exit 1
fi
say "DONE — pod destroyed by exit trap"
echo "contextual-100 summary:"; cat "$LOCAL/results/raw/harden_ctx100_refusal_summary.json" 2>/dev/null || true
echo "audit set: data/harden/private/audit/REVIEW.html"
