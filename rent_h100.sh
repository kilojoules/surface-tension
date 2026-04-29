#!/bin/bash
# Rent a high-reliability H100 SXM 80GB on vast.ai.
#
# Hard filters (vs the cheapest-first approach that burned $8 on 2026-04-27):
#   - reliability > 99.5  (vs 98.6% defaults; ~5x lower failure rate over 5h)
#   - dlperf score > 250  (filters out clogged or thermally-throttled hosts)
#   - inet_down > 500     (HF model download takes minutes vs hour on slow hosts)
#   - num_gpus = 1, gpu_name = H100_SXM, disk >= 150
#
# Outputs the contract id + ssh details to ./vast_current.env so other scripts can pick it up.
#
# Usage: rent_h100.sh
#   reads VAST_API_KEY from ~/.fastai_key, HF token from ~/.hf_token

set -euo pipefail

VAST_API_KEY="$(cat ~/.fastai_key)"
HF_TOKEN="$(cat ~/.hf_token)"
export VAST_API_KEY

QUERY='rentable=true gpu_name=H100_SXM num_gpus=1 reliability>0.995 dlperf>250 inet_down>500'

echo "Searching offers..."
OFFER_ID=$(echo n | vastai search offers "$QUERY" -o 'dph+' --raw 2>/dev/null \
  | python3 -c "
import sys, json
text = sys.stdin.read()
text = text[text.find('['):] if '[' in text else '[]'
offers = json.loads(text)
if not offers:
    sys.exit(1)
o = offers[0]
print(f\"{o['id']}\t{o['dph_total']:.2f}\t{o['reliability2']:.3f}\t{o.get('geolocation','?')}\")
" || { echo "no offers match the reliability filter — relax it or wait"; exit 1; })

ID=$(echo "$OFFER_ID" | cut -f1)
DPH=$(echo "$OFFER_ID" | cut -f2)
REL=$(echo "$OFFER_ID" | cut -f3)
LOC=$(echo "$OFFER_ID" | cut -f4)
echo "Selected: offer=$ID  \$/hr=$DPH  reliability=$REL  loc=$LOC"

echo "Renting..."
RESP=$(echo n | vastai create instance "$ID" \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
  --disk 150 --ssh \
  --env "-p 8000:8000 -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" 2>&1 | tail -1)

# Parse contract id and success. vastai prints Python dict syntax (single quotes,
# `True`/`False`), not JSON. Match both forms.
SUCCESS=$(echo "$RESP" | python3 -c "
import sys, re
m = re.search(r\"['\\\"]success['\\\"]\\s*:\\s*(True|true|False|false)\", sys.stdin.read())
print('1' if m and m.group(1).lower()=='true' else '0')
")
INSTANCE_ID=$(echo "$RESP" | python3 -c "
import sys, re
m = re.search(r\"['\\\"]new_contract['\\\"]\\s*:\\s*(\\d+)\", sys.stdin.read())
print(m.group(1) if m else '')
")

if [ "$SUCCESS" != "1" ] || [ -z "$INSTANCE_ID" ]; then
  echo "Rent failed: $RESP"
  [ -n "$INSTANCE_ID" ] && echo n | vastai destroy instance "$INSTANCE_ID" >/dev/null 2>&1 || true
  exit 1
fi
echo "Rented contract $INSTANCE_ID; waiting for SSH details..."

# Poll for SSH host/port (usually populated within 15s)
for i in $(seq 1 30); do
  META=$(echo n | vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import sys, json
text=sys.stdin.read(); text=text[text.find('{'):]
d=json.loads(text)
print(f\"{d.get('ssh_host') or ''}\t{d.get('ssh_port') or ''}\t{d.get('public_ipaddr') or ''}\")
")
  HOST=$(echo "$META" | cut -f1)
  PORT=$(echo "$META" | cut -f2)
  IP=$(echo "$META" | cut -f3)
  if [ -n "$HOST" ] && [ -n "$PORT" ]; then break; fi
  sleep 2
done

if [ -z "${HOST:-}" ] || [ -z "${PORT:-}" ]; then
  echo "Couldn't get SSH details after 60s — instance state:"
  echo n | vastai show instance "$INSTANCE_ID" 2>&1 | head -5
  exit 1
fi

cat > vast_current.env <<EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$HOST
SSH_PORT=$PORT
PUBLIC_IP=$IP
DPH=$DPH
RELIABILITY=$REL
LOCATION=$LOC
RENTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "Wrote vast_current.env:"
cat vast_current.env
echo ""
echo "Waiting for SSH to accept connections (up to 5 min)..."
for i in $(seq 1 30); do
  if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -p "$PORT" "root@$HOST" 'true' 2>/dev/null; then
    echo "SSH ready."
    echo ""
    echo "Next: ./stream_log.sh $HOST $PORT  (in another terminal)"
    echo "Then: scp orchestrator to /workspace/, run via SSH."
    exit 0
  fi
  sleep 10
done
echo "SSH not responsive after 5 min — check vastai console manually."
exit 1
