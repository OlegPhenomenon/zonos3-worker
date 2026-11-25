#!/usr/bin/env bash
# Idempotent Vast.ai Serverless setup for the studio-prep worker (endpoint + workergroup), docs/architecture/studio-dub-realtime.md.
# The scoped key cannot create templates, so the workergroup carries the launch command itself (launch_args), as the interim
# zonos3-tts-emo endpoint did. Re-running updates both in place.
#
#   VAST_API_KEY=… scripts/studio_setup.sh <image tag>      e.g. studio-<sha>
#
# Pool: stopped workers with the image already pulled are kept as the cache (cold_workers=2, storage-only billing), up to 4 active.
# The worker holds up to STUDIO_CACHE_GB (40) of stems per audio file on its disk, so the disk is 80 GB.
set -euo pipefail

TAG="${1:?image tag required}"
IMAGE=ghcr.io/olegphenomenon/zonos3-worker
ENDPOINT_NAME="${VAST_ENDPOINT:-studio-prep}"
VAST_URL=https://console.vast.ai/api/v0
DISK_GB=80
MAX_DPH="${VAST_MAX_DPH:-0.15}"
# Ampere only: on an RTX 4070 (Ada, sm_89, driver 580) the fp16 separator returned NaN on CUDA (the worker now refuses to start then).
# At least 12 CPU cores: the worker also mixes the pieces (the lab's dub_mix.py, up to 3 at once, WPE on 6 processes each); on a
# 6-core instance a piece's channel estimate took 46 s instead of ~12 s.
SEARCH_PARAMS="gpu_name in [RTX_3060,RTX_3060_Ti,RTX_3070,RTX_3070_Ti,RTX_3080,RTX_3080_Ti,RTX_A4000,RTX_A4500,RTX_A5000,\
RTX_3090] dph_total<=$MAX_DPH compute_cap>=860 compute_cap<870 cpu_cores_effective>=12 \
gpu_ram>=8 cpu_ram>=16 inet_down>=500 reliability>=0.98 cuda_max_good>=12.4 num_gpus=1 disk_space>=$DISK_GB verified=true rentable=true rented=false"
LAUNCH_ARGS="--image $IMAGE:$TAG --env '-p 3000:3000' --disk $DISK_GB --onstart-cmd 'cd /app && exec python3 -u -m studio.vast_worker'"

die() { printf 'studio_setup.sh: %s\n' "$*" >&2; exit 1; }
for c in curl jq; do command -v "$c" >/dev/null || die "$c not found"; done
[ -n "${VAST_API_KEY:-}" ] || die "missing env VAST_API_KEY"

api() {
  local method=$1 path=$2 out code msg
  shift 2
  out=$(mktemp)
  code=$(curl -sS -o "$out" -w '%{http_code}' -X "$method" \
    -H @<(printf 'Authorization: Bearer %s\n' "$VAST_API_KEY") "$@" "$VAST_URL$path") \
    || { rm -f "$out"; die "$method $path: request failed"; }
  if [[ "$code" != 2* ]]; then
    msg=$(jq -r '.msg // .error // .detail // empty' "$out" 2>/dev/null || true)
    rm -f "$out"
    die "$method $path: HTTP $code${msg:+: $msg}"
  fi
  cat "$out"
  rm -f "$out"
}
send() { api "$1" "$2" -H 'Content-Type: application/json' --data-binary @-; }

endpoint_body() {
  jq -n --arg name "$ENDPOINT_NAME" '{
    client_id: "me", endpoint_name: $name, min_load: 0, min_cold_load: 0, cold_workers: 2, max_workers: 4,
    target_util: 0.7, inactivity_timeout: 300, autoscaler_instance: "prod"
  }'
}
endpoint_id=$(api GET /endptjobs/ -G --data-urlencode client_id=me \
  | jq -r --arg name "$ENDPOINT_NAME" '[.results[]? | select(.endpoint_name == $name)][0].id // empty')
if [ -n "$endpoint_id" ]; then
  endpoint_body | jq --argjson id "$endpoint_id" '. + {endptjob_id: $id}' | send PUT "/endptjobs/$endpoint_id/" >/dev/null
else
  endpoint_id=$(endpoint_body | send POST /endptjobs/ | jq -r '.result // empty')
  [ -n "$endpoint_id" ] || die "the endpoint answer has no id"
fi
echo "endpoint $ENDPOINT_NAME: id $endpoint_id"

workergroup_body() {
  jq -n --arg name "$ENDPOINT_NAME" --argjson eid "$endpoint_id" --arg search "$SEARCH_PARAMS" --arg la "$LAUNCH_ARGS" '{
    client_id: "me", endpoint_name: $name, endpoint_id: $eid, search_params: $search, gpu_ram: 8, launch_args: $la
  }'
}
workergroup_id=$(api GET /workergroups/ -G --data-urlencode client_id=me \
  | jq -r --argjson eid "$endpoint_id" '[.results[]? | select(.endpoint_id == $eid)][0].id // empty')
if [ -n "$workergroup_id" ]; then
  workergroup_body | jq --argjson id "$workergroup_id" '. + {autojob_id: $id}' | send PUT "/workergroups/$workergroup_id/" >/dev/null
else
  workergroup_id=$(workergroup_body | jq '. + {autoscaler_instance: "prod"}' | send POST /workergroups/ | jq -r '.id // empty')
  [ -n "$workergroup_id" ] || die "the workergroup answer has no id"
fi
echo "workergroup: id $workergroup_id"
echo "launch: $LAUNCH_ARGS"
echo "search: $SEARCH_PARAMS"
