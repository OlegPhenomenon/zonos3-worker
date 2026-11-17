#!/usr/bin/env bash
# Idempotent Vast.ai Serverless setup for the Zonos TTS worker (template, endpoint, workergroup), adapted from
# talking_photos_gpu/scripts/vast_serverless_setup.sh. Re-running updates all three in place.
#
#   VAST_API_KEY=… scripts/vast_setup.sh <image tag>
#
# Cost policy: Zonos needs ~3 GB VRAM, so the workergroup only admits cheap 8–24 GB Ampere/Ada cards under a price
# cap; Vast's engine recruits by price-performance, so cheap cards win and pricier ones are taken only when nothing
# cheaper is free. cold_workers=3 keeps three stopped workers on different hosts with the image already pulled (storage-only billing),
# which is Vast's equivalent of a warm cache.
set -euo pipefail

TAG="${1:?image tag required}"
IMAGE=ghcr.io/olegphenomenon/zonos3-worker
TEMPLATE_NAME=zonos3-tts
ENDPOINT_NAME="${VAST_ENDPOINT:-zonos3-tts}"
VAST_URL=https://console.vast.ai/api/v0
DISK_GB=16
GPU_RAM_GB=8
MAX_DPH="${VAST_MAX_DPH:-0.15}"
SEARCH_PARAMS="gpu_name in [RTX_3060,RTX_3060_Ti,RTX_3070,RTX_3070_Ti,RTX_3080,RTX_3080_Ti,RTX_A4000,RTX_A4500,\
RTX_A5000,RTX_4060_Ti,RTX_4070,RTX_4070_Ti,RTX_4000Ada,RTX_3090,RTX_4080,RTX_4090] dph_total<=$MAX_DPH \
compute_cap>=860 compute_cap<900 gpu_ram>=$GPU_RAM_GB inet_down>=500 reliability>=0.98 cuda_max_good>=12.4 num_gpus=1 \
disk_space>=$DISK_GB verified=true rentable=true rented=false"

die() { printf 'vast_setup.sh: %s\n' "$*" >&2; exit 1; }
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

template_body() {
  jq -n --arg name "$TEMPLATE_NAME" --arg image "$IMAGE" --arg tag "$TAG" --argjson disk "$DISK_GB" '{
    name: $name, image: $image, tag: $tag, env: "-p 3000:3000",
    runtype: "args", args_str: "vast", use_ssh: false, ssh_direct: false, jup_direct: false,
    recommended_disk_space: $disk, private: true,
    desc: "ZONOS3-Mini TTS: Vast PyWorker on port 3000 (/tts)"
  }'
}
filters=$(jq -cn --arg name "$TEMPLATE_NAME" --arg image "$IMAGE" '{name: {eq: $name}, image: {eq: $image}}')
old=$(api GET /template/ -G --data-urlencode "select_filters=$filters" | jq -c '.templates[0] // empty | {id, hash_id}')
if [ -n "$old" ]; then
  # v0 PUT /template/ is gone (HTTP 410); v1 takes the template's id and hash_id.
  template_hash=$(template_body | jq --argjson o "$old" '. + $o' \
    | VAST_URL=https://console.vast.ai/api/v1 send PUT /template/ | jq -r '.template.hash_id // .hash_id // empty')
else
  template_hash=$(template_body | send POST /template/ | jq -r '.template.hash_id // empty')
fi
[ -n "$template_hash" ] || die "the template answer has no hash_id"
echo "template $TEMPLATE_NAME: $IMAGE:$TAG ($template_hash)"

endpoint_body() {
  jq -n --arg name "$ENDPOINT_NAME" '{
    client_id: "me", endpoint_name: $name, min_load: 0, min_cold_load: 0, cold_workers: 3, max_workers: 3,
    inactivity_timeout: 120, autoscaler_instance: "prod"
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
  jq -n --arg name "$ENDPOINT_NAME" --argjson eid "$endpoint_id" --arg hash "$template_hash" \
    --arg search "$SEARCH_PARAMS" --argjson ram "$GPU_RAM_GB" '{
    client_id: "me", endpoint_name: $name, endpoint_id: $eid, template_hash: $hash, search_params: $search, gpu_ram: $ram
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
echo "search: $SEARCH_PARAMS"
