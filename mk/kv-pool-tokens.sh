#!/usr/bin/env sh
# The tokens a named --vram-kv-pool-bytes holds (ADR 0045), as the server's plan
# reads it, for `make config`: a token count rounded up to whole 64-token
# pages, a byte count cut to the whole pages it buys at the model's paged
# bytes per token. The plan event (`kv_pool_tokens`) is the authority; the
# per-token costs below are pinned to the Rust ones by a test in
# crates/server/src/runtime.rs.
#
#   kv-pool-tokens.sh <value> <27b|flash-next> <hq-e8-2b|bf16|nvfp4>
v=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
format=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
case "$format" in hq|hq_e8_2b) format=hq-e8-2b ;; esac
case "$2:$format" in
  flash-next:bf16) per_token=25344 ;;
  flash-next:hq-e8-2b) per_token=4224 ;;
  27b:bf16) per_token=65536 ;;
  27b:hq-e8-2b) per_token=9216 ;;
  27b:nvfp4) per_token=18432 ;;
  *) printf '%s\n' "unsupported KV format/geometry: $2:$format" >&2; exit 2 ;;
esac
case "$v" in
  *ktok) n=${v%ktok}; scale=1024 ;;
  *mtok) n=${v%mtok}; scale=1048576 ;;
  *tok) n=${v%tok}; scale=1 ;;
  *)
    n=${v%%[!0-9]*}
    case "${v#"$n"}" in
      ""|b) scale=1 ;;
      k|kb|kib) scale=1024 ;;
      m|mb|mib) scale=1048576 ;;
      g|gb|gib) scale=1073741824 ;;
      *) echo "?"; exit 0 ;;
    esac
    [ -n "$n" ] || { echo "?"; exit 0; }
    echo $(( n * scale / (per_token * 64) * 64 ))
    exit 0
    ;;
esac
case "$n" in ""|*[!0-9]*) echo "?"; exit 0 ;; esac
echo $(( (n * scale + 63) / 64 * 64 ))
