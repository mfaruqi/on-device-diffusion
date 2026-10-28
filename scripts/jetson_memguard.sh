#!/usr/bin/env bash
# Memory guard for runs on the Jetson Orin Nano (8 GB shared CPU/GPU memory).
# Usage:  bash jetson_memguard.sh <out-dir> <command...>
# Runs <command> in its own process group and polls /proc/meminfo every POLL_S seconds. If
# MemAvailable drops below MIN_AVAIL_MB, or swap use grows by more than MAX_SWAP_GROWTH_MB since the
# start, the whole group is killed with SIGKILL and the reason is written to <out-dir>/memguard.json.
# Why: on 2026-10-11 two runs pushed RAM to ~7.4/7.6 GB with 1.5-2.3 GB of swap; one was OOM-killed,
# the other hung the board until it rebooted. GPU allocations come out of the same RAM, so
# MemAvailable covers them. A guard kill is a failed run, recorded as such; it never changes the
# configuration being tested.
set -uo pipefail
OUT=$1; shift
MIN_AVAIL_MB=${MIN_AVAIL_MB:-400}
MAX_SWAP_GROWTH_MB=${MAX_SWAP_GROWTH_MB:-768}
POLL_S=${POLL_S:-0.1}
mkdir -p "$OUT"

meminfo_mb() { awk -v k="$1:" '$1==k {print int($2/1024)}' /proc/meminfo; }
swap_used_mb() { echo $(( $(meminfo_mb SwapTotal) - $(meminfo_mb SwapFree) )); }

SWAP0=$(swap_used_mb)
AVAIL0=$(meminfo_mb MemAvailable)
setsid "$@" &
PID=$!
PGID=$PID
MIN_SEEN=$AVAIL0; MAX_SWAP=$SWAP0; REASON=""
while kill -0 "$PID" 2>/dev/null; do
  A=$(meminfo_mb MemAvailable); S=$(swap_used_mb)
  (( A < MIN_SEEN )) && MIN_SEEN=$A
  (( S > MAX_SWAP )) && MAX_SWAP=$S
  if (( A < MIN_AVAIL_MB )); then REASON="MemAvailable ${A} MB < ${MIN_AVAIL_MB} MB"; fi
  if (( S - SWAP0 > MAX_SWAP_GROWTH_MB )); then REASON="swap grew $((S - SWAP0)) MB > ${MAX_SWAP_GROWTH_MB} MB"; fi
  if [[ -n "$REASON" ]]; then
    kill -KILL -- "-$PGID" 2>/dev/null
    break
  fi
  sleep "$POLL_S"
done
wait "$PID"; RC=$?
cat > "$OUT/memguard.json" <<EOF
{"killed_by_guard": $([[ -n "$REASON" ]] && echo true || echo false),
 "reason": "${REASON}",
 "exit_code": $RC,
 "min_avail_mb_threshold": $MIN_AVAIL_MB, "max_swap_growth_mb_threshold": $MAX_SWAP_GROWTH_MB, "poll_s": $POLL_S,
 "mem_available_start_mb": $AVAIL0, "mem_available_min_mb": $MIN_SEEN,
 "swap_used_start_mb": $SWAP0, "swap_used_max_mb": $MAX_SWAP}
EOF
exit $RC
