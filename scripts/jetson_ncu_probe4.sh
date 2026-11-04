#!/usr/bin/env bash
# Fourth Nsight Compute probe on the Jetson Orin Nano (ga10b). Kernel replay with the full counter set needs 3
# hardware passes; between passes Nsight saves and restores device memory, which with a ~4 GB model loaded ran the
# 8 GB board down to 290 MB (sd.cpp capture, 2026-11-01). A metric group that fits in ONE pass needs no restore.
# This probe finds such groups. The toy holds 2 GiB of device memory (small enough that a full backup still fits) (a stand-in for model weights) and launches
# 122 small kernels with known traffic; for each group it reports the hardware passes per kernel and the lowest
# MemAvailable seen. A single-pass group should leave MemAvailable about where the toy alone puts it.
# Output: ~/results/ncu-probe4-20261101/{<case>.log,<case>.csv,summary.txt}. About 5 minutes.
# Usage (on the Jetson, nothing else running):  sudo bash scripts/jetson_ncu_probe4.sh
set -uo pipefail
[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] || { echo "run with sudo (Nsight Compute needs root on this board)"; exit 1; }
export HOME=/home/$SUDO_USER
NCU=/usr/local/cuda/bin/ncu
PY=$HOME/venvs/torch-jp6-ailab/bin/python
OUT=$HOME/results/ncu-probe4-20261101
rm -rf "$OUT"; mkdir -p "$OUT"
DUR=gpu__time_duration.sum
READ=lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum
WRITE=lts__t_sectors_aperture_sysmem_op_write.sum
BF16=sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum

cat > "$OUT/work.py" <<'EOF'
import torch
weights = torch.empty(2 << 30, device='cuda', dtype=torch.uint8)   # 2 GiB held, never touched by the kernels
x = torch.ones(16 << 20, device='cuda', dtype=torch.bfloat16)       # 32 MiB
a = torch.ones(1024, 1024, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
for _ in range(40):
    y = x + 1
    c = a @ a
    x = y * 0.5
torch.cuda.synchronize()
print('work done')
EOF

avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
probe() {  # name, metrics
  local min; min=$(avail)
  timeout -k 10 300 $NCU --target-processes all --replay-mode kernel --cache-control none --clock-control none \
      --metrics "$2" --export "$OUT/$1" "$PY" "$OUT/work.py" > "$OUT/$1.log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do a=$(avail); (( a < min )) && min=$a; sleep 0.2; done
  wait $pid; local rc=$?
  $NCU --import "$OUT/$1.ncu-rep" --csv --page raw --print-units base > "$OUT/$1.csv" 2>/dev/null
  echo "$1 rc=$rc kernels=$(grep -c '^"[0-9]' "$OUT/$1.csv") passes:$(grep -o '[0-9]* pass\(es\)\?$' "$OUT/$1.log" | sort | uniq -c | tr -s ' \n' ' ') min_avail_mb=$min metrics=$2" | tee -a "$OUT/summary.txt"
  sleep 3
}
echo "baseline MemAvailable before the toy: $(avail) MB" | tee "$OUT/summary.txt"
probe duration "$DUR"
probe read "$DUR,$READ"
probe write "$DUR,$WRITE"
probe bytes "$DUR,$READ,$WRITE"
probe bf16 "$DUR,$BF16"
chown -R "$SUDO_USER:$SUDO_USER" "$OUT"
echo "done: $OUT"
