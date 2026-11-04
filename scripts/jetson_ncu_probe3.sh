#!/usr/bin/env bash
# Third Nsight Compute probe on the Jetson Orin Nano (ga10b). The engine counter windows (application replay,
# duration + L2 sysmem bytes + Tensor Core ops) stopped launching after 20-24 kernels for both sd.cpp and
# PyTorch, while a duration-only pass over 11,148 launches completed. This probe separates the two suspects
# on a 120-kernel toy workload, each case under a 5 minute timeout:
#   A/B. kernel replay, full set vs bytes only: the log prints the number of hardware passes per kernel;
#   C/D. application replay, full set vs bytes only: does the stall reproduce, and only with several passes?
# Output: ~/results/ncu-probe3-20261031/{A,B,C,D}.{csv,log}, summary.txt. About 10 minutes at most.
# Usage (on the Jetson, nothing else running):  sudo bash scripts/jetson_ncu_probe3.sh
set -uo pipefail
[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] || { echo "run with sudo (Nsight Compute needs root on this board)"; exit 1; }
export HOME=/home/$SUDO_USER
NCU=/usr/local/cuda/bin/ncu
PY=$HOME/venvs/torch-jp6-ailab/bin/python
OUT=$HOME/results/ncu-probe3-20261031
rm -rf "$OUT"; mkdir -p "$OUT"

BYTES=gpu__time_duration.sum,lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum,lts__t_sectors_aperture_sysmem_op_write.sum
FULL=$BYTES,sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum,sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum,sm__ops_path_tensor_src_tf32_dst_fp32_sparsity_off.sum

cat > "$OUT/work.py" <<'EOF'
import torch
x = torch.ones(16 << 20, device='cuda', dtype=torch.bfloat16)   # 32 MiB
a = torch.ones(1024, 1024, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
for _ in range(40):            # 3 kernels per iteration, 120 profiled launches
    y = x + 1
    c = a @ a
    x = y * 0.5
torch.cuda.synchronize()
print('work done')
EOF

probe() {  # name, replay mode, metrics
  timeout -k 10 300 $NCU --target-processes all --replay-mode "$2" --cache-control none --clock-control none \
      --metrics "$3" --csv --page raw --print-units base "$PY" "$OUT/work.py" > "$OUT/$1.csv" 2> "$OUT/$1.log"
  local rc=$?
  local n; n=$(grep -c '^"[0-9]' "$OUT/$1.csv")
  echo "$1 replay=$2 rc=$rc kernels=$n passes: $(grep -o '[0-9]* passes\?' "$OUT/$1.log" | sort | uniq -c | tr '\n' ' ')" | tee -a "$OUT/summary.txt"
  pkill -f "$OUT/work.py" 2>/dev/null; sleep 5
}
probe A kernel "$FULL"
probe B kernel "$BYTES"
probe C application "$FULL"
probe D application "$BYTES"
chown -R "$SUDO_USER:$SUDO_USER" "$OUT"
echo "done: $OUT"
