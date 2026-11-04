#!/usr/bin/env bash
# Which Nsight Compute counters can measure memory traffic on the Jetson Orin Nano (GPU ga10b), and do they
# return sensible values? Nsight Compute on this board needs root even to list metrics.
#   1. saves the full metric list for ga10b;
#   2. profiles a small workload with known traffic (BF16 2048^3 GEMM; a 256 MiB device-to-device copy that
#      must read and write 256 MiB each) using every candidate byte/ops metric that exists on this chip.
# Output: ~/results/ncu-probe-20261029/{metrics.txt,candidates.txt,probe.csv,probe.log}. About one minute.
# Usage (on the Jetson, nothing else running):  sudo bash scripts/jetson_ncu_probe.sh
set -uo pipefail
[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] || { echo "run with sudo (Nsight Compute needs root on this board)"; exit 1; }
export HOME=/home/$SUDO_USER
NCU=/usr/local/cuda/bin/ncu
PY=$HOME/venvs/torch-jp6-ailab/bin/python
OUT=$HOME/results/ncu-probe-20261029
rm -rf "$OUT"; mkdir -p "$OUT"

$NCU --version > "$OUT/ncu-version.txt" 2>&1
$NCU --query-metrics --chip ga10b > "$OUT/metrics.txt" 2>&1
# Candidates: DRAM bytes (A100 names), L2 <-> system-memory sectors (an integrated GPU reaches LPDDR5 through
# the SoC fabric), the time metric and the Tensor Core / FP32 op counts used on the A100.
CANDIDATES="gpu__time_duration dram__bytes_read dram__bytes_write dram__bytes
lts__t_sectors_srcunit_tex_aperture_sysmem lts__t_sectors_aperture_sysmem lts__t_sectors_srcunit_tex_aperture_device
lts__t_sectors_aperture_device lts__t_bytes lts__t_sectors_op_read lts__t_sectors_op_write
fbpa__dram_read_bytes fbpa__dram_write_bytes
sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off
smsp__sass_thread_inst_executed_op_ffma_pred_on"
METRICS=""
for m in $CANDIDATES; do
  if grep -qE "^${m}[[:space:]]" "$OUT/metrics.txt"; then echo "$m present" >> "$OUT/candidates.txt"; METRICS="$METRICS,$m.sum"
  else echo "$m absent" >> "$OUT/candidates.txt"; fi
done
METRICS=${METRICS#,}
echo "metrics: $METRICS" | tee "$OUT/probe.log"

cat > "$OUT/work.py" <<'EOF'
import torch
a = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16); b = torch.randn_like(a)
src = torch.empty(256 << 20, dtype=torch.uint8, device='cuda'); dst = torch.empty_like(src)
torch.cuda.synchronize()
c = a @ b              # 2*2048^3 = 17.2 GFLOP, at least 3 x 8 MiB of traffic
dst.copy_(src)         # 256 MiB read + 256 MiB write
torch.cuda.synchronize()
EOF
$NCU --target-processes all --replay-mode kernel --cache-control none --clock-control none \
     --metrics "$METRICS" --csv --page raw --print-units base "$PY" "$OUT/work.py" > "$OUT/probe.csv" 2>> "$OUT/probe.log"
echo "exit $?" >> "$OUT/probe.log"
chown -R "$SUDO_USER:$SUDO_USER" "$OUT"
echo "done: $OUT"; cat "$OUT/candidates.txt"
