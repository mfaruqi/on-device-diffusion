#!/usr/bin/env bash
# Second Nsight Compute probe on the Jetson Orin Nano (ga10b). The first probe (jetson_ncu_probe.sh) showed that
# ga10b has no dram__* / fbpa__* metrics; memory traffic is only visible at L2. This probe checks whether the L2
# miss / fill counters reproduce known DRAM traffic, so they can stand in for dram__bytes on the roofline:
#   - elementwise add of two 256 MiB BF16 tensors: 512 MiB read + 256 MiB written (L2 is 2 MiB, no reuse);
#   - one 256 MiB write (fill_);
#   - BF16 2048^3 GEMM: at least 16 MiB read + 8 MiB written.
# Output: ~/results/ncu-probe2-20261030/{probe.csv,probe.log}. About one minute.
# Usage (on the Jetson, nothing else running):  sudo bash scripts/jetson_ncu_probe2.sh
set -uo pipefail
[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] || { echo "run with sudo (Nsight Compute needs root on this board)"; exit 1; }
export HOME=/home/$SUDO_USER
NCU=/usr/local/cuda/bin/ncu
PY=$HOME/venvs/torch-jp6-ailab/bin/python
OUT=$HOME/results/ncu-probe2-20261030
rm -rf "$OUT"; mkdir -p "$OUT"
METRICS="gpu__time_duration.sum,lts__t_bytes.sum,lts__d_sectors_fill_sysmem.sum,lts__d_sectors_fill_device.sum,\
lts__t_sectors_aperture_sysmem_lookup_miss.sum,lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum,\
lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum,lts__t_sectors_aperture_sysmem_op_read_lookup_hit.sum,\
lts__t_sectors_aperture_sysmem_op_write_lookup_hit.sum,lts__t_sectors_srcunit_tex_aperture_sysmem_lookup_miss.sum,\
sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum"
cat > "$OUT/work.py" <<'PY'
import torch
n = 128 << 20                                  # 128 Mi BF16 elements = 256 MiB per tensor
x = torch.empty(n, device='cuda', dtype=torch.bfloat16)
y = torch.empty_like(x); z = torch.empty_like(x)
a = torch.empty(2048, 2048, device='cuda', dtype=torch.bfloat16); b = torch.empty_like(a)
torch.cuda.synchronize()
x.fill_(1.0)                                   # 256 MiB write
y.fill_(2.0)                                   # 256 MiB write
torch.add(x, y, out=z)                         # 512 MiB read + 256 MiB write
c = a @ b                                      # >= 16 MiB read + 8 MiB write
torch.cuda.synchronize()
PY
echo "metrics: $METRICS" > "$OUT/probe.log"
$NCU --target-processes all --replay-mode kernel --cache-control none --clock-control none \
     --metrics "$METRICS" --csv --page raw --print-units base "$PY" "$OUT/work.py" > "$OUT/probe.csv" 2>> "$OUT/probe.log"
echo "exit $?" >> "$OUT/probe.log"
chown -R "$SUDO_USER:$SUDO_USER" "$OUT"
echo "done: $OUT"
