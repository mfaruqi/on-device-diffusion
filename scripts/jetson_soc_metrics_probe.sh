#!/usr/bin/env bash
# Probe which Nsight Systems hardware metrics this Jetson can record (both need root): SoC metrics (set t234: memory
# controller / DRAM bandwidth) and GPU metrics (set ga10b). A 6-second torch workload (BF16 matmuls plus a large copy)
# runs under each option; the exported metric names and a few values are written to ~/results/soc-probe-20261028/.
# Usage (on the Jetson):  sudo bash scripts/jetson_soc_metrics_probe.sh
set -uo pipefail
[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] || { echo "run with sudo"; exit 1; }
export HOME=/home/$SUDO_USER PATH=/usr/local/cuda/bin:$PATH
OUT=$HOME/results/soc-probe-20261028; rm -rf "$OUT"; mkdir -p "$OUT"
PY=$HOME/venvs/torch-jp6-ailab/bin/python
cat > "$OUT/work.py" <<'EOF'
import time, torch
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); b = torch.randn_like(a)
big = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
t = time.time()
while time.time() - t < 3:
    c = a @ b
torch.cuda.synchronize(); t = time.time()
while time.time() - t < 3:
    big2 = big.clone(); del big2
torch.cuda.synchronize()
EOF
run() {  # run <label> <nsys options...>
  local label=$1; shift
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true --export=sqlite "$@" \
    -o "$OUT/$label" "$PY" "$OUT/work.py" > "$OUT/$label.log" 2>&1
  echo "$label rc=$?" | tee -a "$OUT/summary.txt"
  [[ -f "$OUT/$label.sqlite" ]] && /usr/bin/python3 - "$OUT/$label.sqlite" > "$OUT/$label.metrics.txt" 2>&1 <<'EOF'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
tables = [r[0] for r in db.execute("select name from sqlite_master where type='table'")]
print('tables:', [t for t in tables if 'METRIC' in t.upper() or 'SOC' in t.upper()])
for t in tables:
    if t.startswith('TARGET_INFO') and 'METRIC' in t.upper():
        cols = [c[1] for c in db.execute(f'pragma table_info({t})')]
        print(t, cols)
        for row in db.execute(f'select * from {t} limit 80'):
            print('  ', row)
for t in ('GPU_METRICS', 'SOC_METRICS'):
    if t in tables:
        print(t, 'rows', db.execute(f'select count(*) from {t}').fetchone()[0])
        print(t, 'per metric (id, n, avg, max):')
        for row in db.execute(f'select metricId, count(*), avg(value), max(value) from {t} group by metricId'):
            print('  ', row)
EOF
}
run soc --soc-metrics=true --soc-metrics-frequency=1000
run gpu --gpu-metrics-devices=all --gpu-metrics-frequency=1000
tegrastats --interval 500 > "$OUT/tegrastats-root.txt" & TP=$!; sleep 2; kill $TP
chown -R "$SUDO_USER:$SUDO_USER" "$OUT"
echo "done: $OUT"
