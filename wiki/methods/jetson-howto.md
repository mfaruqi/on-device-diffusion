---
type: method
summary: Jetson access, CUDA setup, and stage-labelled profiling bundle transfer, build and capture procedures.
status: active
updated: 2026-10-04
---

# Working on the Jetson Orin Nano

## Connect from the Mac

In macOS Terminal, on the same local network:

```bash
ssh mfaruqi@192.168.4.61
```

Login: `mfaruqi`; hostname shown by the remote prompt: `mfaruqi-desktop`.
This address and a successful login were observed on 2026-09-28
([setup evidence](../../raw/jetson-setup-2026-09-28.md#successful-ssh-session-from-the-mac)).
Use the Jetson account password. Commands at the remote prompt execute on the Jetson.
Keep the board powered and connected to the network; the display and USB input
devices are unnecessary for SSH operation.

Treat the IP as the last known address, not a permanent assignment. If it stops
working, check the router's client list or run `hostname -I` locally on the Jetson.
A router DHCP reservation can keep the address stable.

## CUDA shell setup

CUDA 12.6 is installed, and `nvcc` reports V12.6.68 over SSH
([setup evidence](../../raw/jetson-setup-2026-09-28.md)). The resolved shell issue was
a missing CUDA bin directory in `PATH`; variable names are case-sensitive.
The following line is already configured in `~/.bashrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

For a shell where `nvcc` is missing, run that line and then `nvcc --version`.
The full-path diagnostic is `/usr/local/cuda/bin/nvcc --version`.
Compiler availability alone does not establish successful GPU execution.

## Persistent work and next checks

Build preparation and CUDA execution have since been verified; sd.cpp detects Orin.
Disk-backed parameter storage enabled image generation, followed by a repeated baseline and
a validated later-generation profile ([record](../../experiments/jetson-flux-klein-003.md)).
For future setup, run on the Jetson through SSH:

```bash
sudo apt update
sudo apt install -y git cmake build-essential ninja-build tmux
tmux new -s flux
```

Run long builds and tests inside `tmux`. Detach with Ctrl-B, then D.
After an SSH disconnection, reconnect and run `tmux attach -t flux`.
This preserves work across connection loss, not a Jetson reboot.

Record versions and power mode for each experiment. CMake and power mode observations
are preserved with the [first attempt](../../experiments/jetson-flux-klein-001.md).
The setup login banner requested a restart; completion is not recorded
([source](../../raw/jetson-setup-2026-09-28.md#successful-ssh-session-from-the-mac)).

This setup supports Week 1's “A100 and Jetson image baselines” and RQ1:
“Which diffusion optimization choices transfer across devices and workloads?”
([proposal overview](../project/overview.md)). See the [device page](../systems/jetson-orin-nano.md)
and [metric definitions](baseline-metrics.md) before recording benchmark results.

## Stage-labelled harness profile

The [capture script](../../scripts/profile_jetson_stages.py) runs one generation using the
[NVTX harness](../../engines/sdcpp/bench.cpp) and
[profile config](../../configs/jetson-flux-klein-stage-profile.json). This is a planned
instrumentation check of Jetson 003, not a repeated baseline. It verifies hashes before
capture (warming the filesystem cache), records whole-system tegrastats, and checks callback
ordering and NVTX label presence. Original CLI and harness profiles are distinct captures.

### What was sent and why

The [source bundle](../../bundles/README.md) was introduced in commit `e6be8fb`
on `feat/jetson-stage-profile`. Its original four files were:

| File | Purpose |
|---|---|
| [bench.cpp](../../engines/sdcpp/bench.cpp) | Calls sd.cpp through its public API, loads a context once, and emits NVTX ranges from log/progress callbacks. Adds optional explicit sampler/scheduler arguments. |
| [CMakeLists.txt](../../engines/sdcpp/CMakeLists.txt) | Links the harness against existing sd.cpp static libraries; adds an NVTX3 header fallback for Jetson's older CMake. |
| [profile_jetson_stages.py](../../scripts/profile_jetson_stages.py) | Checks the engine commit, model hashes and power mode, runs Nsight plus tegrastats, and checks stage callbacks and labels. |
| [profile config](../../configs/jetson-flux-klein-stage-profile.json) | Specifies one generation with the successful quantized, disk-backed Jetson settings and explicit harness arguments. |

The refactored capture also ships five standard-library helper modules:
`jetson_profile.py`, `jetson_device.py`, `measurement.py`, `measurement_events.py` and `benchlib.py`.
[The bundle builder](../../scripts/build_jetson_bundle.py) also packages the repeated runner,
sd.cpp parsing/image helper and two baseline/smoke configurations;
[CPU tests](../../tests/test_jetson_capture.py) verify saved command/schema compatibility,
monitor cleanup and execution of the extracted CLI. It has since run on Jetson hardware:
repeated baseline, targeted profile and full-protocol profile on 2026-09-30
([record](../../experiments/jetson-flux-klein-003.md)). Rebuild and transfer the entire archive, not only the
entry script ([bundle instructions](../../bundles/README.md)).

The archive contains source/configuration, not model weights or compiled executables.
The existing engine checkout at `~/tools/stable-diffusion.cpp` provides the built CUDA
libraries; models remain at `~/models/flux2-klein`. The new executable is
`~/tools/sd-bench-build/sd-bench-nvtx`. The harness labels are `load`, `generate`,
`text_encode`, `denoise_step_0` through `denoise_step_3`, and `vae_decode`
([source](../../engines/sdcpp/bench.cpp)). Callback boundaries and timing limitations
follow the [measurement definitions](baseline-metrics.md#jetson-stage-labelled-profile-capture).

### Transfer and build

On the Mac, from the local repository checked out at the desired bundle version:

```bash
scp bundles/jetson-stage-profile.tar.gz mfaruqi@192.168.4.61:~/
```

Then on Jetson, inside tmux:

```bash
mkdir -p ~/tools/jetson-stage-profile
tar --touch -xzf ~/jetson-stage-profile.tar.gz -C ~/tools/jetson-stage-profile
cd ~/tools/jetson-stage-profile
```

For future changes, rebuild the archive using the command in the
[bundle README](../../bundles/README.md), transfer it again, extract it, and rebuild the
harness. The extracted files are a snapshot, not a Git checkout; updating Git on the Mac
or Gilbreth does not update the Jetson copy.

Build and capture from `~/tools/jetson-stage-profile`:

```bash
cmake -S engines/sdcpp -B ~/tools/sd-bench-build \
  -DSDCPP_DIR="$HOME/tools/stable-diffusion.cpp" \
  -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build ~/tools/sd-bench-build --target sd-bench-nvtx -j2
sudo -v
python3 scripts/profile_jetson_stages.py --config configs/jetson-flux-klein-stage-profile.json
```

The script prints its run directory; inspect `status.json`, `nsys-stats.txt`, `results.jsonl`
and the trace before using stage attribution. Raw images are RGB files, not PNGs.
Repeated timing uses the separate unprofiled entry point below; the single-profile
script intentionally still rejects multi-generation protocols.

## Repeated unprofiled measurements

The [runner](../../scripts/run_jetson_sdcpp.py) reuses the C++ harness, shared sd.cpp timing
parser/image hashing and measurement summaries. Start reading at `benchmark()`.
The [baseline config](../../configs/jetson-flux-klein-q4-512-disk-baseline.json) preserves
Jetson 003's component revision/hash pins, 512×512 four-step workload and quantized
disk-backed execution settings. Only the protocol changes: one first, three warm-ups,
ten measured generations in one model context, without Nsight. It is a baseline for
this compressed, lower-resolution workload, not the A100 BF16 1024×1024 reference.

CPU replay and fake-engine integration are covered by
[tests](../../tests/test_jetson_baseline.py). The two-generation smoke and 14-generation
baseline passed on Jetson ([record](../../experiments/jetson-flux-klein-003.md#repeated-unprofiled-baseline-2026-09-30)).
The [smoke config](../../configs/jetson-flux-klein-q4-512-disk-smoke.json) runs one first
and one measured generation; it is a functional check, not a performance baseline.

Rebuild/transfer/extract the complete bundle using the steps above. From
`~/tools/jetson-stage-profile`, build the **plain** harness and check dependencies:

```bash
cmake -S engines/sdcpp -B ~/tools/sd-bench-build \
  -DSDCPP_DIR="$HOME/tools/stable-diffusion.cpp" \
  -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build ~/tools/sd-bench-build --target sd-bench -j2
python3 -c 'from PIL import Image; print("Pillow available")'
sudo -v
python3 scripts/run_jetson_sdcpp.py \
  --config configs/jetson-flux-klein-q4-512-disk-smoke.json --kind attempt
```

Pillow is used only for output PNGs after measurement; a missing import fails before
loading. No torch/diffusers installation, model downloads or Nsight capture is needed.
Check `status.json`, callback rows, output images and memory evidence before the full run:

```bash
sudo -v
python3 scripts/run_jetson_sdcpp.py \
  --config configs/jetson-flux-klein-q4-512-disk-baseline.json
```

The runner checks the pinned engine commit, requested power mode, all three model
hashes, callback counts/order, output sizes and the engine's reported execution settings.
It does not change power mode, precision, residency or resolution after failure.
Hash verification reads the model files before capture and warms the filesystem cache;
the load result is not labelled a cold-disk measurement. Display processes, engine
working-tree/submodule status, binary hash and CUDA linkage are recorded for review.

Outputs are under `~/results/runs/<id>__<kind>__<timestamp>/`. Follow progress in
`engine/results.jsonl`; each completed generation is flushed by the harness.
Failures keep the raw log, partial JSON and any raw image outputs for diagnosis.
Complete runs write `runs.csv`, `stages.csv`, `load.json`, `summary.json`, image hashes,
two PNGs, tegrastats diagnostics and derived events. The
[memory definition](baseline-metrics.md#jetson-repeated-baseline-memory) remains separate
from A100 NVML and per-generation allocation peaks.

### Confirmed installation status

The transferred bundle built successfully on Jetson with GCC 11.4.0 and CUDA 12.6.68:
`[100%] Built target sd-bench-nvtx`. An unchecked directory-creation return-value warning
was emitted; it did not prevent linking ([build evidence](../../raw/jetson-stage-profile-build-2026-09-28.md)).
This confirms that earlier harness build only. The corrected harness later passed callback and NVTX label-presence checks
([record](../../experiments/jetson-flux-klein-003.md#corrected-harness-profile)); the imported timeline has been reviewed.
Repeated unprofiled measurements have since passed ([baseline record](../../experiments/jetson-flux-klein-003.md#repeated-unprofiled-baseline-2026-09-30)).

## Full-protocol profile

Use the [full-profile config](../../configs/jetson-flux-klein-q4-512-disk-profile-full.json)
for the standard sd.cpp profiling protocol: one first generation, three warm-ups and ten
measured generations in one context, with Nsight tracing loading and all 14 generations.
This matches the A100 sd.cpp profiling scope. The baseline config's model, workload,
execution settings and generation counts are unchanged; adding profiling is the only change.
This remains a profiled run, not a clean baseline. This mode passes CPU integration
checks and [hardware validation](../../experiments/jetson-flux-klein-003.md#full-protocol-profile-2026-09-30). The older targeted profile remains valid
as a separately labelled diagnostic.

On the Mac, transfer the updated bundle:

```bash
scp /Users/mahadfaruqi/Documents/on-device-diffusion/bundles/jetson-stage-profile.tar.gz mfaruqi@192.168.4.61:~/
```

On Jetson, use the already validated NVTX binary; rebuild with the instructions below if
the capability check fails. `--touch` avoids stale objects when a rebuild is needed.

```bash
tar --touch -xzf ~/jetson-stage-profile.tar.gz -C ~/tools/jetson-stage-profile
cd ~/tools/jetson-stage-profile
~/tools/sd-bench-build/sd-bench-nvtx --capabilities 1
sudo -v
python3 scripts/run_jetson_sdcpp.py \
  --config configs/jetson-flux-klein-q4-512-disk-profile-full.json
```

The capability response must be `{"profile_generation":true}`. The runner uses no capture
range selector in this mode. Completion requires one initial-load NVTX range, 14 generation
ranges, 14 occurrences of every stage, CUDA activity and valid callbacks/images for all rows.
`profile/capture.json` identifies traced indices 0–13 and measured indices 4–13. Medians
and min–max use only those ten measured generations. W&B labels the basis `measured_median`,
scope `profiled`, and `baseline_comparison_eligible=false`; median summary fields retain
those semantics. Copy the entire new run directory back after completion.

For analysis, `--generation 4` selects the first measured generation and `--generation 13`
the last measured generation. The default `-1` still analyzes only the last trace generation;
it does not aggregate all ten profiles. Use separate output names to retain analyses of
multiple generations. No existing run or uploaded record is reclassified.

## Profile a later generation

The optional [targeted profile config](../../configs/jetson-flux-klein-q4-512-disk-profile.json) keeps the baseline's
weights, seed, resolution and disk-backed settings. It runs five generations in one context:
first → three warm-ups → one traced generation. This is a diagnostic capture, not another
ten-sample baseline. The [metric definitions](baseline-metrics.md#jetson-targeted-later-generation-capture)
explain scope and timing. This supports Week 1/RQ1; the selector passed
[target validation](../../experiments/jetson-flux-klein-003.md#targeted-later-generation-profile-2026-09-30).

Transfer the refreshed bundle from the Mac:

```bash
scp /Users/mahadfaruqi/Documents/on-device-diffusion/bundles/jetson-stage-profile.tar.gz mfaruqi@192.168.4.61:~/
```

On Jetson, extract and rebuild the NVTX harness against the already-built pinned engine:

```bash
tar --touch -xzf ~/jetson-stage-profile.tar.gz -C ~/tools/jetson-stage-profile
cd ~/tools/jetson-stage-profile
cmake -S engines/sdcpp -B ~/tools/sd-bench-build \
  -DSDCPP_DIR="$HOME/tools/stable-diffusion.cpp" \
  -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build ~/tools/sd-bench-build --target sd-bench-nvtx -j2
~/tools/sd-bench-build/sd-bench-nvtx --capabilities 1
sudo -v
python3 scripts/run_jetson_sdcpp.py \
  --config configs/jetson-flux-klein-q4-512-disk-profile.json
```

The config selects kind `profile` and `sd-bench-nvtx` automatically. The runner rejects a
stale/plain harness and saves the exact command in `engine/command.json`. Nsight uses
`--capture-range=nvtx --nvtx-capture=profile_capture --capture-range-end=stop`; the capture
marker uses a registered NVTX string as recommended by the
[NVIDIA guide](https://docs.nvidia.com/nsight-systems/UserGuide/).
No new packages or weights are downloaded. Keep other applications and power settings
consistent with the baseline. Do not run another generation job concurrently.

The capability command must print `{"profile_generation":true}` before starting the runner.
The [bundle builder](../../scripts/build_jetson_bundle.py) normalizes archive timestamps to zero;
extract with GNU tar's `--touch` so changed sources receive current timestamps and rebuild.
Without it, Make can reuse an older object and still report a successful target build.
If the capability command instead reports `missing required --out-dir`, confirm
`grep -n capabilities engines/sdcpp/bench.cpp` finds the new handler, then run
`touch engines/sdcpp/bench.cpp` and rebuild `sd-bench-nvtx`. This recompiles the small
harness; it does not rebuild the engine or change any weights. Keep failed run directories.

Progress is in `engine/results.jsonl`. The output directory contains the normal summary/CSVs,
`profile/trace.nsys-rep`, exported `profile/trace.sqlite`, `profile/nsys-stats.txt` and a validated
`profile/capture.json` receipt. Failure keeps partial evidence and marks `status.json` failed.
Copy back the entire new `__profile__` directory, including trace files. On a machine with
Nsight available, the existing analysis CLI can read that directory:

```bash
python3 scripts/analyze_profile.py --run-dir results/runs/<new-profile-directory> \
  --generation 0 --out-name denoise_kernels
python3 scripts/analyze_profile.py --run-dir results/runs/<new-profile-directory> \
  --generation 0 --stages 'text_encode|vae_decode' --out-name other_stages
```

Here `--generation 0` selects the only generation in the trace, which is generation 4
of the full harness process. These commands run from the repository; analysis scripts
are not needed in the transfer bundle.
