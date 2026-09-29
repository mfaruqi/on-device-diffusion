---
type: method
summary: Jetson access, CUDA setup, and stage-labelled profiling bundle transfer, build and capture procedures.
status: active
updated: 2026-09-28
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
Disk-backed parameter storage subsequently enabled one saved image; visual smoke check passed,
benchmarking remains pending ([record](../../experiments/jetson-flux-klein-003.md)).
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
on `feat/jetson-stage-profile`. It contains four files:

| File | Purpose |
|---|---|
| [bench.cpp](../../engines/sdcpp/bench.cpp) | Calls sd.cpp through its public API, loads a context once, and emits NVTX ranges from log/progress callbacks. Adds optional explicit sampler/scheduler arguments. |
| [CMakeLists.txt](../../engines/sdcpp/CMakeLists.txt) | Links the harness against existing sd.cpp static libraries; adds an NVTX3 header fallback for Jetson's older CMake. |
| [profile_jetson_stages.py](../../scripts/profile_jetson_stages.py) | Checks the engine commit, model hashes and power mode, runs Nsight plus tegrastats, and checks stage callbacks and labels. |
| [profile config](../../configs/jetson-flux-klein-stage-profile.json) | Specifies one generation with the successful quantized, disk-backed Jetson settings and explicit harness arguments. |

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
tar -xzf ~/jetson-stage-profile.tar.gz -C ~/tools/jetson-stage-profile
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
Repeated timing still requires the Jetson memory adapter and baseline runner integration.

### Confirmed installation status

The transferred bundle built successfully on Jetson with GCC 11.4.0 and CUDA 12.6.68:
`[100%] Built target sd-bench-nvtx`. An unchecked directory-creation return-value warning
was emitted; it did not prevent linking ([build evidence](../../raw/jetson-stage-profile-build-2026-09-28.md)).
This confirms the harness build only. The corrected harness later passed callback and NVTX label-presence checks
([record](../../experiments/jetson-flux-klein-003.md#corrected-harness-profile)); the imported timeline has been reviewed. Repeated unprofiled measurements remain pending.
