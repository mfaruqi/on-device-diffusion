---
type: method
summary: Jetson Orin Nano access from the Mac, CUDA shell setup, and the next verification steps.
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
