# Jetson setup evidence — 2026-09-28

Source: terminal screenshots supplied by Mahad Faruqi on 2026-09-28, plus his
confirmation that `nvcc` worked after correcting `Path` to `PATH`.
This is a selective text transcription, not a benchmark or an executable log.

## Local terminal observations

- `/etc/nv_tegra_release`: R36, REVISION 4.4; KERNEL_VARIANT: oot.
- `uname -m`: `aarch64`.
- `free -h`: RAM 7.4 GiB total, 1.8 GiB used, 5.4 GiB available;
  swap 3.7 GiB total, 0 B used. These are one-time setup readings.
- `df -h /`: `/dev/mmcblk0p1`, 116G size, 21G used, 91G available, 19% used.
- `/usr/local/cuda` links through `/etc/alternatives/cuda`;
  `/usr/local/cuda-12.6` exists.
- `/usr/local/cuda/bin/nvcc --version`: CUDA release 12.6, V12.6.68.
- Package listing marks `cuda-nvcc-12-6`, `cuda-toolkit-12`, and
  `cuda-toolkit-12-6` as installed.
- Bare `nvcc` initially failed. The first shell edit used `Path`; correcting
  it to `export PATH=/usr/local/cuda/bin:$PATH` in `~/.bashrc` resolved it.

## Successful SSH session from the Mac

```text
ssh mfaruqi@192.168.4.61
Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.15.148-tegra aarch64)
mfaruqi@mfaruqi-desktop:~$ nvcc --version
Cuda compilation tools, release 12.6, V12.6.68
mfaruqi@mfaruqi-desktop:~$ uname -r
5.15.148-tegra
```

The login banner also displayed `*** System restart required ***`.
The address was observed on the local network; a DHCP reservation was not established
by this evidence. No GPU execution test, engine build, model download, power-mode
query result, or diffusion benchmark is documented by these screenshots.
