# Jetson stage-profile bundle build evidence — 2026-09-28

Source: terminal output supplied from `mfaruqi@mfaruqi-desktop`.
Bundle version: repository commit `e6be8fb` on `feat/jetson-stage-profile`.
The transfer was reported completed; no checksum of the received archive was supplied.

Commands executed:

```bash
mkdir -p ~/tools/jetson-stage-profile
tar -xzf ~/jetson-stage-profile.tar.gz -C ~/tools/jetson-stage-profile
cd ~/tools/jetson-stage-profile
cmake -S engines/sdcpp -B ~/tools/sd-bench-build \
  -DSDCPP_DIR="$HOME/tools/stable-diffusion.cpp" \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ~/tools/sd-bench-build --target sd-bench-nvtx -j2
```

Selected terminal lines, transcribed:

```text
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Found CUDAToolkit: /usr/local/cuda/include (found version "12.6.68")
-- Found OpenMP: TRUE (found version "4.5")
-- Configuring done
-- Generating done
-- Build files have been written to: /home/mfaruqi/tools/sd-bench-build
[ 50%] Building CXX object CMakeFiles/sd-bench-nvtx.dir/bench.cpp.o
warning: ignoring return value of ‘int system(const char*)’ declared with attribute ‘warn_unused_result’ [-Wunused-result]
[100%] Linking CXX executable sd-bench-nvtx
[100%] Built target sd-bench-nvtx
```

The warning points to `bench.cpp` calling `system` to create the raw-output directory
without checking its return value. Compilation and linking completed. No execution
result for this stage-labelled harness was supplied with the build output.
