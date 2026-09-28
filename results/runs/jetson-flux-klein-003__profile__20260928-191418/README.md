# Initial Jetson CLI profile: CUDA events confirmed, originals pending

The supplied Nsight stats contain CUDA kernel, memory-operation and API rows;
selected values are preserved in [stats excerpts](nsys-stats-excerpts.json).
This confirms event presence, not complete trace coverage or stage attribution.
Import all original files into an `originals/` subdirectory to preserve curated
status/config, then inspect the timeline and profiler exit status.
Profiler timings are not baseline latency. This CLI has no added harness NVTX stage markers.
