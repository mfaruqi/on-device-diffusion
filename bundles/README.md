# Jetson stage-profile transfer bundle

`jetson-stage-profile.tar.gz` contains the harness sources, its CMake build file,
the single-profile capture script, and the Jetson profile configuration. No models
or run outputs are included. From the repository root, rebuild it after changing
any of these files:

```bash
tar -czf bundles/jetson-stage-profile.tar.gz \
  engines/sdcpp/bench.cpp engines/sdcpp/CMakeLists.txt \
  scripts/profile_jetson_stages.py configs/jetson-flux-klein-stage-profile.json
```

Transfer from a Mac on the Jetson's network:

```bash
scp bundles/jetson-stage-profile.tar.gz mfaruqi@192.168.4.61:~/
```

Extract into `~/tools/jetson-stage-profile` on Jetson, then follow the
[build and capture instructions](../wiki/methods/jetson-howto.md#stage-labelled-harness-profile).
