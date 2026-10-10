# Jetson measurement transfer bundle

`jetson-stage-profile.tar.gz` contains the harness sources, its CMake build file,
the single-profile capture and repeated baseline scripts, their helper modules, and the Jetson configurations. No models
or run outputs are included. From the repository root, rebuild it after changing
any of these files:

```bash
python3 scripts/build_jetson_bundle.py
python3 -m unittest discover -s tests -v
```

[build_jetson_bundle.py](../scripts/build_jetson_bundle.py) lists every bundled file
explicitly. The capture imports `jetson_profile`, `jetson_device`, `measurement`,
`measurement_events` and `benchlib`; all five ship alongside it in `scripts/`.
The manifest lists the included files. The older single-profile entry point uses only Python's standard
library; the repeated/targeted runner also needs Pillow to save two PNG images after timing.
The tests check that the archive matches the current sources,
rebuilds byte-for-byte, and that its extracted CLI runs outside the repository.

The repeated runner passed its Jetson smoke and full baseline
([record](../experiments/jetson-flux-klein-003.md#repeated-unprofiled-baseline-2026-09-30)).
The refactored single-profile entry point still needs target validation.

Transfer from a Mac on the Jetson's network:

```bash
scp bundles/jetson-stage-profile.tar.gz mfaruqi@192.168.4.61:~/
```

Extract with `tar --touch -xzf ~/jetson-stage-profile.tar.gz -C ~/tools/jetson-stage-profile`
on Jetson, then follow the
[build and capture instructions](../wiki/methods/jetson-howto.md#stage-labelled-harness-profile).
`--touch` matters: archive timestamps are normalized to zero, so a normal extraction
can leave changed source older than cached build objects and incorrectly skip recompilation.
For the unprofiled smoke check and baseline, use the
[repeated measurement instructions](../wiki/methods/jetson-howto.md#repeated-unprofiled-measurements).
For the standard 14-generation profile matching A100 sd.cpp, use the
[full-protocol instructions](../wiki/methods/jetson-howto.md#full-protocol-profile).
This mode passes CPU integration tests and [Jetson hardware validation](../experiments/jetson-flux-klein-003.md#full-protocol-profile-2026-09-30).
For the optional diagnostic trace after first + three warm-up generations, use the
[targeted profile instructions](../wiki/methods/jetson-howto.md#profile-a-later-generation).
This mode traces only generation index 4 in one persistent model context. CPU checks pass; the new capture selector passed its
[Jetson validation](../experiments/jetson-flux-klein-003.md#targeted-later-generation-profile-2026-09-30).
