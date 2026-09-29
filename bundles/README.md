# Jetson stage-profile transfer bundle

`jetson-stage-profile.tar.gz` contains the harness sources, its CMake build file,
the single-profile capture script and its helper modules, and the Jetson profile configuration. No models
or run outputs are included. From the repository root, rebuild it after changing
any of these files:

```bash
python3 scripts/build_jetson_bundle.py
python3 -m unittest discover -s tests -v
```

[build_jetson_bundle.py](../scripts/build_jetson_bundle.py) lists every bundled file
explicitly. The capture imports `jetson_profile`, `jetson_device`, `measurement`,
`measurement_events` and `benchlib`; all five ship alongside it in `scripts/`.
The archive contains nine files. These imports use only Python's
standard library. The tests check that the archive matches the current sources,
rebuilds byte-for-byte, and that its extracted CLI runs outside the repository.

The refactored bundle has CPU validation only; capture on Jetson hardware is pending.

Transfer from a Mac on the Jetson's network:

```bash
scp bundles/jetson-stage-profile.tar.gz mfaruqi@192.168.4.61:~/
```

Extract into `~/tools/jetson-stage-profile` on Jetson, then follow the
[build and capture instructions](../wiki/methods/jetson-howto.md#stage-labelled-harness-profile).
