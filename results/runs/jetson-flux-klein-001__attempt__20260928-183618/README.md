# Failed Jetson feasibility attempt

This directory preserves a partial evidence import, not runner-generated benchmark output.
`terminal-excerpts.json` contains verbatim selected error and post-failure sampling lines,
plus clearly labelled setup observations. `config.json`, `status.json`, and `summary.json`
were reconstructed from that evidence and the supplied command.

Full originals remain on the Jetson at:
`/home/mfaruqi/results/jetson-flux-klein-q4-512-20260928-183618/`.
Import `generation.log`, `tegrastats.log`, and `exit-code.json` without overwriting the excerpts.
Do not infer peak memory or whole-generation latency from this partial import.
