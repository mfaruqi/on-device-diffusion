# Successful Jetson CLI feasibility attempt

Full supplied command, generation.log, tegrastats.log, before/after snapshots, power mode,
original status and image have been imported from the Jetson. Earlier terminal excerpts
remain as provenance. Config/status are curated records; status-original.json is unchanged.

Reproduce derived diagnostics with:
`python3 scripts/analyze_jetson_feasibility.py --run-dir results/runs/jetson-flux-klein-003-20260928-185009`

Outputs: feasibility-analysis.json (input hashes and diagnostics), tegrastats-samples.csv.
Engine settings and residency observations: engine-audit.json. Image check: image-inspection.json.
Logs and images are local, git-ignored artifacts; derived JSON/CSV preserve cited evidence.
This is one feasibility attempt, not the repeated benchmark protocol. No stage-aligned
memory peak, process wall time, physical disk I/O or formal quality score is established.
