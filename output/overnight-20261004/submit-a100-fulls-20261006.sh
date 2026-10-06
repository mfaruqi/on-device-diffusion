#!/bin/bash
set -euo pipefail
options=/home/mfaruqi/on-device-diffusion-campaigns/20261005-options
distilled=/home/mfaruqi/on-device-diffusion-campaigns/20261005-distilled
cd "$options"
mkdir output/full-protocol-launch-20261006
squeue -u mfaruqi -o '%.18i %.30j %.10T %.35R'
ref=$(sbatch --parsable --job-name=base-cache-full-control --export=ALL,CONFIG=configs/a100-sdcpp-flux-klein-base-cache-control.resolved.json scripts/gilbreth-sdcpp.slurm)
printf 'base-control\t%s\n' "$ref" | tee -a output/full-protocol-launch-20261006/jobs.tsv
squeue -u mfaruqi -h -o '%i %j %T'
cache=$(sbatch --parsable --job-name=base-easycache-full --dependency=afterok:"$ref" --export=ALL,CONFIG=configs/a100-sdcpp-flux-klein-base-easycache.resolved.json scripts/gilbreth-sdcpp.slurm)
printf 'base-easycache\t%s\n' "$cache" | tee -a output/full-protocol-launch-20261006/jobs.tsv
cd "$distilled"
mkdir output/full-protocol-launch-20261006
squeue -u mfaruqi -h -o '%i %j %T'
control=$(sbatch --parsable --job-name=distilled-control-full --dependency=afterany:"$cache" --export=ALL,CONFIG=configs/a100-sdcpp-flux-klein-bf16-cache-control.resolved.json scripts/gilbreth-sdcpp.slurm)
printf 'cache-control\t%s\n' "$control" | tee -a output/full-protocol-launch-20261006/jobs.tsv
previous=$control
for option in easycache conditioning-cache no-prefetch mmap; do
 squeue -u mfaruqi -h -o '%i %j %T'
 job=$(sbatch --parsable --job-name="distilled-$option-full" --dependency="afterok:$control,afterany:$previous" --export="ALL,CONFIG=configs/a100-sdcpp-flux-klein-bf16-$option.resolved.json" scripts/gilbreth-sdcpp.slurm)
 printf '%s\t%s\n' "$option" "$job" | tee -a output/full-protocol-launch-20261006/jobs.tsv
 previous=$job
done
