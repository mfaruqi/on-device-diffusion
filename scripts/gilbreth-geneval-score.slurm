#!/bin/bash
# Score GenEval image folders with the official evaluator (djghosh13/geneval af4902f; Mask2Former Swin-S + CLIP ViT-L-14).
#   IMAGE_DIRS="<run>/images <run>/images" sbatch scripts/gilbreth-geneval-score.slurm
# Writes <run>/geneval-results.jsonl and <run>/geneval-summary.txt next to each image folder.
#SBATCH --job-name=geneval-score
#SBATCH --account=you139
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --constraint=G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/geneval-score-%j.out
set -euo pipefail
G=/scratch/gilbreth/$USER/geneval
export HF_HOME=/scratch/gilbreth/$USER/huggingface HF_HUB_OFFLINE=1
cd $G/geneval
for images in ${IMAGE_DIRS:?set IMAGE_DIRS}; do
  run=$(dirname "$images")
  [ -s "$run/geneval-results.jsonl" ] && { echo "skip (scored) $run"; continue; }
  $G/env/bin/python evaluation/evaluate_images.py "$images" --outfile "$run/geneval-results.jsonl" --model-path "$G/models"
  $G/env/bin/python evaluation/summary_scores.py "$run/geneval-results.jsonl" | tee "$run/geneval-summary.txt"
done
