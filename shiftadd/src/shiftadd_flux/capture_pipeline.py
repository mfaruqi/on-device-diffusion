"""Capture selected real Klein activations on calibration prompts only."""

import argparse
import json
import os
import traceback
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

from shiftadd_flux.capture import ActivationCapture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    write("config.json", config)
    write("status.json", {"status": "running"})
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        write(
            "environment.json",
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "node": os.uname().nodename,
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
            },
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            config["model_path"], torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        with (
            ActivationCapture(
                pipe.transformer,
                config["modules"],
                config["tokens_per_call"],
                config["max_calls"],
            ) as capture,
            torch.inference_mode(),
        ):
            for sample in config["samples"]:
                pipe(
                    prompt=sample["prompt"],
                    height=512,
                    width=512,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    max_sequence_length=512,
                    generator=torch.Generator(device="cuda").manual_seed(sample["seed"]),
                    output_type="latent",
                )
        for name, samples in capture.samples.items():
            if len(samples) != 4 * len(config["samples"]):
                raise RuntimeError(f"Unexpected call count: {name}")
        summary = capture.save(out / "activations.npz")
        write(
            "summary.json",
            {
                "modules": summary,
                "scope": "BF16 calibration activation capture; CPU transfers invalidate timing",
                "call_order": "sample order, then four denoising calls per sample",
            },
        )
        write("status.json", {"status": "completed"})
    except BaseException:
        write("status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
