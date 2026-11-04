"""Paired image smoke test; dense reconstructed execution, not packed kernels."""

import argparse
import json
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2KleinPipeline

from shiftadd_flux.activation_hooks import ActivationQuantization
from shiftadd_flux.case_validation import validate_case_sequence
from shiftadd_flux.loader import apply_artifacts, apply_uniform_artifacts
from shiftadd_flux.upstream_loader import apply_upstream_artifacts


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    out = Path(c["output"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "config.json", c)
    write(out / "status.json", {"status": "running"})
    try:
        write(out / "case-sequence-audit.json", validate_case_sequence(c))
        assert torch.cuda.is_available()
        write(
            out / "environment.json",
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "execution": "dense BF16 reconstruction; quality smoke only",
            },
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            c["model_path"], torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        results = []
        reference = None
        for case in c["cases"]:
            if case["artifacts"] is not None:
                if case.get("weight_format", "sign_planes") == "upstream_bcq":
                    audit = apply_upstream_artifacts(
                        pipe.transformer,
                        case["artifacts"],
                        case.get("tensors"),
                        chunk_rows=c.get("decode_chunk_rows", 128),
                    )
                elif case.get("weight_format", "sign_planes") == "symmetric_uniform":
                    audit = apply_uniform_artifacts(
                        pipe.transformer,
                        case["artifacts"],
                        case.get("tensors"),
                        chunk_rows=c.get("decode_chunk_rows", 128),
                    )
                elif case.get("weight_format", "sign_planes") == "sign_planes":
                    audit = apply_artifacts(
                        pipe.transformer,
                        case["artifacts"],
                        case["planes"],
                        case["tensors"],
                        chunk_rows=c.get("decode_chunk_rows"),
                    )
                else:
                    raise ValueError("Unknown weight format")
                write(out / (case["name"] + "-audit.json"), audit)
            generator = torch.Generator(device="cuda").manual_seed(c["seed"])
            torch.cuda.synchronize()
            start = time.perf_counter()
            activation = case.get("activation_quantization")
            context = (
                ActivationQuantization(
                    pipe.transformer,
                    activation["modules"],
                    activation["bits"],
                    group_size=activation.get("group_size"),
                )
                if activation
                else nullcontext()
            )
            with torch.inference_mode(), context:
                latent = pipe(
                    prompt=c["prompt"],
                    height=c["height"],
                    width=c["width"],
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    max_sequence_length=512,
                    generator=generator,
                    output_type="latent",
                ).images
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if activation:
                if context.calls == 0:
                    raise ValueError("Activation hooks were never called")
                write(
                    out / (case["name"] + "-activation-audit.json"),
                    {
                        "bits": activation["bits"],
                        "modules": activation["modules"],
                        "calls": context.calls,
                        "calls_by_module": context.calls_by_module,
                        "group_size": activation.get("group_size"),
                        "execution": "fake quantization with floating GEMMs",
                    },
                )
                expected = activation.get("expected_calls_per_module")
                if expected is not None and any(
                    n != expected for n in context.calls_by_module.values()
                ):
                    raise ValueError(
                        "Activation coverage disagrees with configured per-module call count"
                    )
            with torch.inference_mode():
                decoded = pipe.vae.decode(latent, return_dict=False)[0]
                image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
                image.save(out / (case["name"] + ".png"))
            array = latent.float().cpu().numpy()
            np.save(out / (case["name"] + "-latent.npy"), array)
            if reference is None:
                reference = array.copy()
            results.append(
                {
                    "case": case["name"],
                    "generation_to_latent_s": elapsed,
                    "relative_latent_l2": float(
                        np.linalg.norm(array - reference) / np.linalg.norm(reference)
                    ),
                    "finite": bool(np.isfinite(array).all()),
                }
            )
            write(
                out / "summary.json",
                {
                    "results": results,
                    "quality_scope": "one calibration prompt, diagnostic only",
                    "unavailable_metrics": [
                        "packed_runtime_latency",
                        "packed_runtime_memory",
                        "energy",
                    ],
                },
            )
        write(out / "status.json", {"status": "completed"})
    except BaseException:
        write(
            out / "status.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
