"""Smoke test for the pretraining loop: 32 samples, 2 optimizer steps, eval + save.

Pass an output dir as argv[1], or one will be auto-created. Returns 0 on success.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Where to write the smoke run (default: auto temp dir)")
    args = parser.parse_args()

    out = args.output_dir or tempfile.mkdtemp(prefix="pisco_smoke_pretrain.")
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    print(f"[smoke pretrain] output_dir={out}")

    cmd = [
        sys.executable, "pisco/train.py", "--config-name=pretraining",
        f"output_dir={out}",
        "data.samples=32",
        "hf_training.per_device_train_batch_size=2",
        "hf_training.per_device_eval_batch_size=2",
        "hf_training.gradient_accumulation_steps=1",
        "+hf_training.max_steps=2",
        "hf_training.warmup_steps=0",
        "hf_training.logging_steps=1",
        "hf_training.eval_steps=2",
        "hf_training.save_steps=2",
        "hf_training.gradient_checkpointing=false",
        "hf_training.optim=adamw_torch",
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[smoke pretrain] OK -> {out}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
