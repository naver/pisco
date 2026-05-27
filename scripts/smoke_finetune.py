"""Smoke test for the finetuning loop: 32 samples, 2 optimizer steps, eval + save.

Three modes:
    python scripts/smoke_finetune.py                          # fresh PISCO via Hydra overrides
    python scripts/smoke_finetune.py OUT --checkpoint CKPT    # load CKPT via PISCO.from_pretrained
    python scripts/smoke_finetune.py OUT --chain              # run smoke_pretrain.py first, then load its output

--chain exercises the full save->load->finetune round trip, including PISCO.from_pretrained
on a freshly-saved checkpoint.
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
    parser.add_argument("--checkpoint", default=None,
                        help="Existing PISCO checkpoint to load via from_pretrained")
    parser.add_argument("--chain", action="store_true",
                        help="Run smoke_pretrain.py first and load its output checkpoint")
    args = parser.parse_args()

    if args.chain and args.checkpoint:
        parser.error("--chain and --checkpoint are mutually exclusive")

    out = args.output_dir or tempfile.mkdtemp(prefix="pisco_smoke_finetune.")
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint
    if args.chain:
        pt_out = tempfile.mkdtemp(prefix="pisco_smoke_pretrain.")
        print(f"[smoke finetune] chaining via pretrain -> {pt_out}")
        rc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).parent / "smoke_pretrain.py"), pt_out]
        ).returncode
        if rc != 0:
            return rc
        ckpt = str(pathlib.Path(pt_out) / "model")
        if not pathlib.Path(ckpt).is_dir():
            print(f"[smoke finetune] expected {ckpt} to exist after pretrain", file=sys.stderr)
            return 1

    print(f"[smoke finetune] output_dir={out}  checkpoint={ckpt or '<fresh model>'}")

    if ckpt:
        model_overrides = [f"model_name_or_path={ckpt}"]
    else:
        model_overrides = [
            "+model.init_args._target_=pisco.model.PISCO",
            "+model.init_args.config._target_=pisco.model.PISCOConfig",
            "+model.init_args.config.decoder_model_name=Qwen/Qwen3-0.6B",
            "+model.init_args.config.compressor_model_name=Qwen/Qwen3-0.6B",
            "+model.init_args.config.compr_rate=16",
            "+model.init_args.config.compressor_mlp_hidden_dim=4096",
            "+model.init_args.config.lora_decoder=true",
            "+model.init_args.config.lora_r_decoder=64",
        ]

    cmd = [
        sys.executable, "pisco/train.py", "--config-name=finetune",
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
        "+hf_training.optim=adamw_torch",
        *model_overrides,
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[smoke finetune] OK -> {out}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
