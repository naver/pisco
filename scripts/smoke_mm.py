"""Smoke test for the MULTIMODAL pipeline: 2 steps of stage-2 pretraining from a text PISCO
checkpoint -> save -> PISCO.from_pretrained (strict adapter check) -> 2 steps of multimodal
finetuning -> eval_mm_docqa.py on 4 rows. Needs the mm datasets to exist (build_mm_*.py).

  python scripts/smoke_mm.py [OUT] --ckpt expQ/qwen35_4b_pt_5M_143228/model \
      --pt_data data/mm_pretrain_160k --ft_data data/mm_ft_docpages_20k --eval_data eval_data/mm_docqa_500
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).parent.parent


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", nargs="?", default=None)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "expQ/qwen35_4b_pt_5M_143228/model"))
    ap.add_argument("--pt_data", default=str(REPO_ROOT / "data/mm_pretrain_160k"))
    ap.add_argument("--ft_data", default=str(REPO_ROOT / "data/mm_ft_docpages_20k"))
    ap.add_argument("--eval_data", default=str(REPO_ROOT / "eval_data/mm_docqa_500"))
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir or tempfile.mkdtemp(prefix="pisco_smoke_mm."))
    pt_out, ft_out = out / "pt", out / "ft"
    for p in (pt_out, ft_out):
        p.mkdir(parents=True, exist_ok=True)
    common = [
        "data.samples=32",
        "hf_training.per_device_train_batch_size=2",
        "hf_training.per_device_eval_batch_size=2",
        "hf_training.gradient_accumulation_steps=1",
        "+hf_training.max_steps=2",
        "hf_training.warmup_steps=0",
        "hf_training.logging_steps=1",
        "hf_training.eval_steps=2",
        "hf_training.save_steps=2",
        "hf_training.dataloader_num_workers=0",
    ]
    rc = run([sys.executable, "pisco/train.py", "--config-name=pt_lora_mm", f"hydra.run.dir={pt_out}",
              f"model_name_or_path={pathlib.Path(args.ckpt).resolve()}",
              f"data.training_dataset={pathlib.Path(args.pt_data).resolve()}", *common])
    if rc:
        return rc
    pt_model = pt_out / "model"
    if not (pt_model / "connector.pt").exists():
        print("[smoke mm] pretrain did not save a model", file=sys.stderr); return 1
    if not (pt_model / "compressor" / "preprocessor_config.json").exists():
        print("[smoke mm] image processor was not persisted in the checkpoint", file=sys.stderr); return 1

    rc = run([sys.executable, "pisco/train.py", "--config-name=finetune_mm", f"hydra.run.dir={ft_out}",
              f"model_name_or_path={pt_model.resolve()}",
              f"data.training_dataset={pathlib.Path(args.ft_data).resolve()}", *common])
    if rc:
        return rc
    ft_model = ft_out / "model"
    rc = run([sys.executable, "scripts/eval_mm_docqa.py", "--mode", "pisco", "--checkpoint_path", str(ft_model.resolve()),
              "--data_path", str(pathlib.Path(args.eval_data).resolve()), "--output_path", str(out / "eval_smoke.json"),
              "--limit", "4"])
    if rc == 0:
        print(f"[smoke mm] OK -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
