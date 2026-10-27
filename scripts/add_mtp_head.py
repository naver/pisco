"""Copy of an exported PISCO decoder (scripts/vllm_prep_pisco.py) with the base model's MTP head added, for vLLM
speculative decoding (scripts/bench_vllm.py --spec_mtp N). Runs in the vLLM env (torch + safetensors, no pisco).

The head is the BASE Qwen3.5 one: it was trained on the base model's hidden states, not on the LoRA-merged PISCO
decoder's, so its acceptance rate on PISCO is what the benchmark measures. Files of the export are symlinked; only
config.json (+ mtp_num_hidden_layers) and mtp.safetensors are written.

  python scripts/add_mtp_head.py --export expQ/ft_mm_4b_172671/vllm_decoder --out expQ/archq/vllm_decoder_mtp
"""
import argparse
import json
import os
import pathlib

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--export", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
args = ap.parse_args()

src, out = pathlib.Path(args.export).resolve(), pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "config.json" and not (out / f.name).exists():
        os.symlink(f, out / f.name)
base = pathlib.Path(snapshot_download(args.base, allow_patterns=["*.json", "*.safetensors"]))
bcfg = json.load(open(base / "config.json"))
btext = bcfg.get("text_config", bcfg)
cfg = json.load(open(src / "config.json"))
for k in ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings"):
    cfg[k] = btext[k]
json.dump(cfg, open(out / "config.json", "w"), indent=2)

tensors = {}
for shard in sorted(base.glob("*.safetensors")):
    with safe_open(str(shard), "pt") as fh:
        for k in fh.keys():
            if k.startswith("mtp."):
                tensors[k] = fh.get_tensor(k).contiguous()
assert tensors, f"no mtp.* weights in {base}"
save_file(tensors, str(out / "mtp.safetensors"), metadata={"format": "pt"})
print(f"wrote {out}: {len(tensors)} mtp tensors, config mtp_num_hidden_layers={cfg['mtp_num_hidden_layers']}")
