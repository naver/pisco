#!/usr/bin/env python
"""Campaign rebuild: a run's resolved training_config.yaml may differ from the recorded recipe ONLY in the keys
the law allows for its arm (campaign_rebuild/CAMPAIGN.md, "Rules").

  python scripts/rebuild/config_check.py <recorded training_config.yaml> <run training_config.yaml> <allowed,...>

`allowed` = comma-separated dotted key prefixes (e.g. hf_training.seed,collator_kwargs.mix_override). Prints every
difference marked ALLOWED / UNEXPECTED. Exit 0 = only allowed differences, 3 = an unexpected one (the launcher then
kills the run), 2 = bad usage. Needs omegaconf (the pixi env); runs on a compute node, never the login node.
"""

import sys

from omegaconf import OmegaConf

MISSING = "<absent>"


def flat(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = node
    return out


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 2
    rec_path, run_path, allowed = argv[1], argv[2], [p for p in argv[3].split(",") if p]
    rec = flat(OmegaConf.to_container(OmegaConf.load(rec_path), resolve=False))
    run = flat(OmegaConf.to_container(OmegaConf.load(run_path), resolve=False))
    keys = sorted(k for k in set(rec) | set(run) if rec.get(k, MISSING) != run.get(k, MISSING))
    bad = 0
    print(f"[config_check] {run_path} vs {rec_path}; allowed: {allowed}")
    for k in keys:
        ok = any(k == p or k.startswith(p + ".") for p in allowed)
        bad += not ok
        print(f"  {'ALLOWED   ' if ok else 'UNEXPECTED'} {k}: {rec.get(k, MISSING)!r} -> {run.get(k, MISSING)!r}")
    print(f"[config_check] {len(keys)} difference(s), {bad} unexpected -> {'FAIL' if bad else 'OK'}")
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
