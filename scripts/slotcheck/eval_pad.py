"""Campaign slotcheck, arm P (check 2b): the shared document eval with the compressor batch padded as in TRAINING.

Imports scripts/eval_mm_docqa.py (never edited) and wraps its make_collator so the collator carries
pad_to_multiple_of = SLOTCHECK_PAD (env; "64" = training's value, pisco/mm_collator.py:431; "none" = the eval's own
None, the control). collate_compressor reads that attribute to pad the compressor batch; nothing else in the eval
uses it (the collator serves only example_chunks there). Every CLI flag goes to eval_mm_docqa.main() unchanged.

  SLOTCHECK_PAD=64 python scripts/slotcheck/eval_pad.py --mode pisco --checkpoint_path ... (eval_mm_docqa.py flags)
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import eval_mm_docqa as E  # noqa: E402

raw = os.environ.get("SLOTCHECK_PAD", "").strip().lower()
assert raw in ("none", "64"), f"SLOTCHECK_PAD must be 'none' or '64', got {raw!r}"
PAD = None if raw == "none" else int(raw)
_orig = E.make_collator


def make_collator(*args, **kwargs):
    c = _orig(*args, **kwargs)
    assert c.pad_to_multiple_of is None, "the shared eval changed its padding: this wrapper's control no longer holds"
    c.pad_to_multiple_of = PAD
    return c


E.make_collator = make_collator   # main() looks the name up in its module at call time
print(f"[eval_pad] compressor pad_to_multiple_of={PAD} (eval_mm_docqa.py imported, unchanged)", flush=True)

if __name__ == "__main__":
    sys.exit(E.main())
