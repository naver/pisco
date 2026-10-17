"""Stage-2 v2 mix: dense PDF pages (data/mm_pretrain_pdfa) + a slice of the Wikipedia screenshots
(data/mm_pretrain_160k) so screenshot skills are rehearsed. Same {image, text, title, docid, source} schema.

  python scripts/build_mm_pretrain_v2.py --pdfa data/mm_pretrain_pdfa --wiki data/mm_pretrain_160k --n_wiki 40000 \
      --out data/mm_pretrain_v2
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfa", required=True)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--n_wiki", type=int, default=40000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import datasets

    def load(p):
        d = datasets.load_from_disk(p)
        return d["train"] if isinstance(d, datasets.DatasetDict) else d

    pdfa, wiki = load(args.pdfa), load(args.wiki)
    wiki = wiki.shuffle(seed=args.seed).select(range(min(args.n_wiki, len(wiki))))
    cols = ["image", "text", "title", "docid", "source"]
    pdfa = pdfa.select_columns(cols).cast(wiki.select_columns(cols).features)
    full = datasets.concatenate_datasets([pdfa, wiki.select_columns(cols)]).shuffle(seed=args.seed)
    datasets.DatasetDict({"train": full}).save_to_disk(args.out)
    print(f"v2 mix: {len(pdfa)} pdfa + {len(wiki)} wiki = {len(full)} rows -> {args.out}")


if __name__ == "__main__":
    main()
