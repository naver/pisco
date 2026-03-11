import torch
import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from pisco.model import PISCO
from pisco.eval_utils import build_ir_eval_from_splare_dataset
import numpy as np


def run_eval_mteb(pisco, config):
    """
    Tested with MTEB 2.1.11
    """
    import mteb
    from mteb import MTEB
    from mteb.models import ModelMeta
    from mteb.models import SearchEncoderWrapper

    meta = ModelMeta(
        **{
            "name": "naver/no_model_name_available",
            "revision": "no_revision_available",
            "release_date": None,
            "languages": None,
            "n_parameters": None,
            "memory_usage_mb": None,
            "max_tokens": None,
            "embed_dim": None,
            "license": None,
            "open_weights": False,
            "public_training_code": None,
            "public_training_data": None,
            "framework": [],
            "reference": None,
            "similarity_fn_name": None,
            "use_instructions": None,
            "training_datasets": None,
            "adapted_from": None,
            "superseded_by": None,
            "is_cross_encoder": None,
            "modalities": ["text"],
            "loader": None,
        }
    )
    meta.name = config.name
    pisco.mteb_model_meta = meta
    pisco_encoder = SearchEncoderWrapper(pisco)

    benchmark = mteb.get_benchmark(config.benchmark)
    tasks = mteb.filter_tasks(benchmark, task_types=["Retrieval"])

    evaluation = MTEB(tasks=tasks)

    # 3. run the evaluation
    evaluation.run(
        pisco_encoder,
        encode_kwargs={
            "batch_size": config.batch_size,
            "max_length": config.max_length,
        },
        output_folder=config.out_dir,
    )


@hydra.main(config_path="configs", config_name="eval_nano_beir", version_base="1.3")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    os.makedirs(config.out_dir, exist_ok=True)
    with open(
        os.path.join(config.out_dir, "eval_config.yaml"),
        "w",
    ) as f:
        OmegaConf.save(config, f)

    pisco = PISCO.from_pretrained(config.model_name_or_path, load_decoder=False).cuda()
    pisco.eval()
    pisco.prepare()

    # Splare-dataset-style evaluation
    if hasattr(config, "datasets"):
        for ds in config.datasets:
            print(f"Evaluating ds {ds.document_collection}")

            ir_evaluator = build_ir_eval_from_splare_dataset(
                d_collection=ds.document_collection,
                q_collection=ds.query_collection,
                data_local_cache=config.data_local_cache,
                batch_size=config.batch_size,
            )

            metrics_path = os.path.join(
                config.out_dir, f"{ds.collection_name}_metrics.json"
            )

            # Pisco class exposes sentence transformers methods so we can use its ir_evaluator directly
            results = ir_evaluator(pisco, output_path=config.out_dir)

            print(results)

            with open(metrics_path, "w") as f:
                json.dump(results, f, indent=2)

    # MTEB style evaluation:
    else:
        run_eval_mteb(pisco, config)


if __name__ == "__main__":
    with torch.no_grad():
        main()
