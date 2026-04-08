"""
PISCO
Copyright (c) 2026-present NAVER Corp.
All Rights Reserved.
"""

import datasets
import datasets.config
import os
import hydra
from typing import cast

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_class


from transformers import Trainer, TrainingArguments

from pisco.metrics import hard_metrics
from pisco.collator_utils import print_collated_sample
from pisco.model import PISCO
from pisco.hydra_utils import register_resolvers


def preprocess_logits_for_metrics(logits, labels):
    # logits can be tuple for some models
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred, model):
    """
    Computes the evaluation metrics
    """
    preds, labels = eval_pred

    original_model = model.module if hasattr(model, "module") else model
    ignore_positions = labels == -100

    labels[ignore_positions] = original_model.decoder_tokenizer.pad_token_id
    preds[ignore_positions] = original_model.decoder_tokenizer.pad_token_id

    preds_str = original_model.decoder_tokenizer.batch_decode(
        preds, skip_special_tokens=True
    )
    labels_str = original_model.decoder_tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    print("_" * 15)
    # Select two random indices from the available range
    random_indices = random.sample(range(len(preds_str)), 2)

    # Print random examples
    for i in random_indices:
        print("\n")
        print("###############")
        print("#####LABEL: ", labels_str[i])
        print("#####PRED:  ", preds_str[i])

    print("_" * 15, flush=True)
    metrics = hard_metrics(predictions=preds_str, references=labels_str)
    # not all keys matter during this training eval
    metrics = {
        k: v
        for k, v in metrics.items()
        if k in ["Rouge-1", "Rouge-2", "Rouge-L", "EM", "Levenshtein"]
    }

    return metrics


register_resolvers()


@hydra.main(config_path="configs", config_name="pretraining", version_base="1.3")
def main(config: DictConfig):
    print("Training config:")
    print(OmegaConf.to_yaml(config, resolve=True))
    os.makedirs(config.out_dir, exist_ok=True)
    with open(
        os.path.join(
            config.out_dir,
            "training_config.yaml",
        ),
        "w",
    ) as f:
        OmegaConf.save(config, f)

    print("Output directory:", config.out_dir)

    # Model
    model_cls = cast(type[PISCO], get_class(config.model_class))
    if getattr(config, "model_name_or_path", None) is not None:
        print(f"Loading existing {model_cls.__name__} at {config.model_name_or_path}")
        model = model_cls.from_pretrained(config.model_name_or_path)
    else:
        print(f"Creating new {model_cls.__name__} model")
        model = cast(PISCO, instantiate(config.model.init_args))

    print(model)

    # Data:
    if os.path.exists(config.data.training_dataset):
        ds = datasets.load_from_disk(config.data.training_dataset)["train"]
    else:
        ds = datasets.load_dataset(config.data.training_dataset, split="train")

    if isinstance(ds, list):
        raise TypeError("Expected a Hugging Face Dataset, got a list instead.")
    ds = ds.select(range(min(len(ds), config.data.samples)))
    ds = ds.train_test_split(test_size=min(64, len(ds) // 10))
    train_ds, eval_ds = ds["train"], ds["test"]
    train_ds = train_ds.shuffle(seed=42)

    print("Train_ds", train_ds)
    print("eval_ds", eval_ds)

    collator = get_class(config.collator_class)(
        compressor_tokenizer=model.compressor_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        compr_rate=model.compr_rate,
        **config.collator_kwargs,
    )

    # We just print an item to see the data:
    sample_batch = next(
        iter(DataLoader(train_ds, batch_size=2, collate_fn=collator.torch_call))
    )
    print_collated_sample(sample_batch, collator)

    training_args = TrainingArguments(output_dir=config.out_dir, **config.hf_training)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=lambda e: compute_metrics(e, model=model),
        data_collator=collator.torch_call,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    # save final
    if trainer.is_world_process_zero():
        final_path = os.path.join(config.out_dir, "model")
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        model.save_pretrained(final_path)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    datasets.disable_caching()
    datasets.config.IN_MEMORY_MAX_SIZE = float(200 * 1024**3)  # 200 GB

    main()
