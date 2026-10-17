"""
PISCO
Copyright (c) 2026-present NAVER Corp.
All Rights Reserved.
"""

import datasets
import datasets.config
import inspect
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
from transformers.trainer_utils import get_last_checkpoint

from pisco.metrics import hard_metrics
from pisco.collator_utils import print_collated_sample
from pisco.model import PISCO, load_trainable_state


class PISCOTrainer(Trainer):
    """HF Trainer whose resume path understands PISCO's partial checkpoints.

    Trainer._save -> PISCO.save_pretrained writes decoder_state.pt / compressor/compressor_state.pt /
    connector.pt into checkpoint-N (next to optimizer.pt, scheduler.pt, trainer_state.json, rng).
    The stock _load_from_checkpoint only knows HF weight files and raises "Can't find a valid
    checkpoint"; this override loads our files instead. Optimizer/scheduler/RNG/trainer state are
    restored by the parent as usual, so a requeued (preempted) job continues where it stopped.
    """

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        m = model if model is not None else self.model
        m = m.module if hasattr(m, "module") else m
        if hasattr(self, "accelerator"):
            m = self.accelerator.unwrap_model(m, keep_torch_compile=False)
        if not os.path.exists(os.path.join(resume_from_checkpoint, "connector.pt")):
            raise ValueError(f"Can't find a PISCO checkpoint (connector.pt) at {resume_from_checkpoint}")
        load_trainable_state(m, resume_from_checkpoint,
                             decoder=not m.config.freeze_decoder, compressor=bool(m.config.lora_compressor))

    # Loss components (online distillation): PISCO.forward returns ce_loss / kd_loss next to
    # the combined loss while training; average them between two log() calls and report them
    # as `ce` / `kd` so the KD weight can be judged from the log (train `loss` is CE + w*KD;
    # eval loss stays plain CE).
    _parts = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        if isinstance(outputs, dict) and "kd_loss" in outputs:
            if self._parts is None:
                self._parts = {"ce": 0.0, "kd": 0.0, "n": 0}
            self._parts["ce"] += float(outputs["ce_loss"])
            self._parts["kd"] += float(outputs["kd_loss"])
            self._parts["n"] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._parts and self._parts["n"] and "loss" in logs:
            n = self._parts["n"]
            logs = {**logs, "ce": round(self._parts["ce"] / n, 4), "kd": round(self._parts["kd"] / n, 4)}
            self._parts = None
        return super().log(logs, *args, **kwargs)


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
    # Select up to 2 random indices from the available range
    random_indices = random.sample(range(len(preds_str)), min(2, len(preds_str)))

    # Print random examples
    for i in random_indices:
        print("\n")
        print("###############")
        print("#####LABEL: ", labels_str[i])
        print("#####PRED:  ", preds_str[i])

    print("_" * 15, flush=True)
    metrics = hard_metrics(predictions=preds_str, references=labels_str)

    return metrics


@hydra.main(config_path="configs", config_name="pretraining", version_base="1.3")
def main(config: DictConfig):
    print("Training config:")
    print(OmegaConf.to_yaml(config, resolve=True))
    with open("training_config.yaml", "w") as f:
        OmegaConf.save(config, f)

    print("Output directory:", os.getcwd())

    # Model
    if getattr(config, "model_name_or_path", None) is not None:
        print(f"Loading existing model at {config.model_name_or_path}")
        # Optional config overrides applied to the saved checkpoint config before build,
        # e.g. +model_config_overrides.frozen_base_dtype=float16 for a V100 finetune.
        _ov = (OmegaConf.to_container(config.model_config_overrides, resolve=True)
               if getattr(config, "model_config_overrides", None) else {})
        if _ov:
            print(f"Applying model_config_overrides: {_ov}")
        model = PISCO.from_pretrained(config.model_name_or_path, **_ov)
    else:
        print("Creating new PISCO model")
        model = cast(PISCO, instantiate(config.model.init_args))

    print(model)

    # Data:
    def _load_one(spec: str):
        # "path" or "path:N" (N = cap on rows taken from that dataset, first N rows)
        path, _, cap = spec.rpartition(":") if ":" in spec and spec.rsplit(":", 1)[-1].isdigit() else (spec, "", "")
        if os.path.exists(path):
            d = datasets.load_from_disk(path)
            if isinstance(d, datasets.DatasetDict):   # a plain Dataset dir (e.g. one shard) is used as-is
                d = d["train"]
        else:
            d = datasets.load_dataset(path, split="train")
        if cap:
            d = d.select(range(min(len(d), int(cap))))
        return d

    specs = [t.strip() for t in str(config.data.training_dataset).split(",") if t.strip()]
    if len(specs) == 1:
        ds = _load_one(specs[0])
    else:   # joint diet: several datasets with the same schema, concatenated then shuffled
        parts = [_load_one(t) for t in specs]
        print("Joint training set:", {t: len(p) for t, p in zip(specs, parts)})
        ds = datasets.concatenate_datasets(parts).shuffle(seed=42)

    if isinstance(ds, list):
        raise TypeError("Expected a Hugging Face Dataset, got a list instead.")
    ds = ds.select(range(min(len(ds), config.data.samples)))
    ds = ds.train_test_split(test_size=min(64, len(ds) // 10))
    train_ds, eval_ds = ds["train"], ds["test"]
    train_ds = train_ds.shuffle(seed=42)

    print("Train_ds", train_ds)
    print("eval_ds", eval_ds)

    collator_cls = get_class(config.collator_class)
    collator_kwargs = dict(config.collator_kwargs)
    # Multimodal collators take the model's image processor; legacy collators don't know
    # the kwarg, so only pass it when the constructor declares it.
    if "image_processor" in inspect.signature(collator_cls.__init__).parameters:
        if getattr(model, "image_processor", None) is None:
            raise ValueError(
                f"{config.collator_class} needs an image processor but the compressor "
                f"{model.config.compressor_model_name} has no vision tower / processor"
            )
        collator_kwargs["image_processor"] = model.image_processor
    collator = collator_cls(
        compressor_tokenizer=model.compressor_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        compr_rate=model.compr_rate,
        **collator_kwargs,
    )

    # We just print an item to see the data:
    sample_batch = next(
        iter(DataLoader(train_ds, batch_size=2, collate_fn=collator.torch_call))
    )
    print_collated_sample(sample_batch, collator)

    training_args = TrainingArguments(output_dir=".", **config.hf_training)

    trainer = PISCOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=lambda e: compute_metrics(e, model=model),
        data_collator=collator.torch_call,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Resume: with a FIXED run dir (hydra.run.dir) a requeued/preempted job finds its last
    # checkpoint-N here and PISCOTrainer reloads the partial PISCO files from it.
    resume = get_last_checkpoint(os.getcwd()) if getattr(config, "resume", True) else None
    if resume:
        print(f"Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # save final
    if trainer.is_world_process_zero():
        final_path = "model"
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
