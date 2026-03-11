import datasets
import os
import hydra
import numpy as np
import random
import torch
import itertools

from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_class
from transformers import TrainingArguments
from transformers import TrainingArguments

from pisco.trainer import RetrievalPiscoTrainer
from pisco.metrics import hard_metrics
from pisco.collator_utils import print_collated_sample
from pisco.model import PISCO
from pisco.train import preprocess_logits_for_metrics, compute_metrics, prepare_dataset


class AlternatingDataLoader:
    def __init__(self, dl_a, dl_b, stop="min"):
        self.dl_a = dl_a
        self.dl_b = dl_b
        self.stop = stop  # "min", "max", or "cycle"

    def __iter__(self):
        it_a = iter(self.dl_a)
        it_b = iter(self.dl_b)

        if self.stop == "cycle":
            it_a = itertools.cycle(it_a)
            it_b = itertools.cycle(it_b)

        while True:
            try:
                yield next(it_a)
            except StopIteration:
                if self.stop == "min":
                    return
                raise

            try:
                yield next(it_b)
            except StopIteration:
                if self.stop == "min":
                    return
                raise

    def __len__(self):
        if self.stop == "min":
            return 2 * min(len(self.dl_a), len(self.dl_b))
        elif self.stop == "max":
            return 2 * max(len(self.dl_a), len(self.dl_b))
        else:
            raise TypeError("cycle has no finite length")


class AlternatingRetrievalPiscoTrainer(RetrievalPiscoTrainer):
    def __init__(
        self,
        *args,
        train_dataset_a=None,
        train_dataset_b=None,
        collator_a=None,
        collator_b=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_dataset_a = train_dataset_a
        self.train_dataset_b = train_dataset_b
        self.collator_a = collator_a
        self.collator_b = collator_b

    def get_eval_dataloader(self, eval_dataset=None):
        # If HF passes an eval_dataset explicitly, just use it with the *right* collator
        # (choose collator based on dataset identity if you want)
        if eval_dataset is not None:
            # safest fallback: require you to pass eval_dataset_a/b and pick accordingly
            # but if you only evaluate on A, just use collator_a here
            return DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.collator_a,  # or pick depending on which dataset
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    def get_train_dataloader(self):
        # IMPORTANT: use Trainer's batch size / workers / etc.
        bs = self.args.per_device_train_batch_size

        dl_a = DataLoader(
            self.train_dataset_a,
            batch_size=bs,
            shuffle=True,  # or use samplers if you’re in DDP
            collate_fn=self.collator_a,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        dl_b = DataLoader(
            self.train_dataset_b,
            batch_size=bs,
            shuffle=True,
            collate_fn=self.collator_b,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return AlternatingDataLoader(dl_a, dl_b)


@hydra.main(
    config_path="configs",
    config_name="finetune_with_retrieval_multiple",
    version_base="1.3",
)
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
    if getattr(config, "model_name_or_path", None) is not None:
        print(f"Loading existing model at {config.model_name_or_path}")
        model = PISCO.from_pretrained(config.model_name_or_path)
    else:
        print("Creating new PISCO model")
        model = instantiate(config.model.init_args)

    # don't forget to call this ...
    model.prepare(retrieval_pooling=config.retrieval_pooling)

    print(model)

    # DATA FOR GENERATION
    if os.path.exists(config.data.gen_training_dataset):
        ds_gen = datasets.load_from_disk(config.data.gen_training_dataset)["train"]
    else:
        ds_gen = datasets.load_dataset(config.data.gen_training_dataset, split="train")

    train_ds_gen, eval_ds_gen = prepare_dataset(ds_gen, config)

    print("Train/Eval ds Generation", train_ds_gen, eval_ds_gen)

    collator_gen = get_class(config.gen_collator_class)(
        compressor_tokenizer=model.compressor_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        compr_rate=model.compr_rate,
        **config.gen_collator_kwargs,
    )

    # DATA FOR RETRIEVAL
    if os.path.exists(config.data.ret_training_dataset):
        ds_ret = datasets.load_from_disk(config.data.ret_training_dataset)["train"]
    else:
        ds_ret = datasets.load_dataset(config.data.ret_training_dataset, split="train")

    train_ds_ret, eval_ds_ret = prepare_dataset(ds_ret, config)

    print("Train/Eval ds Retrieval", train_ds_ret, eval_ds_ret)

    collator_ret = get_class(config.ret_collator_class)(
        compressor_tokenizer=model.compressor_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        compr_rate=model.compr_rate,
        **config.ret_collator_kwargs,
    )

    training_args = TrainingArguments(output_dir=config.out_dir, **config.hf_training)

    trainer = AlternatingRetrievalPiscoTrainer(
        contrastive_weight=config.contrastive_weight,
        contrastive_temperature=config.contrastive_temperature,
        kl_weight=config.kl_weight,
        kl_temperature=config.kl_temperature,
        generation_weight=config.generation_weight,
        eval_d_collection=config.eval.d_collection,
        eval_q_collection=config.eval.q_collection,
        data_local_cache=config.eval.data_local_cache,
        eval_batch_size=config.eval.batch_size,
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_ds_gen,
        train_dataset_a=train_ds_gen,
        train_dataset_b=train_ds_ret,
        collator_a=collator_gen.torch_call,
        collator_b=collator_ret.torch_call,
        compute_metrics=lambda e: compute_metrics(e, model=model),
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
    datasets.config.IN_MEMORY_MAX_SIZE = 200 * 1024**3  # 200 GB

    main()
