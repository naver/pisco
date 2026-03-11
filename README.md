![image](https://oss.navercorp.com/nle-ram/pisco/assets/43696/7f5a9b5d-4e3f-4853-916c-7bc95e0d8feb)


## Example commands

Pretraining:
```
python train.py out_dir=/beegfs/scratch/user/mlouis/calmar/pisco/results/pretraining/Llama_1B_8B_ae_500k \
    ++data.samples=500000 ++hf_training.logging_steps=50 ++hf_training.eval_steps=100 ++ae_ratio=1. \
    ++model.init_args.config.decoder_model_name=meta-llama/Llama-3.1-8B-Instruct ++model.init_args.config.compressor_model_name=meta-llama/Llama-3.2-1B-Instruct
```

Fine-tuning (on RAG multiqa data here):
```
python train.py --config-name=finetune \
    out_dir=/beegfs/scratch/user/mlouis/calmar/pisco/results/finetuning/Llama_1B_8B_ae_500k_ft \
    ++data.samples=500000 ++hf_training.logging_steps=50 ++hf_training.eval_steps=100 ++hf_training.gradient_accumulation_steps=16 \
    ++model_name_or_path=/beegfs/scratch/user/mlouis/calmar/pisco/results/pretraining/Llama_1B_8B_ae_500k/model
```

## How it works

PISCO model holds a 'compressor' and a 'decoder' which are hf models. PISCO expects inputs formatted as done by the different collators.

PISCO "generate" method is decomposed into a compression step, a step where the compression is included in the full decoder prompt, and the actual generation step.
