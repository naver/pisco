# PISCO/OSCAR

PISCO is a model designed for faster RAG inference. Given some RAG collection, PISCO offers to precompute and store embeddings for each of your documents. During inference, instead of using the texts, PISCO features an LLM which can work from the embeddings representations directly. 

PISCO is a **compress-then-generate** model:

- **Compressor (small LM)** reads a document containing `<MEM>` markers and outputs hidden states at those positions.
- **Connector (MLP)** maps those hidden states to the **decoder** hidden size.
- **Decoder (large LM, optionally LoRA)** receives the compressed embeddings by **replacing `<MEM>` tokens in its input embeddings**, then generates normally.

It is implemented as a huggingface (v.4) pretrained model.

OSCAR is very similar to PISCO except that the compression operation is made in a query-dependent fashion (it boosts accuracy) and with a small compressor model (to keep latency advantages).

---

## Compress a document:

```
doc = "Paris is the capital of France. <MEM> It is in Europe. <MEM>"

tok = model.compressor_tokenizer
batch = tok(doc, return_tensors="pt").to("cuda")

embs = model.compress(batch["input_ids"], batch["attention_mask"])  # (num_mem_tokens, decoder_hidden_dim)
````

## Generate from (document + query)

```
document = "Paris is the capital of France. <MEM>"
query = "What is the capital of France?"

comp = model.compressor_tokenizer(document, return_tensors="pt").to("cuda")
dec  = model.decoder_tokenizer("<MEM>\nQ: "+query+"\nA:", return_tensors="pt").to("cuda")

out = model.generate({
    "compressor_input_ids": comp["input_ids"],
    "compressor_attention_mask": comp["attention_mask"],
    "decoder_input_ids": dec["input_ids"],
    "decoder_attention_mask": dec["attention_mask"],
}, max_new_tokens=64)
```

Once compressed, the (variable-length) embeddings for each element of `compressor_input_ids` are inserted in order into the `<MEM>` placeholders in `decoder_input_ids`. 
In particular, in RAG finetuning, different queries may have different number of associated documents, each of them having a variable number of compressed tokens.

## Training

Training a PISCO/OSCAR model typically requires:
- a pretraining stage, during which PISCO/OSCAR is pretrained as a text auto-encoder or as embedding-conditioned text generator. Example quick command:
```
python train.py out_dir=PRETRAINED_MODEL_OUT_PATH \
    ++data.samples=1000 ++hf_training.logging_steps=1 ++hf_training.eval_steps=1 \
    ++model.init_args.config.decoder_model_name=Qwen/Qwen3-0.6B ++model.init_args.config.compressor_model_name=Qwen/Qwen3-0.6B 
```
- a fine-tuning stage, during which PISCO/OSCAR is fine-tuned on RAG-QA data (though one can imagine other applications !). With a command like:
```
 --config-name=finetune out_dir=FINETUNED_MODEL_OUT_PATH \
    ++data.samples=1000 ++hf_training.logging_steps=1 ++hf_training.eval_steps=1 \
    ++model_name_or_path=PRETRAINED_MODEL_OUT_PATH
````

If you use `data.samples=500000` and the appropriate backbones, you should reproduce PISCO results. We do not release the full data for OSCAR release, but you can produce a decent model using PISCO data and `++collator_kwargs.query_dependent=True` during fine-tuning.


## Data-preprocessing and collation

`collator.py` contains collators designed to facilitate training models. They prescribe which texts are compressed, and where the obtained embeddings should be placed within every decoder input. By implementing a collator, one can define fully new routines for embeddings-aware models:

- PretrainingCollator expects inputs containing `texts` and returns pisco-forward inputs for autoencoding or text continuation
- FinetuningCollator expects `docs` `queries` and 'mistral_labels` and enables RAG-like training: the core of PISCO/OSCAR.

### Implementing your own collator

You can implement your own routine which defines which parts to compress and where they should be place in the decoder context:

```
class MyCustomCollator(BaseCollator):
    def torch_call(self, examples):
        # 1. Clean and Tokenize
        # 2. Add Memory Tokens to Compressor IDs
        # 3. Create Decoder Prompt with matching <MEM> count
        # 4. Pad both and Mask Labels
        # 5. Assert Alignment and Return Dict
```


> Rule: the number of `<MEM>` tokens in the decoder prompt must match the number of `<MEM>` tokens in the compressor input.

## 

Note that we released PISCO and OSCAR models from a completely different implementation of the code. THis one is much cleaner and it mostly reproduces the results (+/- ~1%). This implementation of PISCO/OSCAR collators places a number of `<MEM>` tokens proportional to the length of the input documents for compression (unlike in the papers). This implementation also uses a single <MEM> token, where the original PISCO code was using <MEM1>, <MEM2> ... but ablation shows it does not really matter. Also, this implementation systematically uses a "connector" 2-layer MLP between compressor hidden states and decoder input embeddings. This prevents the development of no-pretraining PISCO models, but it enables to use small compressors (which are faster, and generally about as good).

> For OSCAR, compression is query-dependent. The model class is agnostic to this choice: it only materializes with the `query_dependent` arg of the finetuning collator.


## Dependencies

This code heavily relies on `transformers`, `datasets` and `hydra`. It requires `flash_attention` for optimal performance. The given requirements are mostly indicative. With `transformers` close to 4.50 you should be fine.

# Cite
## PISCO
``` 
@inproceedings{louis-etal-2025-pisco,
    title = "{PISCO}: Pretty Simple Compression for Retrieval-Augmented Generation",
    author = "Louis, Maxime  and
      D{\'e}jean, Herv{\'e}  and
      Clinchant, St{\'e}phane",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    year = "2025",
    address = "Vienna, Austria",
}
```
## OSCAR:
```
@inproceedings{
    louis2026oscar,
    title={{OSCAR}: Online Soft Compression for {RAG}},
    author={Maxime Louis and Thibault Formal and Herv{\'e} D{\'e}jean and St{\'e}phane Clinchant},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=ideKAUWvFE}
}
```