"""
PISCO usage example for training
"""

import torch
from pisco.model import PISCO
from pisco.collator import FineTuningCollator


if __name__ == "__main__":

    # The PISCO model, containing both compressor and decoder:
    # It's an HF model, also a nn.Module.
    pisco = PISCO.from_pretrained(
        "/beegfs/scratch/user/mlouis/calmar/pisco/results/finetuning/Llama_1B_8B_ae_2048_ft/model",
        load_decoder=True,
    ).cuda()

    # This particular model has a llama 1B as compressor (fully FT) and Llama 8B as decoder (LoRA)
    print(pisco)

    # Collator object, convenient to convert 'docs' + 'query' into expected pisco inputs (ids and mask for both compressor and decoder)
    collator = FineTuningCollator(
        pisco.compressor_tokenizer,
        pisco.decoder_tokenizer,
        pisco.compr_rate,
        compressor_max_length=512,  # This model was trained with compressor length up to 512. To handle more, you need to chunk your docs.
        decoder_max_length=2048,  # This decoder was trained with contexts up to 2048 (including compressed reps). Quite safe to increase I think.
    )

    query = "What is the capital of France ?"
    docs = [
        "The capital of France is Paris.",
        "Paris is nice in the summer",
        "I love Grenoble",
    ]
    label = "The capital of France is Paris"

    # The collator formats and produces:
    # - compressor_input_ids/compressor_attention_mask for every document, with postpended 'MEM' token (sometimes documents can be chunked if too long and treated as multiple docs)
    # - decoder_input_ids/decoder_attention_mask: containing the prompt (hidden in the collator class) and placeholders which PISCO will replace, internally, with the computed docs embeddings.
    # - labels. Same shape as decoder_input_ids. CAUTION: the labels are masked (-100) except the last 'turn' of conversation. TODO check for your usage.
    batch = collator.torch_call(
        [
            {
                "docs": docs,
                "query": query,
                "mistral_label": label,  # To 'generate', just leave this empty "" and call .generate instead of forward.
            }
        ]
    )

    device = next(pisco.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    print(batch)

    out = pisco.forward(**batch)

    # dict containing 'loss' and 'logits'
    # It's safe to call 'backward' here
    print(out)

    out["loss"].backward()
