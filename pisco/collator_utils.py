"""
PISCO
Copyright (c) 2026-present NAVER Corp.
All Rights Reserved.

Utilities for PISCO data preparation (chunking, string manips)
"""

import torch
import random
from typing import List


def print_collated_sample(batch, collator):
    """
    a printing utility
    """
    n_samples = batch["decoder_input_ids"].size(1)
    n_to_print = min(n_samples, 2)
    print("---compressor inputs----")
    print(
        collator.compressor_tokenizer.batch_decode(
            batch["compressor_input_ids"][:n_to_print, :], skip_special_tokens=True
        )
    )
    print("---compressor attention_mask----")
    print(batch["compressor_attention_mask"][:n_to_print, :])

    print("---- decoder inputs -----")
    print(
        collator.decoder_tokenizer.batch_decode(
            batch["decoder_input_ids"][:n_to_print, :], skip_special_tokens=True
        )
    )
    labels_to_print = batch["labels"][:n_to_print, :]
    print("---- labels -----")
    print(
        collator.decoder_tokenizer.batch_decode(
            labels_to_print.masked_fill(
                labels_to_print == -100, collator.decoder_tokenizer.pad_token_id
            ),
            skip_special_tokens=True,
        )
    )

    if "query_input_ids" in batch:
        print("Query inputs")
        print(
            collator.compressor_tokenizer.batch_decode(
                batch["query_input_ids"][:n_to_print, :], skip_special_tokens=True
            )
        )

    if "doc_input_ids" in batch:
        print("Document inputs")
        print(
            collator.compressor_tokenizer.batch_decode(
                batch["doc_input_ids"][:n_to_print, :], skip_special_tokens=True
            )
        )

    if "scores" in batch:
        print(batch["scores"])


def mask_before_mem(labels, mem_token_id):
    """
    Set to -100 any token preceeding any 'mem' token
    Used to build labels.
    """
    # Now we mask every token before last MEM in labels
    mem_mask = labels == mem_token_id  # (B, L)
    positions = torch.arange(labels.size(1), device=labels.device).expand_as(labels)
    positions = positions.masked_fill(~mem_mask, -1)
    last_mem_index = positions.max(dim=1).values  # (B,)

    # build mask: True for tokens to ignore
    ignore_mask = torch.zeros(labels.shape, device=positions.device, dtype=torch.bool)
    ignore_mask = torch.arange(
        labels.size(1), device=labels.device
    ) <= last_mem_index.unsqueeze(1)

    # apply loss mask
    labels = labels.masked_fill(ignore_mask, -100)

    return labels


def chunk_list(
    data: List[int], chunk_length: int, chunk_overlap: int
) -> List[List[int]]:
    """
    Split a list of integers into overlapping chunks.
    :data: List of integers to chunk.
    :max_length: Maximum size of each chunk.
    :chunk_overlap: Number of overlapping elements between chunks.
    """
    if chunk_overlap >= chunk_length:
        raise ValueError("chunk_overlap must be smaller than max_length")

    chunks = []
    step = chunk_length - chunk_overlap

    for start in range(0, len(data), step):
        chunk = data[start : start + chunk_length]
        if chunk:
            chunks.append(chunk)

    return chunks


def randomly_chunk(lst, k, max_size):
    """
    chunks the list lst into k continuous pieces such that
    each piece length does not exceed max_size
    Used for pretraining, to make pisco robust to sizes and long docs.
    """
    n = len(lst)

    if n < k:
        raise ValueError("List too small for k non-empty pieces")
    if n > k * max_size:
        raise ValueError("Cannot split: max_size too small")

    # Start with minimum 1 per piece
    sizes = [1] * k
    remaining = n - k

    # Distribute remaining randomly, respecting max_size
    indices = list(range(k))
    while remaining > 0:
        i = random.choice(indices)
        if sizes[i] < max_size:
            sizes[i] += 1
            remaining -= 1
        else:
            indices.remove(i)

    # Build the slices
    result = []
    start = 0
    for size in sizes:
        result.append(lst[start : start + size])
        start += size

    return result


def add_memory_tokens_to_inputs(
    input_ids: list,
    tokenizer,
    compr_rate: int,
):
    """
    Appends the proportional number of mem to each entry
    Also appends the retrieval token if it exists in the tokenizer.
    """
    n_mems = [len(elt) // compr_rate + 1 for elt in input_ids]

    out = [
        elt + [tokenizer.mem_token_id] * n_mem
        for (elt, n_mem) in zip(input_ids, n_mems)
    ]

    retrieval_token_id = getattr(tokenizer, "retrieval_token_id", None)
    if retrieval_token_id is not None:
        out = [elt + [retrieval_token_id] for elt in out]

    return out, n_mems


def chunk_random_no_tiny_tail(
    text_ids: List[int], compressor_max_length: int, min_ratio: float = 0.8
) -> List[List[int]]:
    """
    chunk into pieces of at most compressor_max_length and almost always at min min_ratio + compressor_max_length
    ensures the last piece is not too small (but penultimate can be small)
    """
    n = len(text_ids)
    out = []
    i = 0

    min_len = max(1, int(compressor_max_length * min_ratio))
    max_len = compressor_max_length

    while i < n:
        remaining = n - i
        if remaining <= max_len:
            out.append(text_ids[i:])
            break

        # if we take max_len now, what remains?
        # if remainder would be too small, shorten this chunk so tail is >= min_len
        remainder_after_max = remaining - max_len
        if 0 < remainder_after_max < min_len:
            k = max_len - (min_len - remainder_after_max)
        else:
            k = random.randint(min_len, max_len)

        out.append(text_ids[i : i + k])
        i += k

    return out


if __name__ == "__main__":
    print(chunk_random_no_tiny_tail(list(range(20)), 50, 0.9))
