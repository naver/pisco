"""
PISCO
Copyright (c) 2026-present NAVER Corp.
All Rights Reserved.

This file contains metrics for surface-form analysis of texts
It's typically used as a quick proxy for decoder generation quality
"""

import string
import numpy as np
from rouge import Rouge
from collections import Counter


def normalize(s: str) -> str:
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def em_single(prediction, ground_truth):
    return float(normalize(prediction) == normalize(ground_truth))


def exact_match_score(predictions, references):
    return np.mean(
        [
            em_single(prediction, ground_truth)
            for ground_truth, prediction in zip(references, predictions)
        ]
    )


def f1_single(prediction, ground_truth, tokenfun=lambda x: x.split()):
    prediction_tokens = tokenfun(prediction)
    ground_truth_tokens = tokenfun(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def match_single(prediction, ground_truth):  # TODO: fix this
    return float(normalize(ground_truth) in normalize(prediction))


def match_score(predictions, references):
    return np.mean(
        [
            match_single(prediction, gt)
            for gt, prediction in zip(references, predictions)
        ]
    )


def rouge_wrapper(rouge, prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def f1_score(predictions, references, tokenfun=lambda x: x.split()):
    f1, precision, recall = list(), list(), list()
    for ground_truths, prediction in zip(references, predictions):
        f1_, precision_, recall_ = [
            max(values)
            for values in zip(
                *[f1_single(prediction, gt, tokenfun) for gt in ground_truths]
            )
        ]
        f1.append(f1_)
        precision.append(precision_)
        recall.append(recall_)
    return np.mean(f1), np.mean(precision), np.mean(recall)


def compute_rouge_scores(predictions, references):
    rouge = Rouge()
    rouge1, rouge2, rougel = [], [], []
    for ground_truths, prediction in zip(references, predictions):
        rouge1_, rouge2_, rougel_ = rouge_wrapper(rouge, prediction, ground_truths)
        rouge1.append(rouge1_)
        rouge2.append(rouge2_)
        rougel.append(rougel_)
    return {
        "Rouge-1": np.mean(rouge1),
        "Rouge-2": np.mean(rouge2),
        "Rouge-L": np.mean(rougel),
    }


def hard_metrics(predictions: list[str], references: list[str]) -> dict:
    metrics = compute_rouge_scores(predictions, references)
    _, precision, recall = f1_score(predictions=predictions, references=references)
    metrics.update(
        {
            # 'f1': f1,
            "precision": precision,
            "recall": recall,
            # 'EM': exact_match_score(predictions=predictions, references=references),
            "M": match_score(predictions=predictions, references=references),
        }
    )
    return metrics
