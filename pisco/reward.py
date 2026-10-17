"""
Answer-correctness scoring, shared by rollout filtering and data diagnostics.

Used for two things:
  * splitting finetuning data into teacher-correct / teacher-failed (S+ / S-),
  * scoring sampled rollouts in rejection-sampling finetuning (scripts/raft_rollout.py).

Both must use the *same* scorer, otherwise "the teacher failed here" and "the
rollout succeeded here" are measured on different scales.

Caveat on `containment`: it only fires when the gold answer appears verbatim in
the prediction. That is the right call when gold is a short span (nq_open,
triviaqa, hotpotqa, squad, freebase_qa, sciq) but it systematically under-counts
sources whose gold is a full extracted sentence (wikiqa, much of msmarco), where
a correct paraphrase scores 0. Use `f1` there, or report per-source rates so the
bias stays visible.
"""

import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")


def normalize_answer(s: Optional[str]) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace (SQuAD-style)."""
    if not s:
        return ""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", _ARTICLES.sub(" ", s.lower()))).strip()


def containment_em(pred: Optional[str], golds: Optional[Sequence[str]]) -> bool:
    """True if any normalized gold appears as a substring of the normalized prediction."""
    p = normalize_answer(pred)
    if not p:
        return False
    for g in golds or []:
        gn = normalize_answer(g)
        if gn and gn in p:
            return True
    return False


def token_f1(pred: Optional[str], golds: Optional[Sequence[str]]) -> float:
    """Best token-overlap F1 of the prediction against any gold."""
    p_toks = normalize_answer(pred).split()
    if not p_toks:
        return 0.0
    best = 0.0
    for g in golds or []:
        g_toks = normalize_answer(g).split()
        if not g_toks:
            continue
        common = Counter(p_toks) & Counter(g_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        precision = n_same / len(p_toks)
        recall = n_same / len(g_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def is_correct(
    pred: Optional[str],
    golds: Optional[Sequence[str]],
    mode: str = "containment",
    f1_threshold: float = 0.6,
) -> bool:
    """Binary correctness under the chosen scorer. This is the RL/RAFT reward."""
    if mode == "containment":
        return containment_em(pred, golds)
    if mode == "f1":
        return token_f1(pred, golds) >= f1_threshold
    if mode == "either":
        return containment_em(pred, golds) or token_f1(pred, golds) >= f1_threshold
    raise ValueError(f"unknown scoring mode: {mode!r}")


def teacher_is_correct(example: dict, **kwargs) -> bool:
    """Was the cached teacher answer (`mistral_label`) right, per the gold `labels`?"""
    return is_correct(example.get("mistral_label"), example.get("labels"), **kwargs)


_REFUSAL = re.compile(
    r"(not (mentioned|provided|specified|available|stated|shown|given|visible|clear|include|contain|explicitly)"
    r"|does not (contain|mention|provide|specify|show|include|state|say|give)"
    r"|do not (contain|mention|provide|specify|show|include)"
    r"|cannot|can't|unable to|no (information|mention|data|specific)|not possible|is unknown|insufficient)",
    re.I,
)


def is_refusal(text: Optional[str]) -> bool:
    """'The documents do not specify ...'-style non-answers. The Qwen3-8B Yes/No judge sometimes accepts them
    when they echo the gold's wording (mmrl smoke 173654), so they must never become training targets unless
    the gold itself is a non-answer (some Docmatix golds are)."""
    return bool(_REFUSAL.search(text or ""))


_SOURCE_RE = re.compile(r"^[a-zA-Z_]+")


def source_of(q_id: Optional[str]) -> str:
    """Dataset name from a q_id like 'msmarco49731' -> 'msmarco'."""
    m = _SOURCE_RE.match(q_id or "")
    return m.group(0) if m else "unknown"
