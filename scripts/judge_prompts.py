"""The judge prompts and verdict parsing shared by every judge script (judge_mm_docqa.py, judge_study.py).
Pure Python on purpose: the vLLM env imports it without torch.

  project  the project's Yes/No prompt (bergen's, judge_teacher.judge_prompt); verdict = the reply contains "yes".
  openeqa  OpenEQA's LLM-Match prompt (prompts/mmbench.txt, mmbench-extra.txt when the question has extra_answers,
           verbatim, MIT, copied to scripts/openeqa_prompts/): the judge marks 1-5; verdict = (mark - 1) / 4, so the
           mean verdict x 100 is LLM-Match. answers[0] is the reference answer, answers[1:] the extra answers. An
           empty prediction is judged as "I don't know." (the campaign_eqa rule).
  rubric   an explicit rubric; the judge answers {"verdict": "yes"|"no", "reason": "..."}. In the judge study of
           2026-10-20 (expQ/judge_study, 3730 items, 7 conditions) this prompt made the fewest errors on the
           objective anchors (Qwen3-8B: 0.3% false "no" on exact matches, 0.9% false "yes" on wrong numbers,
           against 1.3% / 5.0% for `project`), while every model ranking stayed the same.
"""
import os
import re


def project_prompt(question, gold_answers, prediction):
    """Verbatim from scripts/llm_judge.py (itself bergen's models/evaluators/openai.py)."""
    gold = " | ".join(str(a) for a in gold_answers) if isinstance(gold_answers, list) else str(gold_answers)
    return [
        {"role": "system", "content": "You are an evaluation tool. Just answer by {Yes} or {No}."},
        {"role": "user", "content": (
            "Here is a question, a golden answer and an AI-generated answer. Can you judge "
            "whether the AI-generated answer is correct according to the question and golden "
            "answer, simply answer {Yes} or {No}.\n"
            f" Question: {question}. \ngolden answer: {gold} \n Generated answer: {prediction}.\nResponse:"
        )},
    ]

RUBRIC_SYSTEM = (
    "You are a strict but fair grader of question answering. You will see a question, one or more reference "
    "answers, and a candidate answer. Decide whether the candidate answer is correct.\n"
    "Rules: the candidate is correct if it gives the same answer as any reference answer, in any wording, unit or "
    "number format; extra detail or a full sentence is fine. It is incorrect if it gives a different value, name, "
    "or choice, if it only partially answers (misses a required part), if it refuses or says it cannot tell, or if "
    "it contradicts the reference. Judge only the final answer, not the explanation.\n"
    'Reply with one line of JSON: {"verdict": "yes" or "no", "reason": "<at most 15 words>"}'
)

PROMPTS = ("project", "rubric", "openeqa")

_OPENEQA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openeqa_prompts")
_OPENEQA_TEMPLATES = {}


def openeqa_prompt(question, gold_answers, prediction):
    gold = list(gold_answers) if isinstance(gold_answers, list) else [gold_answers]
    answer, extra = str(gold[0]), [str(a) for a in gold[1:]]
    key = "mmbench-extra" if extra else "mmbench"
    if key not in _OPENEQA_TEMPLATES:
        _OPENEQA_TEMPLATES[key] = open(os.path.join(_OPENEQA_DIR, key + ".txt")).read()
    pred = (prediction or "").strip() or "I don't know."
    kw = dict(question=question, answer=answer, prediction=pred)
    if extra:
        kw["extra_answers"] = extra
    return [{"role": "user", "content": _OPENEQA_TEMPLATES[key].format(**kw)}]



def rubric_prompt(question, gold_answers, prediction):
    gold = gold_answers if isinstance(gold_answers, list) else [gold_answers]
    refs = "\n".join(f"- {a}" for a in gold)
    return [{"role": "system", "content": RUBRIC_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nReference answers:\n{refs}\nCandidate answer: {prediction}"}]


def build_prompt(kind, question, gold_answers, prediction):
    fn = {"project": project_prompt, "rubric": rubric_prompt, "openeqa": openeqa_prompt}[kind]
    return fn(question, gold_answers, prediction)


def parse_verdict(reply, kind):
    """1 / 0 (openeqa: (mark - 1) / 4 in [0, 1]), or None when the reply holds no verdict. A thinking model's text
    after </think> is used."""
    t = reply.split("</think>")[-1].strip().lower() if "</think>" in reply else reply.strip().lower()
    if kind == "openeqa":   # OpenEQA's parse rule: the first integer 1-5 of the reply
        for tok in re.findall(r"\d+", t):
            if 1 <= int(tok) <= 5:
                return (int(tok) - 1) / 4.0
        return None
    if kind == "rubric":
        m = re.search(r'"verdict"\s*:\s*"(yes|no)"', t)
        if m:
            return 1 if m.group(1) == "yes" else 0
    if "yes" in t:
        return 1          # the project rule: any "yes" counts
    if "no" in t:
        return 0
    return None
