"""
Prompts of the multimodal stage-2 CHAT tasks, shared by the training collator
(pisco.mm_collator.MultimodalPretrainingCollator, tasks `desc` / `find`) and the offline teacher that writes the
description targets (scripts/teacher_describe_vllm.py, run in the vLLM env). Pure Python on purpose: the vLLM env
imports this file without torch/peft. The teacher's descriptions are only valid targets if both sides build the
SAME chat (system prompt, user template, question) -- change these strings and the descriptions must be regenerated.

Chat layout = FineTuningCollator's RAG template (same user template), so these tasks train exactly the decoder
interface used at finetuning / inference: "Document j:" + <MEM>s per image, then a question.
"""

# FineTuningCollator's default user template (pisco/collator.py); kept identical.
USER_PROMPT = "\n\nBackground:[documents]\n Question: [question]"

# `find`: short pointer answers, so the default brief-answer system prompt of finetuning.
FIND_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to extract relevant information from "
    "provided documents and to answer to questions as briefly as possible."
)

# `desc`: a long, faithful description is the point, so a dedicated system prompt.
DESC_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to describe the provided documents precisely and completely: "
    "transcribe the important text, give the numbers shown in charts and tables, and describe the layout, "
    "the objects and their positions. Write plain text without Markdown, in at most 200 words."
)


def desc_query(j: int) -> str:
    return f"Describe Document {j} in detail."


def find_query(snippet: str) -> str:
    return f'Which document contains this text: "{snippet}"? Answer with the document number.'


def find_answer(j: int) -> str:
    return f"Document {j}"


def user_content(documents: str, question: str) -> str:
    return USER_PROMPT.replace("[documents]", documents).replace("[question]", question)
