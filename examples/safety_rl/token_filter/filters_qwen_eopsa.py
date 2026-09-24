#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EOPSA Qwen taxonomy (``TOKEN_FILTER_CLASSIFIER=qwen_eopsa``).

Default Selective Distillation classifier. Categories:
  pivot / intent / risk_lexicon / risk_wo_same / function / same / other

Keep set K: ``pivot``, ``intent``, ``risk_wo_same``.
Same-surface risk stays ``risk_lexicon`` and is not trained by default.
"""

from __future__ import annotations

import re

PIVOT_TEA = {"but", "however", "wait", "which", "maybe", "yet", "still"}
PIVOT_STU = {
    "the", "i", "let", "they", "first", "also", "so", "wait", "my", "now",
    "okay", "alright",
}
INTENT_TEA = {
    "recognize", "recognizing", "acknowledge", "acknowledging", "core", "request",
    "intent", "asking", "user", "check", "consider", "considering",
}
INTENT_STU = {
    "understand", "outlining", "requirements", "key", "context", "example",
    "wants", "working", "needs", "goal", "main", "start", "think", "presented",
    "points",
}
RISK = {
    "harmful", "dangerous", "illegal", "unethical", "crime", "harm", "refuse",
    "reject", "avoid", "serious", "safety", "safe", "guidelines", "sensitive",
    "ethical", "risk", "risks", "threat", "violence", "weapon", "bomb",
}
FUNC = {
    ",", ".", ":", ";", "!", "?", "a", "an", "the", "to", "of", "in", "on", "for",
    "and", "or", "as", "by", "is", "are", "be", "that", "this", "with", "at", "from",
    "it", "to", "'s", "n't",
}

TAXONOMY_KEEP_CATEGORIES = {"pivot", "intent", "risk_wo_same"}
TAXONOMY_DROP_CATEGORIES = {"same", "function", "risk_lexicon", "other"}

_INTENT_PAIRS = {
    ("understand", "recognize"), ("key", "core"), ("key", "user"),
    ("context", "core"), ("requirements", "core"), ("example", "core"),
    ("wants", "is"), ("working", "asking"), ("outlining", "recognizing"),
    ("likely", "asking"), ("needs", "wants"), ("use", "avoid"),
    ("main", "user"), ("main", "core"), ("make", "check"), ("make", "be"),
    ("start", "consider"), ("start", "check"), ("should", "remember"),
}


def _norm_token(s: str) -> str:
    s = (s or "").strip().lower().replace("’", "'")
    if re.fullmatch(r"[\W_]+", s or ""):
        return s
    s2 = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s2 if s2 else s


def _classify_token_qwen_eopsa_1p7b_overall(stu: str, tea: str) -> str:
    """EOPSA Qwen taxonomy label for (student, teacher) top1."""
    ns, nt = _norm_token(stu), _norm_token(tea)

    # 1) pivot
    if nt in PIVOT_TEA and ns != nt:
        return "pivot"
    if ns in PIVOT_STU and nt in PIVOT_TEA:
        return "pivot"
    if ns in {"i", "the", "let", "they", "so", "first", "wait", "also"} and nt in PIVOT_TEA | {"the", "that"}:
        if nt in PIVOT_TEA or (ns == "i" and nt in {"the", "that"}):
            return "pivot"

    # 2) risk (split): same-surface vs disagree
    if ns in RISK or nt in RISK:
        return "risk_lexicon" if ns == nt else "risk_wo_same"

    # 3) same surface (after risk so same-surface risk stays risk_*)
    if ns == nt:
        return "same"

    # 4) intent
    if (ns, nt) in _INTENT_PAIRS:
        return "intent"
    if nt in INTENT_TEA or ns in INTENT_STU:
        return "intent"

    # 5) function (both sides FUNC, or either side pure punct)
    if ns in FUNC and nt in FUNC:
        return "function"
    if re.fullmatch(r"[\W_]+", (stu or "").strip()) or re.fullmatch(r"[\W_]+", (tea or "").strip()):
        return "function"

    return "other"


classify_token_qwen_eopsa_1p7b_overall = _classify_token_qwen_eopsa_1p7b_overall
classify_token_qwen_eopsa = _classify_token_qwen_eopsa_1p7b_overall
classify_token_qwen = _classify_token_qwen_eopsa_1p7b_overall
classify = _classify_token_qwen_eopsa_1p7b_overall
