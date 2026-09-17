#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rubric-extracted Qwen taxonomy for EOPSA Selective Distillation.

Official classifier (paper §Selective Distillation). Categories map to:
  Pivot / Intent / Risk(=risk_wo_same) / Function / Consistent(=same) / Other.

Default keep set K: ``pivot``, ``intent``, ``risk_wo_same``.
"""

from __future__ import annotations

import re

PIVOT_TEA = {
    "also", "but", "first", "however", "wait", "which",
}
PIVOT_STU = {
    "also", "but", "first", "for", "however", "i", "let", "maybe", "or", "so",
    "the", "they", "wait",
}
INTENT_TEA = {
    "acknowledge", "acknowledging", "address", "asking", "avoid", "check",
    "compliant", "confirm", "consider", "core", "create",
    "emphasize", "ensure", "explain", "focus", "guidance", "guidelines",
    "intent", "key", "make", "need", "positive", "provide", "recall",
    "recognize", "remember", "request", "require", "safety", "should",
    "suggest", "suggesting", "trying", "understand", "understanding",
    "user", "want", "wants",
}
INTENT_STU = {
    "a", "acknowledge", "actionable", "actually", "address", "also", "be",
    "clarify", "come", "consider", "context", "create", "describe",
    "emphasize", "establish", "ethical", "example", "examples", "explain",
    "explaining", "focus", "generate", "guide", "highlight", "include",
    "information", "just", "key", "looking", "main", "make", "maybe",
    "mention", "might", "original", "possible", "present", "provide", "recall",
    "recalling", "remember", "remind", "request", "requirements", "setting",
    "should", "start", "structure", "structured", "think", "thinking",
    "understand", "use", "user", "working",
}
RISK = {
    "avoid", "cannot", "compliant", "consequences", "ethical", "harm",
    "harmful", "illegal", "importance", "legal", "manipulation", "refuse",
    "risk", "risks", "safe", "safer", "safety", "sensitive",
}
FUNC = {
    '"', "(", ",", ".", ":", "a", "about", "address", "ai", "also", "an",
    "and", "are", "arguments", "as", "be", "bit", "but", "by", "can", "check",
    "clearly", "common", "each", "ensure", "first", "for", "from", "have",
    "his", "historian", "how", "however", "i", "if", "important", "in",
    "include", "information", "introduce", "is", "it", "just", "key", "let",
    "make", "maybe", "me", "mention", "mentioned", "might", "must", "need",
    "not", "now", "of", "on", "or", "paragraph", "possible", "query", "re",
    "realistic", "reports", "request", "response", "s", "should", "since",
    "so", "some", "specific", "start", "stereotypes", "structure", "that",
    "the", "there", "these", "they", "think", "this", "to", "too", "use",
    "user", "wants", "what", "while", "with", "without",
}

TAXONOMY_KEEP_CATEGORIES = {"pivot", "intent", "risk_wo_same"}
TAXONOMY_DROP_CATEGORIES = {"same", "function", "risk_lexicon", "other"}

_INTENT_PAIRS = {
    ("a", "asking"),
    ("acknowledge", "recognize"),
    ("also", "avoid"),
    ("be", "make"),
    ("be", "remember"),
    ("consider", "acknowledge"),
    ("consider", "avoid"),
    ("consider", "recall"),
    ("consider", "understand"),
    ("context", "core"),
    ("create", "make"),
    ("establish", "make"),
    ("include", "avoid"),
    ("include", "ensure"),
    ("information", "guidance"),
    ("key", "core"),
    ("looking", "trying"),
    ("main", "core"),
    ("make", "avoid"),
    ("make", "check"),
    ("make", "ensure"),
    ("make", "recognize"),
    ("make", "understand"),
    ("maybe", "avoid"),
    ("might", "wants"),
    ("original", "core"),
    ("possible", "core"),
    ("provide", "comply"),
    ("recall", "consider"),
    ("recalling", "understanding"),
    ("remind", "avoid"),
    ("request", "core"),
    ("setting", "core"),
    ("should", "need"),
    ("start", "avoid"),
    ("start", "check"),
    ("start", "consider"),
    ("start", "make"),
    ("start", "think"),
    ("think", "consider"),
    ("think", "ensure"),
    ("think", "make"),
    ("thinking", "understanding"),
    ("understand", "recognize"),
    ("use", "avoid"),
    ("user", "core"),
    ("user", "request"),
    ("user", "safety"),
    ("working", "asking"),
}


def _norm_token(s: str) -> str:
    s = (s or "").strip().lower().replace("’", "'")
    if re.fullmatch(r"[\W_]+", s or ""):
        return s
    s2 = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s2 if s2 else s


def _classify_token_qwen_rubric(stu: str, tea: str) -> str:
    """Rubric-extracted taxonomy label for (student, teacher) top1."""
    ns, nt = _norm_token(stu), _norm_token(tea)

    # Consistent / same-surface (includes same-surface risk → not in K)
    if ns == nt:
        return "same"

    # Pivot — teacher-side cue, student disagrees
    if nt in PIVOT_TEA and ns not in PIVOT_TEA:
        return "pivot"

    # Risk (paper) = risk_wo_same: risk lexicon + student ≠ teacher
    if ns in RISK or nt in RISK:
        return "risk_wo_same"

    # Intent — both sides in intent lexicons (AND)
    if nt in INTENT_TEA and ns in INTENT_STU:
        return "intent"

    # Function
    if ns in FUNC and nt in FUNC:
        return "function"
    if re.fullmatch(r"[\W_]+", (stu or "").strip()) or re.fullmatch(r"[\W_]+", (tea or "").strip()):
        return "function"

    return "other"


classify_token_qwen_rubric = _classify_token_qwen_rubric
classify_token_qwen = _classify_token_qwen_rubric
classify = _classify_token_qwen_rubric
