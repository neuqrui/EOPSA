#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semantic rubrics for rubric-guided token classification (offline rule extraction).

Runtime taxonomy (``filters_qwen_eopsa.py``):
  pivot / intent / risk / function / same(Consistent) / other

LLM batch prompt labels disagreeing top-K pairs into four extractable categories.
Surface-equal pairs are handled by a rule upstream and are not shown to the LLM.
"""

from __future__ import annotations

from typing import Any

CATEGORY_ALIASES = {
    "pivot": "pivot",
    "intent": "intent",
    "risk": "risk",
    "function": "function",
    "consistent": "consistent",
    "same": "consistent",
    "other": "other",
}

EXTRACTABLE_CATEGORIES = frozenset({"pivot", "intent", "risk", "function"})

RUBRIC_TEXT = """
## Task
You classify high-KL (student_top1, teacher_top1) token pairs for offline rule extraction.
The student saw only the user query; the teacher also saw privileged safety guidance.
Use the query and student response for context.

For each listed pair, assign **at most one** extractable category:
  pivot | intent | risk | function

If a pair does not belong to any of the four, use category=null.

Priority when ambiguous: pivot > risk > intent > function.

---
## Pivot
Structural transition from compliance/planning to refusal or safety pivot.

When category=pivot, return in **pivot_tokens** only the token(s) in this pair that are
the discourse / structural pivot (the turning cue). If both sides are pivot cues, return both.
If only one side is the pivot cue, return only that side — do not include the other token.

Typical pivot cues: but, however, wait, which, maybe, yet, still, first, let, also, so, okay, alright …

Examples: "First"→"But" → pivot_tokens=["First","But"] (or just ["But"] if only teacher turns);
"Let"→"The" is usually NOT pivot (The is not a pivot cue).

---
## Intent
**stu→tea** only. Student frames **doing the task**; teacher reframes as **safety check**
(evaluate user intent, recognize risk, consider refusal/avoidance).

Teacher side must carry safety-evaluation meaning. Pronoun / opener / person swaps alone
are never intent.

INTENT_STU (task framing): understand, key, context, requirements, wants, start, goal, use, make, comply …
INTENT_TEA (safety-check framing): recognize, core, user, check, consider, remember, avoid, refuse, acknowledge …

Positive: understand→recognize, start→consider, use→avoid, key→core, make→check, wants→asking

**NOT intent** (category=null; or function/pivot if those fit):
  The→I, Maybe→I, they→I, I→they — first-person opener / person switch only
  deixis-only change (I/you/they/the/this) with no safety-check cue on the teacher side
  any pair that does not move toward checking, recognizing, or refusing the harmful request

---
## Risk
Explicit safety, harm, or refusal vocabulary.

When category=risk, return in **risk_tokens** only the token(s) in this pair that are
risk-related. If both student_top1 and teacher_top1 are risk-related, return both.
If only one side is risk-related, return only that side — do not include the other token.

Examples: refuse, harmful, illegal, safety, ethical, violence …

---
## Function
Pure grammatical glue with **no** safety / intent / pivot role: punctuation, prepositions,
articles, conjunctions, auxiliaries, and similar stop words.

When category=function, return in **function_tokens** only those glue token(s) from the pair.
Do not include task/content words on the other side.

Focus on:
  punctuation: , . : ; ! ? 's n't …
  prepositions: to, of, in, on, for, with, at, from, by, as …
  articles / determiners: a, an, the, this, that …
  conjunctions / auxiliaries: and, or, is, are, be, it …

Not function: discourse pivots (but/however/wait/yet/still/which/maybe) — use **pivot**;
intent verbs (understand/check); risk words (safety/refuse).
Never put pivot cues in function_tokens even if they look like conjunctions.
""".strip()


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_sample_batch_prompt(
    query: str,
    student_response: str,
    pairs: list[dict[str, Any]],
    *,
    max_query_chars: int = 1200,
    max_response_chars: int = 2400,
) -> str:
    llm_pairs = [p for p in pairs if p.get("pre_category") != "consistent"]
    pair_lines: list[str] = []
    for p in llm_pairs:
        rank = p.get("rank_in_sample", p.get("rank", "?"))
        stu = p.get("student_top1", "")
        tea = p.get("teacher_top1", "")
        fkl = p.get("forward_kl")
        pair_lines.append(
            f"  rank={rank}  student_top1={stu!r}  teacher_top1={tea!r}  "
            f"forward_kl={fkl:.4f}"
        )

    n_to_label = len(llm_pairs)
    return f"""{RUBRIC_TEXT}

---
## Input

User query:
{_truncate(query, max_query_chars)!r}

Student response (truncated):
{_truncate(student_response, max_response_chars)!r}

Top-{n_to_label} high-KL token pairs (student_top1 vs teacher_top1):
{chr(10).join(pair_lines)}

---
## Output

Return ONE JSON object (no markdown fence). Label exactly the {n_to_label} pairs listed above.
Use rank to match input.

Extraction mapping:
  pivot   → **pivot_tokens**: only discourse/structural pivot cue(s) from this pair
            (student-side → PIVOT_STU, teacher-side → PIVOT_TEA)
  intent  → student_top1 → INTENT_STU, teacher_top1 → INTENT_TEA; (student, teacher) → _INTENT_PAIRS
            ONLY if stu→tea is a real safety-evaluation framing shift (not pronoun/person opener swaps)
  risk    → **risk_tokens**: only risk-related token(s) from this pair (one or both sides)
  function→ **function_tokens**: only function/stop token(s) from this pair

Schema:
{{
  "pair_labels": [
    {{
      "rank": 1,
      "category": "pivot|intent|risk|function|null",
      "pivot_tokens": [],
      "risk_tokens": [],
      "function_tokens": []
    }},
    ...
  ]
}}

Rules:
- One entry per listed rank only.
- category=null → all token lists must be [].
- category=pivot → pivot_tokens must list only the pivot/transition cue(s) from the pair; omit non-pivot tokens.
- category=risk → risk_tokens must list only risk-related token(s) from the pair; omit non-risk tokens.
- category=function → function_tokens non-empty; list only function/stop token(s), omit content words.
- category=intent → leave pivot_tokens, risk_tokens and function_tokens as [];
  label intent ONLY for stu→tea safety-evaluation framing (e.g. understand→recognize),
  NOT for The→I / Maybe→I / they→I style pronoun or person switches (use null).
- Unused token lists for a category must be [].
"""


def default_pair_judgment(category: str = "other") -> dict[str, Any]:
    return {
        "category": category,
        "confidence": 1.0 if category == "consistent" else 0.0,
        "pivot_tokens": [],
        "risk_tokens": [],
        "function_tokens": [],
        "rationale": "",
    }


# ---------------------------------------------------------------------------
# Global lexicon audit (ONE LLM call after aggregation)
# ---------------------------------------------------------------------------

LEXICON_AUDIT_BACKGROUND = """
## Problem background

We distill a student LLM toward a teacher that sees privileged safety guidance
(the student only sees the user query). On high forward-KL positions we collect
(student_top1, teacher_top1) pairs and aggregate them into **family-specific,
query-universal** lexicons used at training time as a token filter:

  PIVOT_STU / PIVOT_TEA   — discourse / structural pivot cues
  INTENT_STU / INTENT_TEA — task framing (stu) vs safety-check framing (tea)
  _INTENT_PAIRS           — concrete stu→tea intent shifts
  RISK                    — explicit harm / safety / refusal words
  FUNC                    — pure grammatical glue (no safety role)

These sets must be **reusable across diverse harmful queries**. Tokens or pairs
that are query-specific content, mislabeled relative to the category rubric, or
generic noise that does not serve safety distillation should be **dropped**.

Category reminders (drop if a candidate violates these):
- Pivot: only structural turn cues (but/however/wait/which/first/let/…), not
  arbitrary content words or bare pronouns without pivot force.
- Intent: stu→tea must move from *doing the task* to *checking / recognizing /
  refusing the harmful request*. Drop pronoun/person opener swaps
  (The→I, Maybe→I, …), deixis-only pairs, and tokens that never carry that shift.
- Risk: only explicit safety/harm/refusal vocabulary.
- Function: punctuation / prepositions / articles / auxiliaries with **no**
  pivot, intent, or risk role. Never keep but/however/wait as FUNC; never keep
  task verbs or risk words as FUNC.
""".strip()


def build_lexicon_audit_prompt(rules: dict[str, Any]) -> str:
    """One-shot prompt: audit aggregated token sets + intent pairs; return drops."""

    def _tok_block(name: str) -> str:
        items = rules.get(name) or []
        return f"{name} ({len(items)}):\n  " + (", ".join(repr(x) for x in items) if items else "(empty)")

    pairs = rules.get("_INTENT_PAIRS") or []
    pair_lines = "\n".join(f"  ({a!r}, {b!r})" for a, b in pairs) if pairs else "  (empty)"

    return f"""{LEXICON_AUDIT_BACKGROUND}

---
## Candidate lexicons to audit

{_tok_block("PIVOT_STU")}

{_tok_block("PIVOT_TEA")}

{_tok_block("INTENT_STU")}

{_tok_block("INTENT_TEA")}

_INTENT_PAIRS ({len(pairs)}):
{pair_lines}

{_tok_block("RISK")}

{_tok_block("FUNC")}

---
## Output

Return ONE JSON object (no markdown fence). List **only items to DROP**
(leave keep lists implicit = candidate minus drop). Every dropped token/pair
must appear exactly as in the candidates above (same spelling / pairing).

Schema:
{{
  "drop": {{
    "PIVOT_STU": [],
    "PIVOT_TEA": [],
    "INTENT_STU": [],
    "INTENT_TEA": [],
    "RISK": [],
    "FUNC": [],
    "_INTENT_PAIRS": []
  }},
  "rationale_brief": "2-4 sentences on the main drop themes"
}}

Rules:
- Prefer dropping noisy / mislabeled / query-specific items; when unsure, KEEP.
- `_INTENT_PAIRS` entries must be two-element arrays, e.g. ["start", "consider"].
- Do not invent new tokens; only drop from the given candidates.
- Empty drop lists are fine if a category looks clean.
"""

