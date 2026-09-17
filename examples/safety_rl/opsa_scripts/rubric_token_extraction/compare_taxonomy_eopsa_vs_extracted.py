#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare taxonomy counts: filters_qwen_eopsa vs extracted_rules (same rules, different lexicons).

Token source (default): step-1 ``stf_drop_logs`` union from tf-category ablation ckpts
(keep_pivot / keep_intent / keep_same / …). This yields step-1 training positions that
cover **all** taxonomy labels when re-classified. Pure no-token-filter ckpts do not
persist per-token step-1 logs (``token_filter=false`` → no stf_drop export).

Usage:
  python compare_taxonomy_eopsa_vs_extracted.py \\
    --rules outputs/200all/extracted_rules.json

  python compare_taxonomy_eopsa_vs_extracted.py \\
    --tokens-json /path/to/tokens.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SAFETY_RL = SCRIPT_DIR.parents[1]  # .../examples/safety_rl
REPO_ROOT = SCRIPT_DIR.parents[3]  # .../EOPSA
DEFAULT_RULES = SCRIPT_DIR / "outputs" / "200all" / "extracted_rules.json"

CATEGORY_ORDER = [
    "pivot", "intent", "risk_wo_same", "risk_lexicon",
    "same", "function", "other",
]


def _norm_token(s: str) -> str:
    s = (s or "").strip().lower().replace("\u2019", "'")
    if re.fullmatch(r"[\W_]+", s or ""):
        return s
    s2 = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s2 if s2 else s


def load_rules(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    for key in ("PIVOT_STU", "PIVOT_TEA", "INTENT_STU", "INTENT_TEA", "RISK", "FUNC"):
        rules[key] = set(rules.get(key) or [])
    rules["_INTENT_PAIRS"] = {tuple(p) for p in (rules.get("_INTENT_PAIRS") or []) if len(p) == 2}
    return rules


def classify_with_rules(stu: str, tea: str, rules: dict[str, Any]) -> str:
    """Extracted-lexicon classifier (stricter than ``filters_qwen_eopsa``).

    Differences vs eopsa:
      - pivot: only ``nt in PIVOT_TEA and ns != nt`` (no STU∩TEA / special-case branches)
      - intent lexicon hit: ``nt in INTENT_TEA and ns in INTENT_STU`` (AND, not OR)
    """
    ns, nt = _norm_token(stu), _norm_token(tea)
    pivot_tea: set[str] = rules["PIVOT_TEA"]
    intent_tea: set[str] = rules["INTENT_TEA"]
    intent_stu: set[str] = rules["INTENT_STU"]
    risk: set[str] = rules["RISK"]
    func: set[str] = rules["FUNC"]
    intent_pairs: set[tuple[str, str]] = rules["_INTENT_PAIRS"]

    # 1) pivot — tea-side pivot cue only, disagree required
    if nt in pivot_tea and ns != nt:
        return "pivot"

    # 2) risk
    if ns in risk or nt in risk:
        return "risk_lexicon" if ns == nt else "risk_wo_same"

    # 3) same surface
    if ns == nt:
        return "same"

    # 4) intent — exact pairs, or both sides in intent lexicons
    if (ns, nt) in intent_pairs:
        return "intent"
    if nt in intent_tea and ns in intent_stu:
        return "intent"

    # 5) function
    if ns in func and nt in func:
        return "function"
    if re.fullmatch(r"[\W_]+", (stu or "").strip()) or re.fullmatch(r"[\W_]+", (tea or "").strip()):
        return "function"

    return "other"


def classify_eopsa(stu: str, tea: str) -> str:
    from token_filter.filters_qwen_eopsa import classify_token_qwen_eopsa_1p7b_overall

    return classify_token_qwen_eopsa_1p7b_overall(stu, tea)


def load_tf_ablate_step1_union() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ckpt_root = REPO_ROOT / "checkpoints" / "opsd_safety_safechain"
    paths = sorted(
        ckpt_root.glob(
            "opsd_tfablate_*keep_*_20260819_154515/stf_drop_logs/step_000001.json"
        )
    )
    if not paths:
        raise FileNotFoundError(f"No tf-ablate step_000001.json under {ckpt_root}")

    by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    run_names: list[str] = []
    for p in paths:
        run_names.append(p.parent.parent.name)
        payload = json.loads(p.read_text(encoding="utf-8"))
        for tok in payload.get("tokens") or []:
            key = (tok.get("uid"), tok.get("pos"))
            by_key[key] = tok

    meta = {
        "source": "tf_ablate_step1_union",
        "step": 1,
        "n_runs": len(paths),
        "run_names": run_names,
        "n_tokens_unique": len(by_key),
        "note": (
            "Union of step-1 kept tokens from per-category tf ablation ckpts; "
            "re-classifying covers all taxonomy labels. "
            "Pure no-token-filter ckpts do not write stf_drop_logs."
        ),
    }
    return list(by_key.values()), meta


def load_tokens_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        tokens = payload
    elif isinstance(payload, dict) and "tokens" in payload:
        tokens = payload["tokens"]
    else:
        raise ValueError(f"Unsupported tokens json layout: {path}")
    return tokens, {"source": str(path), "n_tokens": len(tokens)}


def print_count_table(
    eopsa: Counter[str],
    extracted: Counter[str],
    *,
    title: str,
    n_tokens: int,
) -> None:
    labels = [c for c in CATEGORY_ORDER if eopsa[c] or extracted[c]]
    print(f"\n=== {title} (n={n_tokens}) ===")
    print(f"{'类别':<16} {'eopsa':>8} {'extracted':>10} {'差值':>8}")
    print("-" * 46)
    for lab in labels:
        a, b = eopsa[lab], extracted[lab]
        print(f"{lab:<16} {a:>8} {b:>10} {b - a:>+8}")
    print("-" * 46)
    print(f"{'合计':<16} {sum(eopsa.values()):>8} {sum(extracted.values()):>10} {sum(extracted.values()) - sum(eopsa.values()):>+8}")

    keep = {"pivot", "intent", "risk_wo_same"}
    ek = sum(eopsa[k] for k in keep)
    xk = sum(extracted[k] for k in keep)
    print(f"{'keep合计':<16} {ek:>8} {xk:>10} {xk - ek:>+8}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument(
        "--tokens-json",
        type=Path,
        default=None,
        help="Optional JSON with {tokens:[{stu_top1_str, tea_top1_str, ...}]}",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if str(SAFETY_RL) not in sys.path:
        sys.path.insert(0, str(SAFETY_RL))

    rules = load_rules(args.rules)
    if args.tokens_json:
        tokens, meta = load_tokens_json(args.tokens_json)
    else:
        tokens, meta = load_tf_ablate_step1_union()

    eopsa_c: Counter[str] = Counter()
    extracted_c: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for tok in tokens:
        stu = tok.get("stu_top1_str") or tok.get("student_top1") or ""
        tea = tok.get("tea_top1_str") or tok.get("teacher_top1") or ""
        ce = classify_eopsa(stu, tea)
        cx = classify_with_rules(stu, tea, rules)
        eopsa_c[ce] += 1
        extracted_c[cx] += 1
        rows.append({**tok, "category_eopsa": ce, "category_extracted": cx, "changed": ce != cx})

    print(f"Rules: {args.rules}")
    print(f"Token source: {json.dumps(meta, ensure_ascii=False, indent=2)}")
    print(
        f"Lexicon sizes — PIVOT_STU={len(rules['PIVOT_STU'])} PIVOT_TEA={len(rules['PIVOT_TEA'])} "
        f"INTENT_STU={len(rules['INTENT_STU'])} INTENT_TEA={len(rules['INTENT_TEA'])} "
        f"RISK={len(rules['RISK'])} FUNC={len(rules['FUNC'])} pairs={len(rules['_INTENT_PAIRS'])}"
    )

    n_changed = sum(1 for r in rows if r["changed"])
    print(f"Label changes (eopsa vs extracted): {n_changed}/{len(rows)}")

    print_count_table(eopsa_c, extracted_c, title="各类别数量对比", n_tokens=len(rows))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "meta": meta,
            "rules_path": str(args.rules),
            "counts_eopsa": dict(eopsa_c),
            "counts_extracted": dict(extracted_c),
            "n_changed": n_changed,
            "tokens": rows,
        }
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
