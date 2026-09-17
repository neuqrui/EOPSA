#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-classify step1 trained tokens with extracted_rules.json lexicons.

Uses the same decision order as ``filters_qwen_eopsa.py`` but loads
PIVOT_*/INTENT_*/RISK/FUNC/_INTENT_PAIRS from an ``extracted_rules.json``
produced by the rubric extraction pipeline. Does **not** modify
``filters_qwen_eopsa.py``.

Usage:
  python reclassify_with_extracted_rules.py \\
    --rules outputs/200all/extracted_rules.json \\
    --tokens /path/to/step1_trained_tokens.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RULES = SCRIPT_DIR / "outputs" / "200all" / "extracted_rules.json"
DEFAULT_TOKENS = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "opsd_safety_safechain"
    / "opsd_safety_qwen3-1.7b_safechain_topk_forward_kl_think_max128_wjwc_tsync0_dk512_dsr100_valguard_tftaxonomy_keep_drop_stutop1_fmtapi_overall"
    / "step1_trained_tokens.json"
)

TAXONOMY_KEEP = frozenset({"pivot", "intent", "risk_wo_same"})


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
    pairs_raw = rules.get("_INTENT_PAIRS") or []
    rules["_INTENT_PAIRS"] = {tuple(p) for p in pairs_raw if len(p) == 2}
    return rules


def classify_with_rules(stu: str, tea: str, rules: dict[str, Any]) -> str:
    """Same priority order as ``filters_qwen_eopsa._classify_token_qwen_eopsa_1p7b_overall``."""
    ns, nt = _norm_token(stu), _norm_token(tea)
    pivot_tea: set[str] = rules["PIVOT_TEA"]
    pivot_stu: set[str] = rules["PIVOT_STU"]
    intent_tea: set[str] = rules["INTENT_TEA"]
    intent_stu: set[str] = rules["INTENT_STU"]
    risk: set[str] = rules["RISK"]
    func: set[str] = rules["FUNC"]
    intent_pairs: set[tuple[str, str]] = rules["_INTENT_PAIRS"]

    # 1) pivot
    if nt in pivot_tea and ns != nt:
        return "pivot"
    if ns in pivot_stu and nt in pivot_tea:
        return "pivot"
    if ns in {"i", "the", "let", "they", "so", "first", "wait", "also"} and nt in pivot_tea | {"the", "that"}:
        if nt in pivot_tea or (ns == "i" and nt in {"the", "that"}):
            return "pivot"

    # 2) risk (split)
    if ns in risk or nt in risk:
        return "risk_lexicon" if ns == nt else "risk_wo_same"

    # 3) same surface
    if ns == nt:
        return "same"

    # 4) intent
    if (ns, nt) in intent_pairs:
        return "intent"
    if nt in intent_tea or ns in intent_stu:
        return "intent"

    # 5) function
    if ns in func and nt in func:
        return "function"
    if re.fullmatch(r"[\W_]+", (stu or "").strip()) or re.fullmatch(r"[\W_]+", (tea or "").strip()):
        return "function"

    return "other"


def classify_eopsa_ref(stu: str, tea: str) -> str:
    """Reference labels from ``filters_qwen_eopsa.py`` (unchanged)."""
    from token_filter.filters_qwen_eopsa import classify_token_qwen_eopsa_1p7b_overall

    return classify_token_qwen_eopsa_1p7b_overall(stu, tea)


def _print_count_table(
    title: str,
    counters: dict[str, Counter[str]],
    *,
    order: list[str] | None = None,
) -> None:
    default_order = [
        "pivot", "intent", "risk_wo_same", "risk_lexicon",
        "same", "function", "other",
    ]
    order = order or default_order
    names = list(counters)
    all_labels = [lab for lab in order if any(counters[n][lab] for n in names)]
    all_labels += sorted(
        {lab for c in counters.values() for lab in c if lab not in all_labels}
    )

    col_w = max(10, max(len(n) for n in names) + 2)
    header = f"{'类别':<16}" + "".join(f"{n:>{col_w}}" for n in names)
    if len(names) == 2:
        header += f"{'差值':>10}"
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for lab in all_labels:
        vals = [counters[n][lab] for n in names]
        line = f"{lab:<16}" + "".join(f"{v:>{col_w}}" for v in vals)
        if len(names) == 2:
            line += f"{vals[1] - vals[0]:>+10}"
        print(line)
    print("-" * len(header))
    totals = [sum(counters[n].values()) for n in names]
    line = f"{'合计':<16}" + "".join(f"{v:>{col_w}}" for v in totals)
    if len(names) == 2:
        line += f"{totals[1] - totals[0]:>+10}"
    print(line)

    keep = {"pivot", "intent", "risk_wo_same"}
    if any(counters[n][k] for n in names for k in keep):
        keep_vals = [sum(counters[n][k] for k in keep) for n in names]
        line = f"{'keep合计':<16}" + "".join(f"{v:>{col_w}}" for v in keep_vals)
        if len(names) == 2:
            line += f"{keep_vals[1] - keep_vals[0]:>+10}"
        print(line)


def _confusion(
    rows: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    mat: dict[str, Counter[str]] = defaultdict(Counter)
    for old, new in rows:
        mat[old][new] += 1
    return {k: dict(v) for k, v in sorted(mat.items())}


def _print_confusion(title: str, mat: dict[str, dict[str, int]]) -> None:
    print(f"\n=== {title} ===")
    all_cols = sorted({c for row in mat.values() for c in row})
    label = "old\\new"
    header = f"{label:>16}" + "".join(f"{c:>14}" for c in all_cols)
    print(header)
    for old in sorted(mat):
        row = mat[old]
        print(f"{old:>16}" + "".join(f"{row.get(c, 0):>14}" for c in all_cols))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output with per-token re-labels",
    )
    args = parser.parse_args()

    safety_rl = SCRIPT_DIR.parents[1]
    if str(safety_rl) not in sys.path:
        sys.path.insert(0, str(safety_rl))

    rules = load_rules(args.rules)
    with open(args.tokens, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tokens = payload["tokens"]
    enriched: list[dict[str, Any]] = []
    vs_train: list[tuple[str, str]] = []
    vs_eopsa: list[tuple[str, str]] = []
    vs_eopsa_file: list[tuple[str, str]] = []

    for tok in tokens:
        stu = tok.get("stu_top1_str") or tok.get("y_str") or ""
        tea = tok.get("tea_top1_str") or tok.get("y_str") or ""
        cat_train = tok.get("category", "")
        cat_eopsa_file = tok.get("category_eopsa_offline", "")
        cat_eopsa_ref = classify_eopsa_ref(stu, tea)
        cat_extracted = classify_with_rules(stu, tea, rules)

        row = {
            **tok,
            "category_extracted": cat_extracted,
            "category_eopsa_ref": cat_eopsa_ref,
            "in_keep_extracted": cat_extracted in TAXONOMY_KEEP,
            "changed_vs_train": cat_extracted != cat_train,
            "changed_vs_eopsa_file": cat_extracted != cat_eopsa_file,
            "changed_vs_eopsa_ref": cat_extracted != cat_eopsa_ref,
        }
        enriched.append(row)
        vs_train.append((cat_train, cat_extracted))
        vs_eopsa_file.append((cat_eopsa_file, cat_extracted))
        vs_eopsa.append((cat_eopsa_ref, cat_extracted))

    n = len(tokens)
    print(f"Rules:   {args.rules}")
    print(f"Tokens:  {args.tokens}  (n={n})")
    print(f"Lexicon sizes: PIVOT_STU={len(rules['PIVOT_STU'])} PIVOT_TEA={len(rules['PIVOT_TEA'])} "
          f"INTENT_STU={len(rules['INTENT_STU'])} INTENT_TEA={len(rules['INTENT_TEA'])} "
          f"RISK={len(rules['RISK'])} FUNC={len(rules['FUNC'])} pairs={len(rules['_INTENT_PAIRS'])}")

    count_train = Counter(t["category"] for t in tokens)
    count_eopsa_file = Counter(t.get("category_eopsa_offline") for t in tokens)
    count_eopsa_ref = Counter(r["category_eopsa_ref"] for r in enriched)
    count_extracted = Counter(r["category_extracted"] for r in enriched)

    _print_count_table(
        "各类别数量：eopsa (filters_qwen_eopsa) vs extracted (extracted_rules)",
        {
            "eopsa": count_eopsa_ref,
            "extracted": count_extracted,
        },
    )
    _print_count_table(
        "各类别数量：train (实际训练 category) vs extracted",
        {
            "train": count_train,
            "extracted": count_extracted,
        },
    )
    _print_count_table(
        "各类别数量：train vs eopsa",
        {
            "train": count_train,
            "eopsa": count_eopsa_ref,
        },
    )

    print("\n--- (file 内 category_eopsa_offline 与 live eopsa_ref 一致: "
          f"{count_eopsa_file == count_eopsa_ref}) ---")

    n_chg_train = sum(1 for a, b in vs_train if a != b)
    n_chg_eopsa_file = sum(1 for a, b in vs_eopsa_file if a != b)
    n_chg_eopsa_ref = sum(1 for a, b in vs_eopsa if a != b)
    print(f"\nChanged labels: vs train={n_chg_train}/{n}  "
          f"vs eopsa_offline={n_chg_eopsa_file}/{n}  vs eopsa_ref={n_chg_eopsa_ref}/{n}")

    _print_confusion("extracted vs train (category)", _confusion(vs_train))
    _print_confusion("extracted vs eopsa_offline (file field)", _confusion(vs_eopsa_file))
    _print_confusion("extracted vs eopsa_ref (filters_qwen_eopsa live)", _confusion(vs_eopsa))

    # Examples of meaningful changes vs eopsa_ref
    print("\n=== sample changes vs eopsa_ref (first 20) ===")
    shown = 0
    for r in enriched:
        if r["category_extracted"] == r["category_eopsa_ref"]:
            continue
        print(
            f"  {r['stu_top1_str']!r} -> {r['tea_top1_str']!r}  "
            f"eopsa={r['category_eopsa_ref']} -> extracted={r['category_extracted']}  "
            f"train={r.get('category')}"
        )
        shown += 1
        if shown >= 20:
            break

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {
            "rules_path": str(args.rules),
            "tokens_path": str(args.tokens),
            "n_tokens": n,
            "summary": {
                "changed_vs_train": n_chg_train,
                "changed_vs_eopsa_offline": n_chg_eopsa_file,
                "changed_vs_eopsa_ref": n_chg_eopsa_ref,
                "counts_extracted": dict(Counter(r["category_extracted"] for r in enriched)),
                "counts_train": dict(Counter(t["category"] for t in tokens)),
                "counts_eopsa_offline": dict(Counter(t.get("category_eopsa_offline") for t in tokens)),
                "counts_eopsa_ref": dict(Counter(r["category_eopsa_ref"] for r in enriched)),
            },
            "tokens": enriched,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
