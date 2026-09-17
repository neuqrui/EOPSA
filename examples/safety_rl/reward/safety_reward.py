# -*- coding: utf-8 -*-
"""RLSD-style safety reward adapter."""

import asyncio
import os
import sys

_CUR = os.path.dirname(os.path.abspath(__file__))
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)

import async_safe_reward as A  # noqa: E402
import rule_safe_reward as R  # noqa: E402

REWARD_NAME = "safety_rlsd"
REWARD_TYPE = "batch"

SAFETY_TYPES = frozenset({"safety", "safety_data", "harmful", "custom_safety_dataset"})
BENIGN_TYPES = frozenset({"overreject", "benign", "harmless", "harmless_queries"})


def _data_source_from_type(data_type) -> str:
    dt = (str(data_type) if data_type is not None else "safety").strip().lower()
    if dt in BENIGN_TYPES:
        return "harmless_queries"
    return "custom_safety_dataset"


def _extract_query(inp: dict) -> str:
    q = inp.get("problem")
    if isinstance(q, str) and q.strip():
        return q
    gt = inp.get("ground_truth")
    if isinstance(gt, str):
        return gt
    if isinstance(gt, dict):
        return str(gt.get("original_query", ""))
    return ""


async def _format_score(response: str, format_mode: str, api_type: str, model_name: str):
    if format_mode == "off":
        return 1.0, {"format_mode": "off"}
    # rule / api / api_overall all use rule-based format scoring (tag count)
    return R.get_format_reward(response), {"format_mode": "rule"}


async def _score_one_rule(inp, data_source, response, fmt):
    """Pure rule-based scoring (no LLM API), used when format_mode == 'rule'."""
    if data_source == "custom_safety_dataset":
        refusal_r = R.get_refusal_scores(response)
        judge = float(refusal_r)
        branch = "safety"
    else:
        benign_r = R.get_benign_rule_score(response)
        judge = float(benign_r)
        branch = "benign"

    overall = judge * float(fmt)
    result = {
        "overall": overall,
        "accuracy": judge,
        "format": float(fmt),
        "think_safe": overall,
        "ans_safe": overall,
        "think_diagnosis": "",
        "answer_diagnosis": "",
        "verifier_feedback": "",
        f"{branch}_judge": judge,
    }
    if data_source == "custom_safety_dataset":
        result["think_ans_mismatch"] = 0.0
    return result


async def _score_one_api_overall(inp, sem, *, api_type, model_name, format_mode):
    """LLM API-based scoring: whole-response safety judge for harmful data; split benign prompt unchanged."""
    response = inp.get("response") or ""
    query = _extract_query(inp)
    data_source = _data_source_from_type(inp.get("data_type"))

    async with sem:
        if data_source == "custom_safety_dataset":
            prompt = A.build_safety_overall_prompt(query, response)
            raw = await A.call_llm_judge_async(prompt=prompt, api_type=api_type, model_name=model_name)
            judge = A._parse_safety_boxed_score(raw)
            think_judge = float(judge)
            ans_judge = float(judge)
            think_diagnosis = ""
            answer_diagnosis = ""
            verifier_feedback = ""
            branch = "safety"
        else:
            cot, ans = A.separate_think_and_answer(response)
            if not ans.strip():
                ans = response.strip()
            prompt = A.build_benign_answer_refusal_prompt(query, cot, ans)
            raw = await A.call_llm_judge_async(prompt=prompt, api_type=api_type, model_name=model_name)
            judge = A._parse_benign_trinary_score(raw)
            think_judge = float(judge)
            ans_judge = float(judge)
            think_diagnosis = ""
            answer_diagnosis = ""
            verifier_feedback = ""
            branch = "benign"

        fmt, _ = await _format_score(response, format_mode, api_type, model_name)

    think_judge *= float(fmt)
    ans_judge *= float(fmt)
    overall = float(judge) * float(fmt)
    result = {
        "overall": overall,
        "accuracy": float(judge),
        "format": float(fmt),
        "think_safe": float(think_judge),
        "ans_safe": float(ans_judge),
        "think_diagnosis": think_diagnosis,
        "answer_diagnosis": answer_diagnosis,
        "verifier_feedback": verifier_feedback,
        f"{branch}_judge": float(judge),
    }
    if data_source == "custom_safety_dataset":
        result["think_ans_mismatch"] = 0.0
    return result


async def _score_one_api(inp, sem, *, api_type, model_name, format_mode):
    """LLM API-based scoring (split think/answer safety judge)."""
    response = inp.get("response") or ""
    query = _extract_query(inp)
    data_source = _data_source_from_type(inp.get("data_type"))

    async with sem:
        if data_source == "custom_safety_dataset":
            prompt = A.build_safety_prompt(query, response)
            raw = await A.call_llm_judge_async(prompt=prompt, api_type=api_type, model_name=model_name)
            think_judge, ans_judge, judge = A._parse_safety_think_answer_scores(raw)
            think_diagnosis, answer_diagnosis, verifier_feedback = A._parse_safety_verifier_feedback(raw)
            branch = "safety"
        else:
            cot, ans = A.separate_think_and_answer(response)
            if not ans.strip():
                ans = response.strip()
            prompt = A.build_benign_answer_refusal_prompt(query, cot, ans)
            raw = await A.call_llm_judge_async(prompt=prompt, api_type=api_type, model_name=model_name)
            judge = A._parse_benign_trinary_score(raw)
            think_judge = float(judge)
            ans_judge = float(judge)
            think_diagnosis = ""
            answer_diagnosis = ""
            verifier_feedback = ""
            branch = "benign"

        fmt, _ = await _format_score(response, format_mode, api_type, model_name)

    raw_think_judge = float(think_judge)
    raw_ans_judge = float(ans_judge)
    think_judge *= float(fmt)
    ans_judge *= float(fmt)
    overall = float(judge) * float(fmt)
    result = {
        "overall": overall,
        "accuracy": float(judge),
        "format": float(fmt),
        "think_safe": float(think_judge),
        "ans_safe": float(ans_judge),
        "think_diagnosis": think_diagnosis,
        "answer_diagnosis": answer_diagnosis,
        "verifier_feedback": verifier_feedback,
        f"{branch}_judge": float(judge),
    }
    if data_source == "custom_safety_dataset":
        result["think_ans_mismatch"] = 1.0 if raw_think_judge != raw_ans_judge else 0.0
    return result


async def _score_one(inp, sem, *, api_type, model_name, format_mode):
    response = inp.get("response") or ""
    data_source = _data_source_from_type(inp.get("data_type"))

    if format_mode == "rule":
        fmt = R.get_format_reward(response)
        return await _score_one_rule(inp, data_source, response, fmt)
    if format_mode == "api_overall":
        return await _score_one_api_overall(
            inp, sem, api_type=api_type, model_name=model_name, format_mode=format_mode
        )
    return await _score_one_api(inp, sem, api_type=api_type, model_name=model_name, format_mode=format_mode)


def compute_score(
    reward_inputs,
    api_type: str = "openai",
    model_name: str = "gpt-4o-mini",
    max_concurrency: int = 16,
    format_mode: str = "api",
    **kwargs,
):
    if not reward_inputs:
        return []

    async def _run():
        sem = asyncio.Semaphore(max_concurrency)
        return await asyncio.gather(
            *[
                _score_one(
                    inp,
                    sem,
                    api_type=api_type,
                    model_name=model_name,
                    format_mode=format_mode,
                )
                for inp in reward_inputs
            ]
        )

    return asyncio.run(_run())
