# -*- coding: utf-8 -*-
"""Load user-editable OPSD token filter module from a file path."""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from types import ModuleType
from typing import Any, Optional


_DEFAULT_REL = os.path.join(
    "examples", "safety_rl", "token_filter", "filters.py"
)


def _resolve_filter_path(path: Optional[str]) -> str:
    if path:
        candidate = path
    else:
        candidate = _DEFAULT_REL
    if os.path.isabs(candidate) and os.path.isfile(candidate):
        return candidate
    cwd = os.getcwd()
    probes = [os.path.join(cwd, candidate), candidate]
    here = os.path.abspath(os.path.dirname(__file__))
    repo_guess = os.path.abspath(os.path.join(here, "..", ".."))
    probes.append(os.path.join(repo_guess, _DEFAULT_REL if not path else candidate))
    for p in probes:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"Token filter module not found. Tried path={path!r} and default {_DEFAULT_REL!r}."
    )


@lru_cache(maxsize=4)
def load_token_filter_module(path: Optional[str] = None) -> ModuleType:
    abs_path = _resolve_filter_path(path)
    mod_name = f"opsd_token_filter_{abs(hash(abs_path))}"
    spec = importlib.util.spec_from_file_location(mod_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load token filter module from {abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "apply_filter") or not hasattr(module, "TokenFilterContext"):
        raise AttributeError(
            f"{abs_path} must define apply_filter and TokenFilterContext"
        )
    return module


def apply_token_filter(
    mode: str,
    *,
    teacher_logits: Any,
    student_logits: Any,
    base_mask: Any,
    responses: Any,
    token_div: Any,
    top_n: int = 16,
    action: str = "drop",
    decode_id=None,
    extra: Optional[dict] = None,
    module_path: Optional[str] = None,
):
    """Convenience wrapper used by the OPSD actor."""
    module = load_token_filter_module(module_path)
    ctx = module.TokenFilterContext(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        base_mask=base_mask,
        responses=responses,
        token_div=token_div,
        top_n=top_n,
        decode_id=decode_id,
        action=action,
        extra=extra or {},
    )
    # New signature: apply_filter(mode, ctx, action=...)
    try:
        return module.apply_filter(mode, ctx, action=action)
    except TypeError:
        # Older single-arg filters
        return module.apply_filter(mode, ctx)
