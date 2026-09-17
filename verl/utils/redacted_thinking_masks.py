# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token-level masks split by the closing ``</think>`` tag only.

``batch["responses"]`` contains rollout tokens only (not the prompt). For
DeepSeek-R1-Distill style models, ``<think>`` is often emitted in
the *prompt* side; the response typically starts with thinking content, then
``</think>``, then the user-facing answer.

We therefore locate **only** the closing tag inside ``responses``:

- think: valid response tokens *before* ``</think>``
- answer: valid response tokens *after* ``</think>`` (excluding the tag itself)

If the closing tag is missing, all valid response tokens are treated as think.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedTokenizer

CLOSE_TAG = "</think>"


def find_subsequence(tokens: list[int], pattern: list[int]) -> Optional[int]:
    if not pattern:
        return None
    plen = len(pattern)
    limit = len(tokens) - plen + 1
    for i in range(limit):
        if tokens[i : i + plen] == pattern:
            return i
    return None


def build_redacted_think_answer_masks(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build think / answer masks using only ``</think>`` as the boundary."""
    close_ids = tokenizer.encode(CLOSE_TAG, add_special_tokens=False)

    think_mask = torch.zeros_like(response_mask)
    answer_mask = torch.zeros_like(response_mask)
    batch_size = responses.size(0)

    for i in range(batch_size):
        valid_len = int(response_mask[i].sum().item())
        if valid_len <= 0:
            continue

        tokens = responses[i, :valid_len].tolist()
        close_pos = find_subsequence(tokens, close_ids)
        if close_pos is None:
            think_mask[i, :valid_len] = 1.0
            continue

        if close_pos > 0:
            think_mask[i, :close_pos] = 1.0

        answer_start = close_pos + len(close_ids)
        if answer_start < valid_len:
            answer_mask[i, answer_start:valid_len] = 1.0

    think_mask = think_mask * response_mask
    answer_mask = answer_mask * response_mask
    return think_mask, answer_mask
