# SafeChain-100 fail @ Qwen3-1.7B (flip probe)

100 SafeChain **train** harmful prompts where **Qwen3-1.7B** failed Llama-Guard-3-8B
(DSR filter; thinking on, T=0.5).

## Files

- `val_harmful_fail100_qwen3-1.7b.jsonl` — flip probe prompts (training schema)
- `fail100_with_responses_qwen3-1.7b.json` — same + student response + guard text
- `meta.json` — provenance (seed=42, scanned=773, labels 62 adv / 38 vanilla)

## Usage (WJWC val + this flip set)

```bash
USE_WJWC_VAL_HARMFUL=true \
FLIP_PROBE_HARMFUL_SOURCE=safechain100_fail_qwen3-1.7b \
bash safety_opsd_train.sh
```

Aliases: `safechain100_fail` / `sc100fail` / `sc100fail17`.
