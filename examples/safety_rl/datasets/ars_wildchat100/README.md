# WildChat-100 (plot pipeline set)

Exact **100** WildChat questions from:

`scripts/outputs/student_prefix_continue/wildchat/qwen3-4b_prefix_pipeline.json`

## Files

- `val_harmful.jsonl` — 100 harmful-only rows (for `FLIP_PROBE_HARMFUL_SOURCE=wildchat100`)
- `val_with_benign.jsonl` — 100 harmful + current DSR100 val benign (for full val)

## Usage

```bash
# A) Replace validation harmful with WildChat-100 (keeps benign); flip reuses val rollouts
USE_WILDCHAT100_VAL_HARMFUL=true bash safety_opsd_train.sh

# B) Keep current val; only flip-probe uses WildChat-100 (extra student rollout)
FLIP_PROBE_HARMFUL_SOURCE=wildchat100 bash safety_opsd_train.sh
```

Flip-probe logs (prefix teacher DSR + selected step) are written to:

`<checkpoint_exp>/flip_probe_logs/step_XXXXXX.json` and `index.jsonl`.
