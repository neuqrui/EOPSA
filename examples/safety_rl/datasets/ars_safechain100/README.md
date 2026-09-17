# SafeChain-100 (DSR100 val harmful)

Exact **100** SafeChain harmful rows from:

`datasets/safety_ds_safechain_dsr100_h4400_b2200/val.jsonl`
(`data_type == safety`: 52 adversarial_harmful + 48 vanilla_harmful)

## Files

- `val_harmful.jsonl` — 100 harmful-only rows (for `FLIP_PROBE_HARMFUL_SOURCE=safechain100`)
- `meta.json` — provenance

## Usage

Keep WJWC as validation harmful, but run flip on SafeChain-100:

```bash
USE_WJWC_VAL_HARMFUL=true \
FLIP_PROBE_HARMFUL_SOURCE=safechain100 \
bash safety_opsd_train.sh
```

(`FLIP_PROBE_HARMFUL_SOURCE=safechain` is an alias of `safechain100`.)

This sets `data.flip_probe_files` to this jsonl, so flip does **not** reuse WJWC val rollouts
(extra student generation on the SafeChain-100 prompts). Exp name gets `_sc100`.
