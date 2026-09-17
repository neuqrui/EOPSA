# WildJailbreak-100 (flip probe)

100 unique **adversarial_harmful** prompts from AllenAI WildJailbreak
(local arrow cache used by `build_val_wjwc_mix.py`).

- Seed `42`
- Excludes prompts already in DSR100 `val_wjwc_mix` / `train` wildjailbreak rows

## Files

- `val_harmful.jsonl` — flip probe prompts
- `meta.json` — provenance

## Usage

Default flip source in `safety_opsd_train.sh` is now `val` (WJ+WC mix via `USE_WJWC_VAL_HARMFUL`).
To use this dedicated WildJailbreak-100 set instead:

```bash
USE_WJWC_VAL_HARMFUL=true \
FLIP_PROBE_HARMFUL_SOURCE=wildjailbreak100 \
bash safety_opsd_train.sh
```

Aliases: `wj100` / `wildjailbreak`.
