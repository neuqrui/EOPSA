Paper SafeChain subset for EOPSA (veRL jsonl schema).

  harmful: 4400
  benign:  2200
  files:   train.jsonl / val.jsonl / val_wjwc_mix.jsonl

Default training uses this directory as-is (do not regenerate):

  bash safety_opsd_train.sh

Optional rebuild from UWNSL/SafeChain (overwrites local files):

  REBUILD_DATA=1 bash safety_opsd_train.sh
  # or: python3 prepare_safechain_subset.py
