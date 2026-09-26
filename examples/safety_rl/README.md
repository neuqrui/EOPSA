# EOPSA 训练入口

主入口：`safety_opsd_train.sh` + `safety_opsd_config.yaml`。

论文默认（脚本已对齐）：

1. **Adaptive Rollout Scheduling (ARS)** — `ENABLE_ADAPTIVE_ROLLOUT=true`
   - 在前缀集合 `S={128,256,1024}` 上探测 Teacher Rescue Rate (TRR)
   - 取最长满足 `TRR_L ≥ τ` 的 `L`（默认 `τ=0.75`）
   - 需要 `VAL_FREQ=50`（与论文 step 50 解锁更长 horizon 一致）
2. **Selective Distillation** — `TOKEN_FILTER=true`
   - 分类器：默认 `qwen_eopsa`，也可设 `qwen_rubric`（别名 `qwen-rubric`）
   - 保留集 `K`：`pivot,intent,risk_wo_same`（论文 Risk = `risk_wo_same`）
   - 默认 harmful / benign 共用同一套 `taxonomy_keep`（`TOKEN_FILTER_BENIGN_MODE` 为空）
3. **蒸馏目标** — `DISTILLATION_LOSS_TYPE=topk_forward_kl`
4. **数据** — `datasets/safechain-subset/`（4400 harmful + 2200 benign，已随仓库提供，默认不重建）

完整安装、评测与消融说明见仓库根目录 [README.md](../../README.md)。

安全评测数据通过环境变量 `EVAL_LLM_SAFETY_DIR` 指向本地评测套件。
可选重建子集：`REBUILD_DATA=1 bash safety_opsd_train.sh` 或 `python3 prepare_safechain_subset.py`。
