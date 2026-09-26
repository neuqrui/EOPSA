<div align="center">
<img src="assets/eopsa-mascot.png" height="108" align="absmiddle" alt="EOPSA 吉祥物：把大量 token 蒸馏为三个安全关键 token，并修剪 rollout 序列">
<img src="assets/eopsa-logo.svg" height="78" align="absmiddle" alt="EOPSA">

**Efficient On-Policy Self-Distilled Safety Alignment**
</div>

<p align="center">
  <a href="README.md#citation">📄 Paper</a> &nbsp;·&nbsp;
  <a href=".">💻 Code</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/collections/neuqrui/eopsa">🤗 Models</a>
</p>

完整说明见 [English README](README.md)。

EOPSA 把 rollout 与梯度预算集中在**监督可靠**且**安全关键**的 token 上：Adaptive Rollout Scheduling 按 Teacher Rescue Rate 截断不可救后缀，Selective Distillation 只更新 `Pivot / Intent / Risk`。相对全序列 OPSD，rollout 计算约减半，反向传播约 2% 的 token。
