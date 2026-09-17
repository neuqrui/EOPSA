<h1 align="center">EOPSA</h1>
<p align="center"><b>Efficient On-Policy Self-Distilled Safety Alignment</b></p>

<p align="center">
  <a href="README.md#citation">📄 Paper</a> &nbsp;·&nbsp;
  <a href=".">💻 Code</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/collections/neuqrui/eopsa">🤗 Model</a>
</p>

完整说明见 [English README](README.md)。

EOPSA 把 rollout 与梯度预算集中在**监督可靠**且**安全关键**的 token 上：Adaptive Rollout Scheduling 按 Teacher Rescue Rate 截断不可救后缀，Selective Distillation 只更新 `Pivot / Intent / Risk`。相对全序列 OPSD，rollout 计算约减半，反向传播约 2% 的 token。

权重：[Hugging Face Collection](https://huggingface.co/collections/neuqrui/eopsa)（Qwen3 1.7B / 4B / 14B / 32B 与 R1-7B）。安全评测：[LLM-Safety-Eval](https://github.com/neuqrui/LLM-Safety-Eval)。
