<div align="center">
<img src="assets/eopsa-mascot.png" height="108" align="absmiddle" alt="EOPSA 吉祥物：把大量 token 蒸馏为三个安全关键 token，并修剪 rollout 序列">
<img src="assets/eopsa-logo.svg" height="78" align="absmiddle" alt="EOPSA">

**Efficient On-Policy Self-Distilled Safety Alignment**

<p>
  <strong>Qirui Liu</strong><sup>1,*</sup>, <strong>Yichen Sun</strong><sup>1,*</sup>, Yan Wang<sup>2</sup>, Yu Mi<sup>1</sup>,<br>
  Wei Cao<sup>1</sup>, Yue Shen<sup>2</sup>, Zhixuan Chu<sup>1,†</sup>, Kui Ren<sup>1</sup><br>
  <sup>1</sup> Zhejiang University &nbsp;&nbsp; <sup>2</sup> Ant Group<br>
  <small>* Equal contribution &nbsp;&nbsp; † Corresponding author: <a href="mailto:zhixuanchu@zju.edu.cn">zhixuanchu@zju.edu.cn</a></small>
</p>

<p align="center">
  <a href="https://neuqrui.github.io/EOPSA-page/"><img src="https://img.shields.io/badge/Project-Page-5e83c7?logo=googlechrome&amp;logoColor=white" alt="Project Page" height="23"></a>
  <a href="https://arxiv.org/abs/2609.34519"><img src="https://img.shields.io/badge/Paper-Read-b31b1b?logo=arxiv&amp;logoColor=white" alt="Paper" height="23"></a>
  <a href="https://github.com/neuqrui/EOPSA"><img src="https://img.shields.io/badge/Code-GitHub-383743?logo=github&amp;logoColor=white" alt="Code" height="23"></a>
  <a href="https://huggingface.co/collections/neuqrui/eopsa"><img src="https://img.shields.io/badge/Models-Hugging%20Face-ffc857?logo=huggingface&amp;logoColor=white" alt="Models" height="23"></a>
</p>
</div>

完整说明见 [English README](README.md)。

EOPSA 把 rollout 与梯度预算集中在**监督可靠**且**安全关键**的 token 上：Adaptive Rollout Scheduling 按 Teacher Rescue Rate 截断不可救后缀，Selective Distillation 只更新 `Pivot / Intent / Risk`。相对全序列 OPSD，rollout 计算约减半，反向传播约 2% 的 token。
