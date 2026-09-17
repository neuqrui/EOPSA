---
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: label
    dtype: string
  - name: response
    dtype: string
  splits:
  - name: train
    num_bytes: 231739322
    num_examples: 40000
  download_size: 124398248
  dataset_size: 231739322
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

### SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities

[[Project Page]](https://safe-chain.github.io/)

This is the dataset developped in our work, check details at Section 5.

To use this dataset,
```
from datasets import load_dataset

dataset = load_dataset('UWNSL/SafeChain')
```

Note: To train with dataset in DeepSeek-R1 style CoT, make sure the chat template is consistent. The response filed in our dataset does not include begin-of-thinking tag, `<think>`, due to the update by official DeepSeek team. If you are using some training library (e.g., LLaMA-Factory), ensure the chat template is correct. To train the model other than R1-series model, make sure to add `<think>` tag in response, to process in other ways.




If you use our dataset, please consider citing our work:
```
@misc{jiang2025safechainsafetylanguagemodels,
      title={SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities}, 
      author={Fengqing Jiang and Zhangchen Xu and Yuetai Li and Luyao Niu and Zhen Xiang and Bo Li and Bill Yuchen Lin and Radha Poovendran},
      year={2025},
      eprint={2502.12025},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.12025}, 
}
``` 