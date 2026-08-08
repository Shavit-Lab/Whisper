# ['The Sparsity Whisperer']()
Here we include the code for the official implementations of Wisp, Wisp+, and Whisper for our paper [The Sparsity Whisperer]().

### Overview

![Llama 2 Scaling](assets/llama_2_ppl_curve.png)

Unstructured sparsity scaling on Llama 2. Perplexity is shown for Llama 2 7B, 13B, and 70B as unstructured sparsity increases from 50% to 75% (lower is better). Wisp and Wisp+ consistently improve over Wanda in the no-update setting, while Whisper improves over SparseGPT at every sparsity and model size. The gains grow as pruning becomes more aggressive, suggesting that preserving pairwise output differences becomes increasingly important at high sparsity. Numbers written next to values indicate absolute perplexity improvement of Whisper over SparseGPT.

### Environment setup

Create a conda environment (installs PyTorch and all Python dependencies):
```
conda env create -f environment.yml
conda activate whisper
```
The pinned wheels are built for CUDA 12.4; edit the `--extra-index-url` in `environment.yml` if you need a different CUDA version (see [pytorch.org](https://pytorch.org)).

For a pip-only install, install PyTorch matching your CUDA version first, then:
```
pip install -r requirements.txt
```

This is enough to run every pruning method and the WikiText-2 perplexity evaluation. Zero-shot evaluation (`--eval_zero_shot`) additionally requires the `v0.3.0` fork of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) that [Wanda](https://github.com/locuslab/wanda) distributes [here](https://drive.google.com/file/d/1zugbLyGZKsH1L19L9biHLfaGGFnEc7XL/view?usp=sharing). Unzip it into the repository root and install it in editable mode:
```
pip install -e lm-evaluation-harness
```
This pulls in the harness's own dependencies, so install it after the steps above.

The bundled `v0.3.0` harness predates `datasets` removing `load_metric`, so `lm_eval/tasks/scrolls.py` fails with `ImportError: cannot import name 'load_metric' from 'datasets'` on the pinned `datasets==3.2.0`. Fix by changing line 33 from `from datasets import load_metric` to `from datasets import load as load_metric`.

### Running the examples



### Results

![Shifting the Pareto Front](assets/pareto.png)

Difference-informed pruning shifts the accuracy-runtime frontier outward. Perplexity vs. end-to-end pruning time is shown for the Llama 2 family at 2:4 sparsity (lower-left is better). Orange markers denote baseline pruning methods, while blue markers denote their difference-informed counterparts. Filled markers denote Pareto-efficient methods and hollow markers denote dominated methods. Lines connect Pareto-efficient points within each group, and the shaded region highlights the improvement in attainable accuracy-runtime tradeoffs. Results collected over three trials.

### Acknowledgments

This repository was built upon the [Wanda](https://github.com/locuslab/wanda), [SparseGPT](https://github.com/IST-DASLab/sparsegpt), and [ALPS](https://github.com/mazumder-lab/ALPS) repositories.

### Cite
If you found our work useful, please cite our paper: