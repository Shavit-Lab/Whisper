# [The Sparsity Whisperer](https://arxiv.org/abs/2608.06630) - NeurIPS 2026
Official implementation of **Wisp**, **Wisp+**, and **Whisper** from [The Sparsity Whisperer](https://arxiv.org/abs/2608.06630), a family of difference-informed one-shot LLM pruning methods that consistently improves Wanda, SparseGPT, RIA, and ALPS across model families, scales, and sparsity settings, shifting the accuracy–pruning-time frontier outward.

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

This is enough to run every pruning method and the WikiText-2 perplexity evaluation. Harness-based evaluation (`--eval`, `--eval_llm_leaderboard_v1`) additionally requires the `v0.3.0` fork of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) that [Wanda](https://github.com/locuslab/wanda) distributes [here](https://drive.google.com/file/d/1zugbLyGZKsH1L19L9biHLfaGGFnEc7XL/view?usp=sharing). Unzip it into the repository root and install it in editable mode:
```
pip install -e lm-evaluation-harness
```
This pulls in the harness's own dependencies, so install it after the steps above.

The bundled `v0.3.0` harness predates `datasets` removing `load_metric`, so `lm_eval/tasks/scrolls.py` fails with `ImportError: cannot import name 'load_metric' from 'datasets'` on the pinned `datasets==3.2.0`. Fix by changing line 33 from `from datasets import load_metric` to `from datasets import load as load_metric`.

### Running the examples

Every method is run through `main.py`. The basic form is:
```
python main.py \
    --model <hf-model-path> \
    --prune_method <method> \
    --sparsity_ratio <ratio> \
    --sparsity_type <unstructured|n:m>
```
- `--model`: either a huggingface model path or a local path of a downloaded huggingface model.
- `--prune_method`: one of `magnitude`, `wanda`, `ria`, `wisp`, `wispplus`, `ria-wisp`, `ria-wispplus`, `sparsegpt`, `whisper`, `alps`, `alps-whisper`.
- `--sparsity_ratio`: fraction of weights removed, e.g. `0.5`. Leaving it at `0` skips pruning and just evaluates the dense model, so it must still be set for `n:m` sparsity even though the pattern already fixes the ratio.
- `--sparsity_type`: `unstructured` (default) or `n:m`, e.g. `2:4`.
- `--pairing`: how Wisp/Whisper form activation differences — `centered` (default, deterministic) or `random` (stochastic).
- `--save` / `--save_eval` / `--save_model`: directories for the perplexity log, the evaluation results, and the pruned checkpoint.
- `--offload_to_cpu`: keeps the model on CPU and moves one layer at a time to the GPU. Use this for models that do not fit in GPU memory (e.g. Llama 2 70B).
- `--seed`: seed for sampling the calibration data (default 0)
- `--nsamples`: number of calibration samples (default 128)

WikiText-2 test perplexity is always reported. To also run harness tasks, pass `--eval` with a matching `--num_fewshot`, or `--eval_llm_leaderboard_v1` for the full Open LLM Leaderboard v1 suite with `mathqa` instead of `gsm8k`.

For example, pruning Llama 2 7B to 2:4 structured sparsity with Whisper and evaluating on WikiText-2 plus a few tasks:
```
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method whisper \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save results/Llama-2-7B/whisper/sparsity_2-4 \
    --save_eval results/Llama-2-7B/whisper/sparsity_2-4 \
    --eval arc_challenge winogrande \
    --num_fewshot 25 5
```
The model can also be downloaded locally prior to running with `huggingface-cli download <hf-model-path> --local-dir <local-path>` and using the local path instead. Swapping `--prune_method whisper` for `sparsegpt` gives the baseline it is compared against, and `--sparsity_type unstructured` reproduces the unstructured setting.

The remaining flags (`--n_pairs`, `--gamma`, `--top_frac`, `--ria_a`, `--starts_with`, `--baseline_fallback`, `--alps_rho`, `--alps_max_iter`, `--true_sequential`, and `--pairing knn`) are ablation knobs, using any of them requires passing `--testing` as well. Run `python main.py --help` for the full list.

### Results

![Shifting the Pareto Front](assets/pareto.png)

Difference-informed pruning shifts the accuracy-runtime frontier outward. Perplexity vs. end-to-end pruning time is shown for the Llama 2 family at 2:4 sparsity (lower-left is better). Orange markers denote baseline pruning methods, while blue markers denote their difference-informed counterparts. Filled markers denote Pareto-efficient methods and hollow markers denote dominated methods. Lines connect Pareto-efficient points within each group, and the shaded region highlights the improvement in attainable accuracy-runtime tradeoffs. Results collected over three trials.

### Acknowledgments

This repository was built upon the [Wanda](https://github.com/locuslab/wanda), [SparseGPT](https://github.com/IST-DASLab/sparsegpt), and [ALPS](https://github.com/mazumder-lab/ALPS) repositories.

### Cite
If you found our work useful, please cite our paper:
```
@inproceedings{kong2026sparsity,
  title={The Sparsity Whisperer},
  author={Kong, Linghao and Subramanian, Inimai and Adler, Micah and Alistarh, Dan and Gutfreund, Dan and Shavit, Nir},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026}
}

```
