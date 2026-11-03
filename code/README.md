# Reproducing the weight-space experiments

This package contains the experiment harness used for the article, with portable paths, provider-neutral configuration, pinned math-checkpoint revisions, and fixes for single-file/sharded checkpoint handling. It does not contain credentials, raw editor transcripts, private API identifiers, or local checkpoints.

## What is reproducible from this package?

| Experiment | Additional inputs |
|---|---|
| Evaluate original and publicly released Qwen3.5-0.8B edit | Public model and dataset downloads; CUDA GPU |
| Audit the released weights and construct control edits | Public model downloads; CPU RAM sufficient for both checkpoints |
| Recompute the reported table from the companion snapshot | No model downloads; Python standard library |
| Run new DeepSeek, GLM, or Qwen editing trials | Your own compatible API access and provider-specific model identifiers |
| Repeat vision experiments | The three matching ViT checkpoints, CIFAR-100 files, and CUB test images |
| Evaluate the exact new local Astra, DeepSeek, GLM, or Qwen edits | Their saved edited checkpoints, not included in this package |

All four local editors have completed evaluations in the article. The vision fine-tunes and new locally edited checkpoints are not bundled. New editor trials are stochastic and do not reproduce the original trajectories exactly. Astra was run separately; this package does not provide an Astra editor launcher.

## Environment

Use Linux, Python 3.11, NVIDIA drivers compatible with your installed PyTorch/vLLM CUDA builds, and `bubblewrap` (`bwrap`) for editor kernels. User namespaces must be enabled. The environment used for validation has CUDA 12.9 builds of torch 2.13.0 and vLLM 0.32.0. Choose matching GPU wheels for your machine; `requirements.txt` records the installed package versions, not a platform-independent lockfile.

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/render_results.py
```

`render_results.py` reads `../results-snapshot.json` and prints public model names and the saved math scores. The tests exercise grading, sharded checkpoint export, credential removal, and a real persistent sandbox kernel. They do not claim a full rerun of the article’s experiments.

## Evaluate the publicly released math edit

```bash
python scripts/eval_math.py --ckpt base --profile think --bench math500 --n 100 --samples 2
python scripts/eval_math.py --ckpt edited --profile think --bench math500 --n 100 --samples 2
python scripts/eval_math.py --ckpt base --profile think --bench gsm8k --n 100 --samples 2
python scripts/eval_math.py --ckpt edited --profile think --bench gsm8k --n 100 --samples 2
```

These use the article’s 32,768-token thinking profile. Results go to `work/out/math/`. `--profile nothink` selects the separate no-thinking sampling profile. Set `CUDA_VISIBLE_DEVICES` to choose a GPU and lower `--gpu_mem` if necessary. A short startup check can use `--n 1 --samples 1 --max_tokens 64 --enforce_eager --tag smoke`; its score is not a benchmark result. On a shared GPU, automatic memory profiling can fail if another process releases memory during startup. For a bounded smoke test, additionally set `--kv_cache_mb 1024 --max_num_seqs 1 --max_num_batched_tokens 256`. The explicit cache setting overrides automatic cache sizing; it is not a total GPU-memory cap. These options are runtime overrides, not the settings used for the article’s saved scores.

To audit and build the controls, run sequentially:

```bash
python scripts/audit_edit.py
python scripts/make_control_edits.py --seeds 0
python scripts/eval_math.py --ckpt work/ckpts/qwen35_c1 --profile think --bench math500 --n 100 --samples 2
python scripts/eval_math.py --ckpt work/ckpts/qwen35_randdir_s0 --profile think --bench math500 --n 100 --samples 2
python scripts/eval_math.py --ckpt work/ckpts/qwen35_randcols_s0 --profile think --bench math500 --n 100 --samples 2
```

Repeat the last three commands with `--bench gsm8k` for the second benchmark. The control builder replaces its own named output directories on rerun. The released edit is pinned to revision `d1671a178705bcab7644806d23c535517f35723b`; the base is pinned to `2fc06364715b967f1860aea9cf38778875588b17`.

## Run a new editor experiment

Configure credentials in your shell; the package does not load a personal environment file. Use the model IDs accepted by your own provider for **DeepSeek V4.1 Flash 1017**, **GLM 5.3 Flash**, and **Qwen 3.8 27B**:

```bash
export OPENAI_BASE_URL='https://your-provider.example/v1'
export OPENAI_API_KEY='your-key'
export WSP_DEEPSEEK_MODEL='your-provider-deepseek-model-id'
export WSP_GLM_MODEL='your-provider-glm-model-id'
export WSP_QWEN38_MODEL='your-provider-qwen-model-id'
```

These example endpoint and model IDs are placeholders. Reasoning effort defaults to `max`, `max`, and `xhigh`; override `WSP_DEEPSEEK_EFFORT`, `WSP_GLM_EFFORT`, or `WSP_QWEN38_EFFORT` if the provider uses different values. Runs using different models or effort settings are not directly equivalent to the reported runs.

Download the math assets before starting the network-isolated kernel:

```bash
python -c 'from wsp import qwen; qwen.base_dir(); qwen.practice_set()'
python scripts/run_agent.py --task qwen_math --model deepseek --passes 10 --trial 0 --gpu 0
python scripts/eval_math.py --ckpt work/runs/qwen_math_p10/deepseek_t0/ckpt_final --profile think --bench math500 --n 100 --samples 2 --tag deepseek-new
```

Evaluate `gsm8k` similarly. A final checkpoint exists only if the editor actually saves an edit. `--model glm` and `--model qwen38` select the other editors. Use a fresh trial number for a new run; do not mix an interrupted run’s artifacts with a new trajectory. The math kernel keeps a Transformers model resident while launching vLLM generation, so allow memory for both.

## Vision inputs

Supply plain torchvision ViT-B/16 state dictionaries with classifier heads of 1,000, 100, and 200 classes, respectively:

```bash
export VIT_IMAGENET_CKPT='/path/to/imagenet.pt'
export VIT_CIFAR100_CKPT='/path/to/cifar100.pt'
export VIT_CUB_CKPT='/path/to/cub.pt'
export CIFAR_ROOT='/path/to/cifar100'
export CUB_ROOT='/path/to/CUB'
python scripts/prep_vit_fix.py --dataset cifar100
python scripts/prep_vit_fix.py --dataset cub
python scripts/run_agent.py --task vit_identify --model deepseek --trial 0
python scripts/run_agent.py --task vit_fix --dataset cifar100 --model deepseek --passes 10 --trial 0 --gpu 0
```

`CIFAR_ROOT` must contain `cifar-100-python/test` and `meta`. `CUB_ROOT/test` must use ImageFolder class directories such as `059.California_Gull`. The two fine-tunes must share the supplied ImageNet parent; unrelated checkpoints change the identification task.

Set `WSP_SCRATCH` to relocate run artifacts, `WSP_OUT` to relocate evaluation output, and `HF_HOME` to select a download cache. Identification hides the home directory by default. If reference checkpoints or caches live elsewhere, set `WSP_HIDE_ROOTS` to a colon-separated list of all private roots before running identification. The interpreter and harness are explicitly made visible inside the sandbox. Other tasks retain read-only filesystem access; the sandbox is a research harness, not a formal proof of protocol compliance.

See [VALIDATION.md](VALIDATION.md) for the checks actually performed and their limits.
