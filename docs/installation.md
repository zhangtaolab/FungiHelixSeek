# Installation

## 1. Clone the repository

```bash
git clone https://github.com/zhangtaolab/FungiHelixSeek.git
cd FungiHelixSeek
```

## 2. Create the environment with [uv](https://docs.astral.sh/uv/)

The recommended setup is a local `.venv` environment in the repository root:

```bash
# Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the .venv environment (Python 3.12)
uv venv --python 3.12 .venv
source .venv/bin/activate
```

## 3. Install PyTorch (choose your platform)

### NVIDIA GPU (CUDA 12.x / 13.x)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### CPU only (debugging)

```bash
uv pip install torch
```

## 4. Install FungiHelixSeek dependencies

```bash
uv pip install -r requirements.txt
# or editable install:
uv pip install -e .
```

### Optional: Flash Attention (NVIDIA GPU only)

```bash
uv pip install flash-attn>=2.0.0 flash-linear-attention==0.4.1
```

> `flash-linear-attention` is already included in `requirements.txt`; the optional `flash-attn` package further accelerates standard attention layers.

## 5. Download model weights

```bash
# Option A: huggingface-cli
huggingface-cli download zhangtaolab/FungiHelixSeek --local-dir ./models/FungiHelixSeek

# Option B: git lfs
git lfs install
git clone https://huggingface.co/zhangtaolab/FungiHelixSeek ./models/FungiHelixSeek
```

> Optional mirror for slow connections: `export HF_ENDPOINT=https://hf-mirror.com`, or use ModelScope: `modelscope download --model zhangtaolab/FungiHelixSeek --local_dir ./models/FungiHelixSeek`

## 6. Verify installation

```bash
python scripts/huggingface/verify_install.py --model_path models/FungiHelixSeek --device auto
```

You should see ✓ for all checks.

## 7. (Optional) End-to-end smoke test

Verify the full toolchain (model loading → inference → pretraining → fine-tuning → prediction) in one command:

```bash
PYTHONPATH=scripts/huggingface python tests/test_smoke.py
# expected: RESULTS: 5 passed, 0 failed, 0 skipped
```

## Tested environment

| Component      | Version      |
| -------------- | ------------ |
| OS             | Linux        |
| Python         | 3.12         |
| PyTorch        | 2.x (cu128)  |
| Transformers   | 4.49.0       |
| flash-linear-attention | 0.4.1 |
| GPU            | NVIDIA RTX 5090 D (32 GB) |

> ⚠️ CPU mode is for code debugging only — expect tens of seconds per sequence with the 263M-parameter model.
