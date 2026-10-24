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

### Apple Silicon (macOS, MPS)

```bash
uv pip install "torch>=2.12"   # 2.12.0 / 2.13.0 / 2.14.1 verified on macOS arm64
```

macOS has no Triton wheels, so the FLA kernels used by the KDA layers are
unavailable and the model runs a pure PyTorch fallback. Treat MPS as a
validation/debug target, not a throughput target. Two caveats:

- **torch <= 2.11**: MPS SDPA returns a query-shaped output when the value head
  dim is smaller than the query head dim, which crashes the MLA layers. Either
  upgrade to torch>=2.12 or patch a downloaded checkpoint:

  ```bash
  python scripts/huggingface/apply_mps_fix.py --model_path models/FungiHelixSeek
  ```

- **Accuracy**: the shipped pure-PyTorch KDA fallback does not match FLA
  semantics (mask recovery drops to ~chance). Pass `--kda_fallback` to also
  apply the exact recurrence:

  ```bash
  python scripts/huggingface/apply_mps_fix.py --model_path models/FungiHelixSeek --kda_fallback
  ```

Both patches are idempotent, keep the original as `attention.py.orig`, and are
scoped to the code path they fix, so CUDA/NPU runs are unchanged. Then pass
`--device mps` to any script.

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

On Apple Silicon use `--device mps`. You should see ✓ for all checks.

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

The macOS/MPS path is verified on macOS 26.6.2 (arm64) with Python 3.12,
PyTorch 2.14.1 and Transformers 4.57.6.

> ⚠️ CPU mode is for code debugging only — expect tens of seconds per sequence with the 263M-parameter model.
