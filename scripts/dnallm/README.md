# FungiHelixSeek — DNALLM Integration Scripts

These scripts use the [DNALLM](https://github.com/zhangtaolab/dnallm) package
(v0.5.2+) which provides high-level wrappers for common DNA foundation model operations.

## ⚠️ Prerequisites

```bash
pip install dnallm  # or from source: cd <dnallm-dir> && pip install -e .
```

## All Scripts

| Script | Task | Status | DNALLM API |
|--------|------|--------|------------|
| `finetune.py` | Classification fine-tuning | ✅ | `DNATrainer(task_type="binary"/"multiclass")` |
| `predict.py` | Fine-tuned model prediction | ✅ | `DNAInference.infer()` |
| `embeddings.py` | Embedding extraction | ✅ | `DNAInference.get_embeddings()` |
| `scoring.py` | Sequence scoring (loss / probability / logits) | ✅ | `DNAInference.scoring()` |
| `mutagenesis.py` | Saturation mutagenesis scan | ✅ | `Mutagenesis.evaluate()` |
| `interpret.py` | Model interpretability (Captum) | ✅ | `DNAInterpret.interpret()` |

## Unsupported Tasks (use `scripts/huggingface/` instead)

| Task | Reason | Alternative |
|------|--------|-------------|
| MLM mask recovery | DNALLM has no `fill_mask` method | `scripts/huggingface/inference.py` |

## Usage Examples

### Fine-tuning

```bash
python scripts/dnallm/finetune.py \
    --model_path models/FungiHelixSeek \
    --dataset_dir data/test_data/finetune_example_data \
    --output_dir tmp/dnallm_finetune
```

### Prediction

```bash
python scripts/dnallm/predict.py \
    --model_path tmp/dnallm_finetune \
    --input_file data/test_data/predict_example.csv \
    --output_file tmp/dnallm_predictions.csv
```

### Embedding Extraction

```bash
python scripts/dnallm/embeddings.py \
    --model_path models/FungiHelixSeek \
    --input_file data/test_data/predict_example.csv \
    --output_file tmp/embeddings.npy \
    --reduce_strategy mean
```

### Sequence Scoring

```bash
# Loss-based scoring (lower = more likely under the model)
python scripts/dnallm/scoring.py \
    --model_path models/FungiHelixSeek \
    --input_file data/test_data/predict_example.csv \
    --output_file tmp/scores.json \
    --score_type loss

# Probability-based scoring
python scripts/dnallm/scoring.py \
    --model_path tmp/dnallm_finetune \
    --input_file data/test_data/predict_example.csv \
    --output_file tmp/scores.json \
    --score_type probability
```

### Saturation Mutagenesis

```bash
# Scan every position × every nucleotide mutation
python scripts/dnallm/mutagenesis.py \
    --model_path tmp/dnallm_finetune \
    --sequence ATCGGCTAAGCCTGTCAGTTACCGATCG \
    --output_dir tmp/mutagenesis
```

### Model Interpretability (Captum)

```bash
# Integrated Gradients attribution
python scripts/dnallm/interpret.py \
    --model_path tmp/dnallm_finetune \
    --sequence ATCGGCTAAGCCTGTCAGTTACCGATCG \
    --method integrated_gradients \
    --output_dir tmp/interpret

# Other methods: deep_lift, gradient_shap, occlusion, saliency, lime
python scripts/dnallm/interpret.py \
    --model_path tmp/dnallm_finetune \
    --sequence ATCGGCTAAGCCTGTCAGTTACCGATCG \
    --method occlusion \
    --output_dir tmp/interpret
```
