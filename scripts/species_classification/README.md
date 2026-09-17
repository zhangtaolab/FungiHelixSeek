# FungiHelixSeek-Taxonomy — Fungal ITS Taxonomic Classification

FungiHelixSeek-Taxonomy is a multi-head classification model fine-tuned from the FungiHelixSeek base. It assigns fungal **internal transcribed spacer (ITS) barcode sequences** to six hierarchical taxonomic ranks — **phylum → class → order → family → genus → species** — through six independent linear heads over a max-pooled HelixSeek backbone, trained with hierarchical label smoothing.

## Pipeline

```
Download                      Predict                       (optional) Fine-tune
huggingface-cli download   →  predict.py                 →   train.py
   (model weights)             (FASTA in, taxonomy CSV out)    (your ITS references)
```

## Model

| Model | Base | Task | Heads | Download |
|------------|------|------|:-----:|----------|
| `FungiHelixSeek-Taxonomy` | FungiHelixSeek 263M | 6-level ITS classification | 6 (18 → 14,742 classes) | [🤗 HF](https://huggingface.co/zhangtaolab/FungiHelixSeek-Taxonomy) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/FungiHelixSeek-Taxonomy) |

Weights ship at `models/FungiHelixSeek-Taxonomy/`; load with `trust_remote_code=True` (handled automatically by the scripts).

### Taxonomic heads

| Level | Classes |
|-------|--------:|
| phylum | 18 |
| class | 70 |
| order | 231 |
| family | 791 |
| genus | 3,695 |
| species | 14,742 |

Input is raw ITS sequence — the FASTA header is not used at inference time.

---

## 1. Download

```bash
huggingface-cli download zhangtaolab/FungiHelixSeek-Taxonomy \
    --local-dir models/FungiHelixSeek-Taxonomy
```

**Via ModelScope**:

```bash
modelscope download --model zhangtaolab/FungiHelixSeek-Taxonomy \
    --local_dir models/FungiHelixSeek-Taxonomy
```

> 💡 Mirror option: `export HF_ENDPOINT=https://hf-mirror.com`.

---

## 2. Prediction (`predict.py`)

```bash
python scripts/species_classification/predict.py \
    --model_path models/FungiHelixSeek-Taxonomy \
    --input_fasta data/species_classification/test2.fasta \
    --output tmp/taxonomy_predictions.csv \
    --batch_size 64
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `models/FungiHelixSeek-Taxonomy` | Model dir (also provides `label_encoders.json`) |
| `--input_fasta` | *(required)* | Input FASTA of ITS sequences |
| `--output` | *(required)* | Output CSV path |
| `--device` | `auto` | `cuda`, `cuda:0`, `cpu`, or `auto` |
| `--batch_size` | 32 | Inference batch size |
| `--max_length` | 1024 | Token truncation length (matches training) |
| `--label_encoders` | `None` | Override path to `label_encoders.json` |

### Output

One CSV row per sequence with per-rank label + softmax confidence:

```csv
seq_id,seq_length,phylum,phylum_confidence,...,species,species_confidence
SH1312607.09FU,632,Ascomycota,1.000,...,Zygotorulaspora_mrakii,0.998
```

### Performance

Accuracy per rank on the held-out filamentous-fungi test set (cleanly-labelled reference subset; the number of evaluated references varies by rank):

| Test set | Phylum | Class | Order | Family | Genus | Species |
|----------|-------:|------:|------:|-------:|------:|--------:|
| Test 2 — Filamentous | 0.998 | 0.982 | 0.961 | 0.918 | 0.811 | **0.708** |

| Rank | Evaluated references |
|------|---------------------:|
| Phylum | 10,792 |
| Class | 10,707 |
| Order | 10,646 |
| Family | 10,095 |
| Genus | 10,134 |
| Species | 6,692 |


---

## 3. Training (`train.py`)

Reproduce or extend the model on your own ITS references.

```bash
python scripts/species_classification/train.py \
    --train_fasta data/species_classification/trainset.fasta \
    --valid_fasta data/species_classification/trainset_valid.fasta \
    --model_template models/FungiHelixSeek-Taxonomy \
    --init_backbone_from models/FungiHelixSeek \
    --output_dir tmp/taxonomy_finetuned \
    --epochs 10 --batch_size 64 --learning_rate 1e-4
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_fasta` | *(required)* | Training FASTA (UNITE-style headers) |
| `--valid_fasta` | `None` | Validation FASTA; enables best-model selection on species accuracy |
| `--model_template` | `models/FungiHelixSeek-Taxonomy` | Dir providing the architecture remote code |
| `--init_backbone_from` | `None` | Base FungiHelixSeek dir to seed backbone weights |
| `--label_encoders` | `None` | Reuse an existing `label_encoders.json` to keep class ids stable |
| `--output_dir` | *(required)* | Where the fine-tuned model is saved |
| `--epochs` / `--batch_size` / `--learning_rate` | 10 / 64 / 1e-4 | Optimization hyperparameters |
| `--max_length` | 1024 | Token truncation length |

### Training recipe

The released model was trained for 10 epochs with: hierarchical label smoothing (per-rank ε = 0.02, uniform ε = 0), inverse-frequency class weights (strength 0.5), AdamW + cosine schedule (10 % warmup, min factor 0.1), gradient clipping at 1.0, max length 1,024 tokens.

Unknown ranks in headers (`s__?`) are masked with `ignore_index=-1` and excluded from every loss/metric at that rank.

### Input format

Headers must follow the UNITE convention:

```
>accession|k__Fungi;p__Ascomycota;c__Saccharomycetes;o__...;f__...;g__...;s__Species_epithet|SH1312607.09FU
```

The training dataset (`trainset.fasta`, ~1.5 M references) ships as the `fungi-its-barcodes` dataset — see the [main README](../../README.md#fine-tune-datasets).
