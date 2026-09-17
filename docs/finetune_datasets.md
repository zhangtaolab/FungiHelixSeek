# Fine-tune Datasets & Model–Dataset Mapping

This document details the **3 dataset repos** and **14 fine-tuned models** in the [FungiHelixSeek collection](https://huggingface.co/collections/zhangtaolab/fungihelixseek). All models are fine-tuned from the FungiHelixSeek base (263M, MLM) with `trust_remote_code=True`.

---

## Overview — Model ↔ Dataset Mapping

| Fine-tuned Model | Task Type | Dataset | Local Model Path | Local Data Path |
|------------------|:---------:|---------|-------------------|-----------------|
| `FungiHelixSeek-Taxonomy` | multi-head classification | fungi-its-barcodes | `models/FungiHelixSeek-Taxonomy` | `data/species_classification` |
| `FungiHelixSeek-Anno` | token classification (17 BILOU) | FungiHelixSeek-Anno | `models/FungiHelixSeek-Anno` | `data/gene_annotation_dataset` |
| `FungiHelixSeek-H3` | binary | fungi-histone-modifications (H3) | `models/finetuned/FungiHelixSeek-H3` | `data/Bio_finetune_data/fungi-histone-modifications/H3` |
| `FungiHelixSeek-H3K4me1` | binary | fungi-histone-modifications (H3K4me1) | `models/finetuned/FungiHelixSeek-H3K4me1` | `.../H3K4me1` |
| `FungiHelixSeek-H3K4me2` | binary | fungi-histone-modifications (H3K4me2) | `models/finetuned/FungiHelixSeek-H3K4me2` | `.../H3K4me2` |
| `FungiHelixSeek-H3K4me3` | binary | fungi-histone-modifications (H3K4me3) | `models/finetuned/FungiHelixSeek-H3K4me3` | `.../H3K4me3` |
| `FungiHelixSeek-H3K9ac` | binary | fungi-histone-modifications (H3K9ac) | `models/finetuned/FungiHelixSeek-H3K9ac` | `.../H3K9ac` |
| `FungiHelixSeek-H3K14ac` | binary | fungi-histone-modifications (H3K14ac) | `models/finetuned/FungiHelixSeek-H3K14ac` | `.../H3K14ac` |
| `FungiHelixSeek-H3K36me3` | binary | fungi-histone-modifications (H3K36me3) | `models/finetuned/FungiHelixSeek-H3K36me3` | `.../H3K36me3` |
| `FungiHelixSeek-H3K79me3` | binary | fungi-histone-modifications (H3K79me3) | `models/finetuned/FungiHelixSeek-H3K79me3` | `.../H3K79me3` |
| `FungiHelixSeek-H4` | binary | fungi-histone-modifications (H4) | `models/finetuned/FungiHelixSeek-H4` | `.../H4` |
| `FungiHelixSeek-H4ac` | binary | fungi-histone-modifications (H4ac) | `models/finetuned/FungiHelixSeek-H4ac` | `.../H4ac` |
| `FungiHelixSeek-H3K27ac` | binary | fungi-histone-modifications (H3K27ac) ★ | `models/finetuned/FungiHelixSeek-H3K27ac` | `.../H3K27ac` |
| `FungiHelixSeek-H3K27me3` | binary | fungi-histone-modifications (H3K27me3) ★ | `models/finetuned/FungiHelixSeek-H3K27me3` | `.../H3K27me3` |

> ★ Custom-built (in-house) tasks completing the histone-mark panel — GUE EMP lacks H3K27 marks. The other 10 subsets derive from the [GUE EMP benchmark](https://github.com/lingxinzhou/GUE) (*S. cerevisiae* ChIP-chip, 500 bp windows).

Download any epigenomic model directly:

```bash
for m in H3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K14ac H3K36me3 H3K79me3 H4 H4ac H3K27ac H3K27me3; do
  huggingface-cli download "zhangtaolab/FungiHelixSeek-$m" \
      --local-dir "models/finetuned/FungiHelixSeek-$m" &
done; wait
```

---

## Datasets

### fungi-histone-modifications — 12 binary tasks

- **Task**: Predict whether a 500 bp fungal genomic region carries a given histone mark.
- **Subsets (12)**: `H3`, `H3K4me1`, `H3K4me2`, `H3K4me3`, `H3K9ac`, `H3K14ac`, `H3K36me3`, `H3K79me3`, `H4`, `H4ac` (GUE EMP) + `H3K27ac`, `H3K27me3` (custom-built)
- **Format**: per-subset `train/dev/test.csv` with `sequence,label` columns
- **Models**: 12 models, one per subset — see mapping above
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/fungi-histone-modifications) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/fungi-histone-modifications)

### fungi-its-barcodes — 6-level taxonomy

- **Task**: Assign fungal ITS barcode sequences to phylum → class → order → family → genus → species.
- **Files**: `trainset.fasta` (~1.5 M references), `trainset_valid.fasta`, `test1.fasta` (yeast), `test2.fasta` (filamentous), `test3.fasta` (large-scale benchmark, 367k sequences), `top50.fasta` (demo)
- **Format**: UNITE-style FASTA headers — `>accession|k__Fungi;p__…;c__…;o__…;f__…;g__…;s__…|SH_id`
- **Source**: UNITE + INSD for Fungi (Abarenkov et al. 2023, [doi:10.15156/bio/2938065](https://doi.org/10.15156/bio/2938065)); partition follows the MyCoAI release (Romeijn et al. 2024, [doi:10.1111/1755-0998.14006](https://doi.org/10.1111/1755-0998.14006))
- **Model**: `FungiHelixSeek-Taxonomy`
- **Pipeline**: [scripts/species_classification/README.md](../scripts/species_classification/README.md)
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/fungi-its-barcodes) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/fungi-its-barcodes)

### FungiHelixSeek-Anno (dataset) — gene structure

- **Task**: Per-nucleotide gene-feature labeling (17 BILOU tags over CDS / intron / 5′UTR / 3′UTR / intergenic) on 8,192 bp genomic windows.
- **Files**: `train.parquet` (245k species-balanced + intron-tilted windows, 122 fungal species), `val.parquet`, `splits/species_split.tsv`, `species_list.tsv`, `subsample_metadata.json`
- **Model**: `FungiHelixSeek-Anno` (two-stage training: stage-1 dual-head weighted CE → stage-2 intron-weighted continuation)
- **Pipeline**: [scripts/gene_annotation/README.md](../scripts/gene_annotation/README.md)
- **Source**: [🤗 dataset](https://huggingface.co/datasets/zhangtaolab/FungiHelixSeek-Anno) · [🧩 ModelScope](https://www.modelscope.cn/datasets/zhangtaolab/FungiHelixSeek-Anno)

---

## Training Summary

| Model group | Training | Epochs | Notes |
|-------------|----------|:------:|-------|
| 12 epigenomic models | `scripts/huggingface/finetune.py` | 3 | lr 2e-5, batch 8, single NVIDIA GPU |
| `FungiHelixSeek-Taxonomy` | `scripts/species_classification/train.py` | 10 | HLS (per-rank ε=0.02) + class-weighted CE, AdamW cosine |
| `FungiHelixSeek-Anno` | bespoke token-classification pipeline | 2 stages | dual-head (BILOU+phase) weighted CE → intron-weighted continuation |

Per-model dev-set scores are listed in the [main README benchmark table](../README.md#benchmark).

---

## Download

### Datasets

```bash
source FungiHelixSeek/bin/activate
export HF_ENDPOINT=https://hf-mirror.com   # optional mirror

# Histone modifications (all 12 subsets)
huggingface-cli download zhangtaolab/fungi-histone-modifications \
    --repo-type dataset \
    --local-dir data/Bio_finetune_data/fungi-histone-modifications

# ITS barcodes
huggingface-cli download zhangtaolab/fungi-its-barcodes \
    --repo-type dataset \
    --local-dir data/species_classification

# Annotation fine-tune data
huggingface-cli download zhangtaolab/FungiHelixSeek-Anno \
    --repo-type dataset \
    --local-dir data/gene_annotation_dataset
```

**Via ModelScope**:

```bash
modelscope download --dataset zhangtaolab/fungi-histone-modifications \
    --local_dir data/Bio_finetune_data/fungi-histone-modifications
modelscope download --dataset zhangtaolab/fungi-its-barcodes \
    --local_dir data/species_classification
modelscope download --dataset zhangtaolab/FungiHelixSeek-Anno \
    --local_dir data/gene_annotation_dataset
```

### Fine-tuned Models

```bash
# Taxonomy + Anno
for m in FungiHelixSeek-Taxonomy FungiHelixSeek-Anno; do
  huggingface-cli download "zhangtaolab/$m" --local-dir "models/$m" &
done

# All 12 epigenomic models in parallel
for m in H3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K14ac H3K36me3 H3K79me3 H4 H4ac H3K27ac H3K27me3; do
  huggingface-cli download "zhangtaolab/FungiHelixSeek-$m" \
      --local-dir "models/finetuned/FungiHelixSeek-$m" &
done; wait
```

**Via ModelScope**:

```bash
for m in H3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K14ac H3K36me3 H3K79me3 H4 H4ac H3K27ac H3K27me3; do
  modelscope download --model "zhangtaolab/FungiHelixSeek-$m" \
      --local_dir "models/finetuned/FungiHelixSeek-$m" &
done; wait
```

---

## Usage

```bash
# Fine-tune on a histone-mark subset
python scripts/huggingface/finetune.py \
    --pretrained_model models/FungiHelixSeek \
    --dataset_dir data/Bio_finetune_data/fungi-histone-modifications/H3K4me3 \
    --output_dir tmp/finetune_output \
    --tokens_length 512 --num_train_epochs 3

# Predict with a fine-tuned model
python scripts/huggingface/predict.py \
    --model_path models/finetuned/FungiHelixSeek-H3K4me3 \
    --input_file data/test_data/predict_example.csv \
    --output_file tmp/predictions.csv
```

> 📌 The bundled `data/test_data/finetune_example_data/` is only a smoke-test fixture; use the datasets above for real experiments.
