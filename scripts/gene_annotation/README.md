# FungiHelixSeek-Anno — Fungal Gene Structure Prediction

FungiHelixSeek-Anno is a token-classification model fine-tuned from the FungiHelixSeek base. It labels every nucleotide in a fungal genome FASTA with one of **17 BILOU tags** spanning gene features (CDS / intron / 5′UTR / 3′UTR / intergenic), then decodes the per-base probabilities into standard GFF3 gene models with a phase-aware HMM Viterbi.

## Pipeline

```
Download                        Predict + HMM decode                        Output
huggingface-cli download   →   annotate_genome.py                      →   annotation.gff3
   (model weights)              (sliding-window inference +                 (gene / mRNA / exon /
                                 15-state Viterbi + span padding)            CDS / UTR features)
```

## Model

| Model | Base | Params | Task | Labels | Download |
|------------|------|:------:|------|:------:|----------|
| `FungiHelixSeek-Anno` | FungiHelixSeek (MLM pretrain) | 263M | Token classification | 17 (BILOU) | [🤗 HF](https://huggingface.co/zhangtaolab/FungiHelixSeek-Anno) · [🧩 MS](https://www.modelscope.cn/models/zhangtaolab/FungiHelixSeek-Anno) |

Weights ship at `models/FungiHelixSeek-Anno/`; load with `trust_remote_code=True` (handled automatically by the script).

> 💡 If `models/FungiHelixSeek-Anno/` is already downloaded, skip [Download](#1-download) and go straight to [Usage](#2-usage-annotate_genomepy).

### Label scheme (17-class BILOU)

| Group | Tags |
|-------|------|
| intergenic | `O` |
| CDS | `B-CDS` `I-CDS` `L-CDS` `U-CDS` |
| Intron | `B-INTRON` `I-INTRON` `L-INTRON` `U-INTRON` |
| 5′ UTR | `B-UTR5` `I-UTR5` `L-UTR5` `U-UTR5` |
| 3′ UTR | `B-UTR3` `I-UTR3` `L-UTR3` `U-UTR3` |

`B`/`I`/`L`/`U` mark the first / interior / last / single base of each element along the direction of transcription.

---

## 1. Download

```bash
huggingface-cli download zhangtaolab/FungiHelixSeek-Anno \
    --local-dir models/FungiHelixSeek-Anno
```

**Via ModelScope**:

```bash
modelscope download --model zhangtaolab/FungiHelixSeek-Anno \
    --local_dir models/FungiHelixSeek-Anno
```

> 💡 Mirror option: `export HF_ENDPOINT=https://hf-mirror.com`.

---

## 2. Usage (`annotate_genome.py`)

```bash
python scripts/gene_annotation/annotate_genome.py \
    --genome-fasta data/gene_annotation/test_genome.fa \
    --model-path models/FungiHelixSeek-Anno \
    --out annotation.gff3 \
    --batch-size 16 --stride 2048 --device auto
```

Run from the repository root. Plain and `.gz` FASTA are both accepted; each contig is processed independently.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--genome-fasta` | *(required)* | Input genome FASTA (plain or gzip) |
| `--model-path` | `models/FungiHelixSeek-Anno` | Model dir (`config.json` + `model.safetensors` + `*.py` architecture code) |
| `--out` | *(required)* | Output GFF3 path |
| `--phase-head` | `data/gene_annotation/phase_head_fungi.npz` | 4-class reading-frame head used by the HMM decoder (ships with this repo; `none` disables it → uniform phase prior, lower decode quality) |
| `--length-priors` | `decoder/length_priors.npz` | Override the empirical intron/exon length-prior tables |
| `--batch-size` | 16 | Inference batch size (RTX 5090: 16 at window 8192) |
| `--stride` | 2048 | Window step; `< window-size` enables overlap averaging (recommended) |
| `--window-size` | 8192 | Inference window (= model max positions) |
| `--pad` | 200 | Gene-span padding in bp, capped at the midpoint between adjacent genes |
| `--device` | auto | `auto` / `cuda:0` / `cpu` |
| `--processes` | 5 | HMM decode worker processes (one contig per task) |
| `--source` | FungiHelixSeek | GFF3 source column |
| `--probs-dir` | `<out>.probs` | Cache dir for per-contig probability npz |
| `--skip-predict` | off | Re-decode from cached probabilities (no GPU needed) |
| `--skip-decode` | off | Stop after caching probabilities (no GFF3) |

### How it works

1. **Predict** — the genome is scanned in 8 192 bp windows on both the forward strand and the reverse complement. With `--stride 2048` each base is covered by four overlapping windows whose softmax probabilities are averaged, which measurably improves boundary quality. Per-base 17-class probabilities, phase probabilities and boundary scores are cached per contig under `--probs-dir`.
2. **HMM decode** — a 15-state phase-aware Viterbi (gene grammar: ATG start → frame-tracking CDS states → GT..AG introns with hard splice constraints → stop codon → UTR) converts per-base probabilities into gene models. Empirical intron/exon length priors (60 train species) and a split-gene rescue pass are applied; `+` strand genes come from the forward arrays, `−` strand genes from the reverse-complement arrays.
3. **Pad** — gene/mRNA footprints are extended by `--pad` bp (gap-capped between neighbours) to recover UTR-flanked spans; CDS/exon coordinates are never touched.

Throughput: ~3 Mb/min on an RTX 5090 D at `--stride 2048`; decoding adds ~1 s/contig-Mb on CPU. Re-tuning the decoder (stride/pad/priors) reuses the cached npz via `--skip-predict` without re-running inference.

---

## 3. Output

`annotation.gff3` — standard GFF3 with `ID`/`Parent` hierarchy:

| Feature | Description |
|---------|-------------|
| `gene` | Gene footprint (CDS span ± detected UTRs, padded by `--pad`) |
| `mRNA` | Single transcript per gene (`<gene>.t1`) |
| `exon` | Exonic intervals (CDS + UTR segments merged across introns) |
| `CDS` | Coding segments with reading-frame phase (0/1/2) in column 8 |
| `five_prime_UTR` / `three_prime_UTR` | UTR segments when predicted (kept unpadded) |

```
test_contig  FungiHelixSeek  gene   1     1368  .  +  .  ID=FungiHelixSeek_test_contig_g1
test_contig  FungiHelixSeek  mRNA   1     1368  .  +  .  ID=FungiHelixSeek_test_contig_g1.t1;Parent=FungiHelixSeek_test_contig_g1
test_contig  FungiHelixSeek  exon   11    1168  .  +  .  Parent=FungiHelixSeek_test_contig_g1.t1
test_contig  FungiHelixSeek  CDS    11    1168  .  +  0  Parent=FungiHelixSeek_test_contig_g1.t1
```

`<out>.probs/<contig>.npz` — cached probability arrays (`fwd/rev_probs17` (L,17) float16, `fwd/rev_phase` (L,4), `fwd/rev_bnd` (L,)), one file per contig.

---

## Requirements

- Python ≥ 3.12
- GPU recommended (bf16 autocast on CUDA); CPU works but is slow
- Deps: `torch`, `transformers`, `safetensors`, `biopython`, `numpy`

## Files

| File | Purpose |
|------|---------|
| `annotate_genome.py` | Main entry: FASTA → probabilities → HMM decode → GFF3 |
| `decoder/` | 15-state phase-aware HMM Viterbi (states, viterbi, GFF3 writer) |
| `decoder/length_priors.npz` | Empirical intron/exon length priors (60 train species) |
| `lib/predict_core.py` | Sliding-window inference + model loading (HF cache workaround) |
| `lib/decode_core.py` | Probability npz → gene dicts (both strands) |
| `lib/extend_core.py` | Gap-capped gene-span padding |
| `data/gene_annotation/phase_head_fungi.npz` | Frozen 4-class phase head for the decoder |
| `data/gene_annotation/test_genome.fa` | 80 kb *S. pombe* test contig |
| `data/gene_annotation/transition_probs_fungi.npz` | Legacy coarse 5-class transition priors (not used by the 15-state decoder; retained for compatibility) |
