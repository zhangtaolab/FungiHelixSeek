# FungiHelixSeek-Anno — Fungal Gene Structure Prediction

FungiHelixSeek-Anno is a token-classification model fine-tuned from the FungiHelixSeek base. It labels every nucleotide in a fungal genome FASTA with one of **17 BILOU tags** spanning gene features (CDS / intron / 5′UTR / 3′UTR / intergenic), then decodes the per-base probabilities into standard GFF3 gene models. The default decoder is a vendored Plant Viterbi with terminal ORF repair; the legacy phase-aware HMM Viterbi remains available with `--decoder hmm`.

## Pipeline

```
Download                        Predict + decode                          Output
huggingface-cli download   →   annotate_genome.py                      →   annotation.gff3
   (model weights)              (sliding-window inference +                 (gene / mRNA / exon /
                                 viterbi+repair by default;                  CDS / UTR features)
                                 --decoder hmm = legacy HMM)
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

The command above uses the default `--decoder repair`. To select another decoder:

```bash
# Legacy Fungi HMM (padding defaults to 200 bp)
python scripts/gene_annotation/annotate_genome.py \
    --genome-fasta genome.fa --out annotation.hmm.gff3 --decoder hmm

# Plant Viterbi without terminal repair
python scripts/gene_annotation/annotate_genome.py \
    --genome-fasta genome.fa --out annotation.viterbi.gff3 --decoder viterbi

# Plant Viterbi with ORF correction and intron-probability merging
python scripts/gene_annotation/annotate_genome.py \
    --genome-fasta genome.fa --out annotation.orf.gff3 --decoder orf
```

Run from the repository root. Plain and `.gz` FASTA are both accepted; each contig is processed independently. Switching decoders or decoder settings reuses cached probabilities with `--skip-predict`.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--genome-fasta` | *(required)* | Input genome FASTA (plain or gzip) |
| `--model-path` | `models/FungiHelixSeek-Anno` | Model dir (`config.json` + `model.safetensors` + `*.py` architecture code) |
| `--phase-head` | `data/gene_annotation/phase_head_fungi.npz` | 4-class reading-frame head used by the legacy HMM only; `none` disables it (uniform phase prior, lower HMM decode quality). Ignored by Plant decoders |
| `--length-priors` | `None` (HMM falls back to `decoder/length_priors.npz`) | Override for the empirical intron/exon length-prior tables used by the legacy HMM only. Ignored by Plant decoders |
| `--out` | *(required)* | Output GFF3 path |
| `--probs-dir` | `<out>.probs` | Cache dir for per-contig probability npz |
| `--batch-size` | 16 | Inference batch size (RTX 5090: 16 at window 8192) |
| `--stride` | 2048 | Window step; `< window-size` enables overlap averaging (recommended) |
| `--window-size` | 8192 | Inference window (= model max positions) |
| `--decoder` | `repair` | `repair` / `viterbi` / `orf` use the vendored Plant decoder; `hmm` is the legacy Fungi HMM |
| `--prior` | `fungi` | 5-class transition prior for the Plant decoders: `fungi` (repository fungal matrix) or `plant` (Plant matrix with stationary initial). Ignored by `hmm` |
| `--intergenic-boost` | 0.0 | Plant Viterbi intergenic score boost |
| `--viterbi-alpha` | 0.0 | Plant Viterbi transition smoothing alpha |
| `--genic-threshold` | 0.25 | Plant Viterbi genic probability threshold |
| `--single-exon-prob` | 0.7 | Minimum confidence for a single-exon gene |
| `--single-exon-min-cds` | 200 | Minimum CDS length for a single-exon gene |
| `--repair-w5` | 300 | Repair 5′ extension window in bp |
| `--repair-w3` | 300 | Repair 3′ extension window in bp |
| `--repair-start-rule` | `nearest` | Repair start-codon selection rule (`nearest`, `first`, `longest`) |
| `--orf-max-shift` | 3 | Maximum ORF frame shift for `--decoder orf` |
| `--pad` | 200 for `hmm`, else 0 | Gene-span padding in bp, capped at the midpoint between adjacent genes |
| `--device` | auto | `auto` / `cuda:0` / `mps` / `cpu` |
| `--processes` | 5 | Decode worker processes (one contig per task) |
| `--source` | FungiHelixSeek | GFF3 source column |
| `--skip-predict` | off | Re-decode from cached probabilities (no GPU needed) |
| `--skip-decode` | off | Stop after caching probabilities (no GFF3) |

The Plant Viterbi also uses a fixed minimum CDS run of 5 bp. It is part of the decoder configuration, not a separate CLI flag.

### Decoder methods

| `--decoder` | Method | Notes |
|-------------|--------|-------|
| `repair` | Plant 5-class Viterbi + terminal ORF repair | Default. Keeps internal splice sites fixed and moves only the first/last CDS boundaries, searching `--repair-w5` / `--repair-w3` bp with the `--repair-start-rule` start codon |
| `viterbi` | Plant 5-class Viterbi | Genic filtering and CDS interval formation without terminal repair |
| `orf` | Plant 5-class Viterbi + ORF correction | Corrects ORFs with frame shifts up to `--orf-max-shift` and merges genes by intron probability |
| `hmm` | Legacy Fungi 15-state phase-aware Viterbi | Uses the 4-class phase head and empirical length priors, applies split-gene rescue, and keeps the historical HMM output |

The `repair`, `viterbi`, and `orf` methods are deterministic decoders vendored under `plant_decoder/`. They reuse the same cached 17-class probability arrays as the HMM; they are not a separate model or a second neural inference pass. No sibling PlantHelixSeek checkout or Plant CLI is required at runtime.

Plant decoder defaults are the validated benchmark settings: `prior=fungi`, intergenic boost 0.0, Viterbi alpha 0.0, genic threshold 0.25, minimum CDS run 5 bp, single-exon probability 0.7 with minimum CDS 200 bp, repair windows 300/300 bp with `nearest` start selection, and ORF maximum shift 3.

### How it works

1. **Predict** — the genome is scanned in 8 192 bp windows on both the forward strand and the reverse complement. With `--stride 2048` each base is covered by four overlapping windows whose softmax probabilities are averaged, which measurably improves boundary quality. Per-base 17-class probabilities, phase probabilities and boundary scores are cached per contig under `--probs-dir`.
2. **Decode** — the selected decoder converts the cached arrays into gene models. The default Plant `repair` path runs a 5-class Viterbi over intergenic / intron / 5′UTR / CDS / 3′UTR using BILOU-derived boundaries, then repairs terminal ORFs. The legacy `hmm` path runs the 15-state phase-aware Viterbi (ATG start → frame-tracking CDS states → GT/GC/AT..AG introns with hard splice constraints → stop codon → UTR), applies empirical intron/exon length priors and split-gene rescue, and uses the phase head. `+` strand genes come from the forward arrays; `−` strand genes come from the reverse-complement arrays.
3. **Pad** — gene/mRNA footprints are extended by `--pad` bp (gap-capped between neighbours) to recover UTR-flanked spans; CDS/exon coordinates are never touched. The default is 0 for the Plant decoders and 200 for the legacy HMM.

Throughput: ~3 Mb/min on an RTX 5090 D at `--stride 2048`; decoding adds ~1 s/contig-Mb on CPU. Re-tuning the decoder (decoder choice, stride, pad, priors) reuses the cached npz via `--skip-predict` without re-running inference.

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

Example from the default `--decoder repair` with `--pad 0` on `data/gene_annotation/test_genome.fa` (unpadded coordinates):

```
test_contig  FungiHelixSeek  gene   1     1168  .  +  .  ID=FungiHelixSeek_test_contig_g1
test_contig  FungiHelixSeek  mRNA   1     1168  .  +  .  ID=FungiHelixSeek_test_contig_g1.t1;Parent=FungiHelixSeek_test_contig_g1
test_contig  FungiHelixSeek  exon   1     1168  .  +  .  Parent=FungiHelixSeek_test_contig_g1.t1
test_contig  FungiHelixSeek  CDS    1     1168  .  +  0  Parent=FungiHelixSeek_test_contig_g1.t1
```

The legacy HMM (`--decoder hmm`) defaults to `--pad 200`, so its gene footprint can be wider than the CDS/UTR span above.

All decoders write the same GFF3 schema. Internal decoder coordinates are 0-based half-open and converted to 1-based inclusive GFF3 at the writer boundary. Gene IDs follow `<source>_<contig>_g<N>` with one global counter in FASTA order; each mRNA is `<gene>.t1`. CDS phase, `Parent` links, exon rows, and individual multi-span 5′/3′ UTR segments are preserved. The vendored Plant decoders were parity-checked against cached Plant bridge outputs on CDS/UTR coordinates, CDS phase, strand, and `Parent` relationships; native Fungi IDs and the source column differ from the experimental bridge schema by design, while the evaluated CDS/UTR structures are the same.

`<out>.probs/<contig>.npz` — cached probability arrays (`fwd/rev_probs17` (L,17) float16, `fwd/rev_phase` (L,4), `fwd/rev_bnd` (L,)), one file per contig. The cache is shared by every decoder. Cached reverse-strand arrays are reverse-complement oriented: the Plant decoders reverse them exactly once into forward genomic coordinates, while the legacy HMM decodes the reverse-complement sequence and mirrors coordinates back.

Legacy compatibility: `--decoder hmm --pad 200` reproduces the historical HMM output, and serial versus parallel decoding is identical.

---

## Requirements

- Python ≥ 3.12
- GPU recommended (bf16 autocast on CUDA); CPU works but is slow
- Deps: `torch`, `transformers`, `safetensors`, `biopython`, `numpy`, `pandas`, `numba` (`pandas` and `numba` back the vendored Plant decoders)
- No external PlantHelixSeek checkout or Plant CLI is needed

## Files

| File | Purpose |
|------|---------|
| `annotate_genome.py` | Main entry: FASTA → probabilities → selected decoder → GFF3 |
| `decoder/` | Legacy 15-state phase-aware HMM Viterbi, plus GFF writer dispatch |
| `decoder/length_priors.npz` | Empirical intron/exon length priors used by the legacy HMM |
| `plant_decoder/` | Vendored Plant decoders: 5-class Viterbi, terminal repair, ORF correction, priors, GFF writer |
| `plant_decoder/transition_probs.npz` | Plant full/partial transition matrices for `--prior plant` |
| `lib/predict_core.py` | Sliding-window inference + model loading (HF cache workaround) |
| `lib/decode_core.py` | Probability npz → gene models; dispatches Plant or HMM by `--decoder` |
| `lib/extend_core.py` | Gap-capped gene-span padding |
| `data/gene_annotation/phase_head_fungi.npz` | Frozen 4-class phase head used by the legacy HMM |
| `data/gene_annotation/test_genome.fa` | 80 kb *S. pombe* test contig |
| `data/gene_annotation/transition_probs_fungi.npz` | Fungal 5-class transition matrix + initial distribution, used by the Plant decoders with `--prior fungi` (default) |
