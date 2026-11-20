#!/usr/bin/env python3
"""FungiHelixSeek-Anno genome annotation: FASTA -> per-base BILOU
probabilities -> selectable decoder -> GFF3 gene models, one command.

Three stages, each skippable so probabilities can be cached and decoding
re-tuned without re-running GPU inference:
  predict   sliding-window inference (window 8192) on both strands with
            overlap averaging when stride < window; per-contig probability
            npz cached under --probs-dir
  decode    selectable decoder -> raw GFF3. The default --decoder repair is a
            vendored Plant-compatible 5-class Viterbi plus terminal ORF
            repair; --decoder viterbi and --decoder orf are also available.
            --decoder hmm selects the legacy 15-state phase-aware HMM
            Viterbi (GT/GC/AT..AG splice hard constraints, empirical
            intron/exon length priors, split-gene rescue). Gene-span padding
            (--pad) defaults to 200 for hmm and 0 for the Plant decoders
  out       final GFF3 with gene / mRNA / exon / CDS / UTR features

Usage (from the repo root):
  python scripts/gene_annotation/annotate_genome.py \
      --genome-fasta genome.fa.gz \
      --model-path models/FungiHelixSeek-Anno \
      --out annotation.gff3
"""
from __future__ import annotations

import argparse
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lib import predict_core, decode_core, extend_core  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("annotate")

DEFAULT_PHASE_HEAD = REPO_ROOT / "data" / "gene_annotation" / "phase_head_fungi.npz"


def resolve_device(target: str):
    if target != "auto":
        return target
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_length_priors(path: Path | None) -> None:
    """Pre-warm the Viterbi length-prior cache with a custom npz
    (arrays ``intron`` and ``exon``, peak-normalized log-pdf penalties)."""
    if path is None:
        return
    from decoder import viterbi as V
    z = np.load(str(path))
    V._LEN_TABLE = (z["intron"], z["exon"])


def run_predict(args, probs_dir: Path) -> None:
    import torch

    device = torch.device(resolve_device(args.device))
    phase_head = None
    if args.phase_head and args.phase_head.lower() != "none":
        phase_head = Path(args.phase_head)
        if not phase_head.exists():
            logger.warning("phase head %s not found -- continuing without it",
                           phase_head)
            phase_head = None
    model, tokenizer, has_phase = predict_core.load_model(
        Path(args.model_path), device, phase_head)
    overlap = args.stride < args.window_size
    seqs = predict_core.read_fasta(Path(args.genome_fasta))
    probs_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, seq) in enumerate(sorted(seqs.items()), 1):
        logger.info("[predict %d/%d] %s (%d bp)", i, len(seqs), name, len(seq))
        p17, phase, bnd = predict_core.predict_contig(
            model, tokenizer, seq, device, args.batch_size, has_phase,
            window=args.window_size, stride=args.stride, overlap_avg=overlap)
        rp17, rphase, rbnd = predict_core.predict_contig(
            model, tokenizer, predict_core.reverse_complement(seq), device,
            args.batch_size, has_phase, window=args.window_size,
            stride=args.stride, overlap_avg=overlap)
        safe = name.replace("/", "_")
        np.savez_compressed(
            probs_dir / f"{safe}.npz", contig_name=np.array(name),
            fwd_probs17=p17.astype(np.float16), fwd_phase=phase, fwd_bnd=bnd,
            rev_probs17=rp17.astype(np.float16), rev_phase=rphase, rev_bnd=rbnd)


def _decode_contig_task(task):
    """Top-level, picklable per-contig decode worker for ``Pool.map``."""
    name, seq, path, opts = task
    length = len(seq)
    if opts["decoder"] == "hmm" and opts.get("length_priors"):
        apply_length_priors(Path(opts["length_priors"]))
    with np.load(path) as z:
        fwd17 = decode_core._array(z, "fwd_probs17", length, decode_core.N_LABELS)
        fwd_phase = decode_core._array(z, "fwd_phase", length,
                                       decode_core.N_PHASE, 1.0 / decode_core.N_PHASE)
        fwd_bnd = decode_core._array(z, "fwd_bnd", length, None, 0.0)
        rev17 = decode_core._array(z, "rev_probs17", length, decode_core.N_LABELS)
        rev_phase = decode_core._array(z, "rev_phase", length,
                                       decode_core.N_PHASE, 1.0 / decode_core.N_PHASE)
        rev_bnd = decode_core._array(z, "rev_bnd", length, None, 0.0)
    return decode_core.decode_contig(
        name, seq, fwd17, fwd_phase, fwd_bnd,
        p17_r=rev17, p_phase_r=rev_phase, p_bnd_r=rev_bnd,
        decoder=opts["decoder"],
        len_prior=opts.get("len_prior", True),
        rescue=opts.get("rescue", True),
        **opts.get("config", {}),
    )


def _gene_start(gene) -> int:
    return gene.start if hasattr(gene, "start") else gene["start"]


def _gene_strand(gene) -> str:
    return gene.strand if hasattr(gene, "strand") else gene["strand"]


def run_decode(args, probs_dir: Path, raw_out: Path) -> None:
    seqs = decode_core.load_fasta(args.genome_fasta)
    sources = {}
    for f in sorted(probs_dir.glob("*.npz")):
        z = np.load(f)
        name = str(z["contig_name"]) if "contig_name" in z.files else f.stem
        sources[name] = str(f)
    lookup = {k: k for k in sources}
    lookup.update({k.replace("|", "_"): k for k in sources})

    opts = {
        "decoder": args.decoder,
        "len_prior": True,
        "rescue": True,
        "length_priors": getattr(args, "length_priors", None),
        "config": {
            "prior": args.prior,
            "intergenic_boost": args.intergenic_boost,
            "viterbi_alpha": args.viterbi_alpha,
            "genic_threshold": args.genic_threshold,
            "single_exon_prob": args.single_exon_prob,
            "single_exon_min_cds": args.single_exon_min_cds,
            "repair_w5": args.repair_w5,
            "repair_w3": args.repair_w3,
            "repair_start_rule": args.repair_start_rule,
            "orf_max_shift": args.orf_max_shift,
        },
    }

    tasks = []
    for name, seq in seqs.items():
        path = sources.get(lookup.get(name, ""))
        if path is None:
            logger.warning("no probability npz for contig %s -- skipped", name)
            continue
        tasks.append((name, seq, path, opts))
    if args.processes > 1 and len(tasks) > 1:
        with Pool(args.processes) as pool:
            results = pool.map(_decode_contig_task, tasks)
    else:
        results = [_decode_contig_task(t) for t in tasks]

    n_genes = 0
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_out, "w") as fh:
        fh.write("##gff-version 3\n")
        for (name, _, _, _), genes in zip(tasks, results):
            for g in sorted(genes, key=_gene_start):
                n_genes += 1
                gid = f"{args.source}_{name}_g{n_genes}"
                for ln in decode_core.gene_to_gff_lines(
                        name, g, gid, args.source, _gene_strand(g)):
                    fh.write(ln + "\n")
    logger.info("decoded %d genes -> %s", n_genes, raw_out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome-fasta", required=True,
                    help="input genome FASTA (plain or bgzip/gzip)")
    ap.add_argument("--model-path", default="models/FungiHelixSeek-Anno",
                    help="FungiHelixSeek-Anno checkpoint dir (config.json + "
                         "model.safetensors + *.py architecture code)")
    ap.add_argument("--phase-head",
                    default=str(DEFAULT_PHASE_HEAD),
                    help="npz with the 4-class phase head (weight/bias) used "
                         "by the HMM decoder; 'none' disables it (uniform "
                         "phase prior, lower decode quality)")
    ap.add_argument("--length-priors", default=None,
                    help="npz with intron/exon length-prior tables "
                         "(default: the tables shipped in decoder/)")
    ap.add_argument("--out", required=True, help="output GFF3 path")
    ap.add_argument("--probs-dir", default=None,
                    help="probability npz cache dir (default <out>.probs)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2048,
                    help="window step; < window-size enables overlap "
                         "averaging (recommended 2048)")
    ap.add_argument("--window-size", type=int, default=8192,
                    help="inference window (= model max positions)")
    ap.add_argument("--decoder", choices=("repair", "viterbi", "orf", "hmm"),
                    default="repair",
                    help="repair | viterbi | orf use the Plant decoder; hmm is "
                         "the legacy Fungi HMM (default repair)")
    ap.add_argument("--prior", choices=("fungi", "plant"), default="fungi",
                    help="transition prior for the Plant decoders (default fungi)")
    ap.add_argument("--intergenic-boost", type=float, default=0.0,
                    help="Plant Viterbi intergenic score boost (default 0)")
    ap.add_argument("--viterbi-alpha", type=float, default=0.0,
                    help="Plant Viterbi transition smoothing alpha (default 0)")
    ap.add_argument("--genic-threshold", type=float, default=0.25,
                    help="Plant Viterbi genic probability threshold (default 0.25)")
    ap.add_argument("--single-exon-prob", type=float, default=0.7,
                    help="minimum confidence for a single-exon gene (default 0.7)")
    ap.add_argument("--single-exon-min-cds", type=int, default=200,
                    help="minimum CDS length for a single-exon gene (default 200)")
    ap.add_argument("--repair-w5", type=int, default=300,
                    help="repair 5' extension window in bp (default 300)")
    ap.add_argument("--repair-w3", type=int, default=300,
                    help="repair 3' extension window in bp (default 300)")
    ap.add_argument("--repair-start-rule",
                    choices=("nearest", "first", "longest"), default="nearest",
                    help="repair start-codon selection rule (default nearest)")
    ap.add_argument("--orf-max-shift", type=int, default=3,
                    help="maximum ORF frame shift for the orf decoder (default 3)")
    ap.add_argument("--pad", type=int, default=None,
                    help="gene-span padding bp, gap-capped between neighbours "
                         "(default 200 for --decoder hmm, else 0)")
    ap.add_argument("--device", default="auto",
                    help="auto | cuda:0 | mps | cpu (default auto)")
    ap.add_argument("--processes", type=int, default=5,
                    help="decode worker processes (per contig)")
    ap.add_argument("--source", default="FungiHelixSeek",
                    help="GFF3 source column")
    ap.add_argument("--skip-predict", action="store_true",
                    help="reuse cached probability npz in --probs-dir")
    ap.add_argument("--skip-decode", action="store_true",
                    help="stop after caching probabilities (no GFF3)")
    args = ap.parse_args()

    if args.stride > args.window_size:
        ap.error("--stride must be <= --window-size")
    if args.pad is None:
        args.pad = 200 if args.decoder == "hmm" else 0

    probs_dir = Path(args.probs_dir) if args.probs_dir else \
        Path(str(args.out) + ".probs")
    raw_out = Path(str(args.out) + ".raw")

    if not args.skip_predict:
        if not Path(args.model_path).exists():
            ap.error(f"--model-path {args.model_path} does not exist "
                     "(see README section 1. Download)")
        run_predict(args, probs_dir)
    if args.skip_decode:
        logger.info("probabilities cached in %s; decode skipped", probs_dir)
        return

    apply_length_priors(Path(args.length_priors) if args.length_priors else None)
    run_decode(args, probs_dir, raw_out)
    extend_core.pad_gene_spans(gff3=str(raw_out), pad=args.pad, out=str(args.out))
    raw_out.unlink()
    logger.info("final GFF3 -> %s", args.out)


if __name__ == "__main__":
    main()
