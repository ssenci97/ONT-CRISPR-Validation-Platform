#!/usr/bin/env python3
"""Run CRISPRlungo for knock-in (HDR / insertion) experiments on one FASTQ or a folder of FASTQs.

Wraps the `CRISPRlungo` CLI with named flags. Reads with the exact desired edit are
classified through the induced-sequence input. Threads default to all detected cores.

Inputs:
    --ref: Reference amplicon FASTA (mark UMIs with parentheses if --umi).
    --input: One FASTQ file or a folder of FASTQ files (.fastq/.fq, optionally .gz).
    --target: Guide target sequence (text, must be present in the reference).
    --induced-sequence-path: File describing the desired knock-in edit (required).
    --control: Untreated FASTQ for background filtering, or use --run-simulation instead.
    --integration-file: Optional FASTA of sequences that may integrate (e.g. donor fragments).

Outputs:
    --output: With a single file, the CRISPRlungo result directory itself.
        With a folder input, a parent directory with one subfolder per sample.

Knock-in options:
    --large-induced-insertion, --only-consider-desired-mutation,
    --induced-partial-similarity, --largeins-cutlen, --largedel-cutlen, --mix-tag

Other:
    --window, --cleavage-pos, --additional-target, --whole-window-between-targets,
    --umi, --threads, --just-visualization, plot options.
    --yaml-in: Load parameters from configs/param_configs.yaml (CLI values win).
    --yaml-out: Save resolved parameters to configs/param_configs.yaml.

Example:
    python run_crispr_knockin.py --ref ref/amplicon.fasta --input raw --output results \
        --target GGCACTGCGGCTGGAGGTGG --induced-sequence-path ref/induced.fasta \
        --control ctrl/mock.fastq.gz --yaml-out
"""
import argparse
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

###################################### config
SCRIPT_NAME = Path(__file__).stem
YAML_PATH = Path("configs/param_configs.yaml")
CORES = os.cpu_count() or 1
EXE = "CRISPRlungo"
EXTS = (".fastq", ".fastq.gz", ".fq", ".fq.gz")
REQUIRED = ("ref", "input", "output", "target", "induced_sequence_path")
SIM_CHOICES = ("pacbio", "ont_sup", "ont_hac", "ont_fast")
VALUES = {
    "control": ("--control", str, "untreated FASTQ for background filtering (input)"),
    "run_simulation": ("--run_simulation", str, f"simulate a control with Badread, one of {SIM_CHOICES}; not with --umi"),
    "induced_sequence_path": ("--induced_sequence_path", str, "desired knock-in edit file (input, required)"),
    "induced_mutation_patterns": ("--induced_mutation_patterns", str, "desired mutations file (input)"),
    "integration_file": ("--integration_file", str, "FASTA of possible integrated sequences (input)"),
    "induced_partial_similarity": ("--induced_paritial_similiarity", float, "similarity above which a read counts as partially induced"),
    "large_induced_insertion": ("--large_induced_insertion", str, "validate only flanking anchors of a large knock-in insertion"),
    "largeins_cutlen": ("--largeins_cutlen", int, "minimum length of a large insertion"),
    "largedel_cutlen": ("--largedel_cutlen", int, "minimum length of a large deletion"),
    "window": ("--window", int, "bases around the cleavage site to analyze"),
    "cleavage_pos": ("--cleavage_pos", int, "cut position within the target"),
    "additional_target": ("--additional_target", str, "second guide sequence (dual cut)"),
    "length_min": ("--length_min", int, "mutations longer than this and absent from control are significant"),
    "allele_plot_window": ("--allele_plot_window", int, "window for the allele plot"),
    "remove_large_mutations_in_plot": ("--remove_large_mutations_in_plot", int, "hide deletions of at least this length (>=100) in the allele plot"),
}
BOOLS = {
    "only_consider_desired_mutation": ("--only_consider_desired_mutation", "ignore all mutation patterns except the desired edit"),
    "mix_tag": ("--mix_tag", "label reads with multiple mutations as Complex"),
    "whole_window_between_targets": ("--whole_window_between_targets", "include the whole region between two targets in the window"),
    "show_all_between_allele": ("--show_all_between_allele", "draw all sequences between two targets in the allele plot"),
    "just_visualization": ("--just_visualization", "only redo plots from a previous run"),
    "umi": ("--umi", "UMI sequencing"),
}

###################################### cli
def parse():
    ap = argparse.ArgumentParser(description="CRISPRlungo knock-in wrapper")
    ap.add_argument("--ref", help="reference amplicon FASTA (input)")
    ap.add_argument("--input", help="FASTQ file or folder of FASTQ files (input)")
    ap.add_argument("--output", help="result directory (output)")
    ap.add_argument("--target", help="guide target sequence")
    for k, (_, t, h) in VALUES.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=t, help=h)
    for k, (_, h) in BOOLS.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, action="store_true", help=h)
    ap.add_argument("--threads", type=int, help=f"threads (default: all detected, {CORES})")
    ap.add_argument("--yaml-in", dest="yaml_in", action="store_true", help=f"load parameters from {YAML_PATH}")
    ap.add_argument("--yaml-out", dest="yaml_out", action="store_true", help=f"save parameters to {YAML_PATH}")
    args = ap.parse_args()
    p = vars(args)
    if args.yaml_in:
        cfg = (yaml.safe_load(YAML_PATH.read_text()) or {}).get(SCRIPT_NAME, {})
        p.update({k: v for k, v in cfg.items() if k in p and p[k] in (None, False)})
    p["threads"] = p["threads"] or CORES
    missing = [k for k in REQUIRED if not p[k]]
    if missing:
        ap.error(f"missing parameters: {', '.join(missing)}")
    if p["run_simulation"] and p["run_simulation"] not in SIM_CHOICES:
        ap.error(f"--run-simulation must be one of {SIM_CHOICES}")
    if p["run_simulation"] and p["umi"]:
        ap.error("--run-simulation does not support --umi")
    if not p["control"] and not p["run_simulation"]:
        print("[PYTHON-INFO] WARNING no --control and no --run-simulation: background errors will not be filtered")
    return p

###################################### helpers
def save_yaml(p):
    cfg = (yaml.safe_load(YAML_PATH.read_text()) if YAML_PATH.exists() else None) or {}
    cfg[SCRIPT_NAME] = {k: v for k, v in p.items() if k not in ("yaml_in", "yaml_out")}
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    YAML_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[PYTHON-INFO] parameters saved to {YAML_PATH} under '{SCRIPT_NAME}'")

def sample_name(fq):
    for e in sorted(EXTS, key=len, reverse=True):
        if fq.name.endswith(e):
            return fq.name[: -len(e)]
    return fq.stem

def build_cmd(p, fq, out):
    cmd = [EXE, p["ref"], str(fq), str(out), p["target"], "-t", str(p["threads"])]
    for k, (flag, _, _) in VALUES.items():
        if p[k] is not None:
            cmd += [flag, str(p[k])]
    cmd += [flag for k, (flag, _) in BOOLS.items() if p[k]]
    return cmd

###################################### main
def main():
    t0 = time.time()
    p = parse()
    if p["yaml_out"]:
        save_yaml(p)
    inp, out = Path(p["input"]), Path(p["output"])
    jobs = [(f, out / sample_name(f)) for f in sorted(inp.iterdir()) if f.name.endswith(EXTS)] if inp.is_dir() else [(inp, out)]
    if not jobs:
        sys.exit(f"[PYTHON-INFO] no FASTQ files found in {inp}")
    print(f"[PYTHON-INFO] {len(jobs)} sample(s), {p['threads']} thread(s)")
    failed = []
    for fq, o in tqdm(jobs, desc="samples"):
        o.parent.mkdir(parents=True, exist_ok=True)
        print(f"[PYTHON-INFO] running {fq.name} -> {o}")
        try:
            subprocess.run(build_cmd(p, fq, o), check=True)
        except subprocess.CalledProcessError as e:
            print(f"[PYTHON-INFO] FAILED {fq.name} (exit {e.returncode})")
            failed.append(fq.name)
    mem = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024
    print(f"[PYTHON-INFO] {datetime.now():%Y-%m-%d %H:%M:%S} | {time.time() - t0:.1f}s | {mem:.0f} MB max memory | {len(jobs) - len(failed)}/{len(jobs)} ok")
    if failed:
        sys.exit(f"[PYTHON-INFO] failed samples: {', '.join(failed)}")

if __name__ == "__main__":
    main()

