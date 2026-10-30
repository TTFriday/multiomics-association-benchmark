# -*- coding: utf-8 -*-
"""Batch runner for the benchmark experiments: hyperparameter development
grid, formal ten-seed comparison, evidence permutation, and missing-evidence
robustness.

Usage: python -m experiments.benchmark.run_all --phase dev     # development grid, 72 fits
      python -m experiments.benchmark.run_all --phase formal  # formal comparison, 60 fits
      python -m experiments.benchmark.run_all --phase e2      # evidence permutation, 30 fits
      python -m experiments.benchmark.run_all --phase e3      # missing-evidence robustness, 40 fits
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "data_cache").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
RESULTS = ROOT / "results" / "benchmark"
LOG = RESULTS / "run_log.jsonl"

MODELS = ["full", "no_info", "hgt", "simplehgn", "distmult", "mlp"]
DEV_SEEDS = [91, 92]
FORMAL_SEEDS = list(range(101, 111))
LR_GRID = [0.001, 0.005]
WD_GRID = [0.0, 0.00001, 0.0005]


def run_cmd(cmd, cwd=None):
    """Run a subprocess command."""
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def log_run(record):
    """Append a run record to the JSONL log."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def result_complete(path):
    """A run is complete when its result file exists and contains the
    stratified evaluation fields; completed runs can be skipped, which makes
    the batch resumable."""
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            rec = json.load(f)
        return rec.get("test", {}).get("head_macro_NDCG@50") is not None
    except Exception:
        return False


def run_dev_grid():
    """Development grid: 6 models x 6 configurations x 2 development seeds = 72 fits."""
    print("=" * 60, flush=True)
    print("Development Grid: 72 fits", flush=True)
    print("=" * 60, flush=True)
    total = len(MODELS) * len(LR_GRID) * len(WD_GRID) * len(DEV_SEEDS)
    count = 0
    for model in MODELS:
        for lr in LR_GRID:
            for wd in WD_GRID:
                for dev_seed in DEV_SEEDS:
                    count += 1
                    print(f"\n[{count}/{total}] {model} lr={lr} wd={wd} dev_seed={dev_seed}", flush=True)
                    cmd = [
                        sys.executable, "-m", "experiments.benchmark.train_unified",
                        "--model", model,
                        "--seed", str(dev_seed),
                        "--dev-seed", str(dev_seed),
                        "--lr", str(lr),
                        "--wd", str(wd),
                        "--epochs", "150",
                    ]
                    start = time.perf_counter()
                    try:
                        run_cmd(cmd)
                        elapsed = time.perf_counter() - start
                        # read back the saved result file
                        result_file = RESULTS / f"{model}_seed{dev_seed}_dev{dev_seed}.json"
                        with result_file.open(encoding="utf-8") as f:
                            result = json.load(f)
                        log_run({
                            "phase": "dev", "model": model, "seed": dev_seed,
                            "lr": lr, "wd": wd, "elapsed": elapsed,
                            "status": "success", "val_ndcg": result.get("val", {}).get("macro_NDCG@50"),
                        })
                        print(f"  [OK] Done in {elapsed:.1f}s, val_NDCG@50={result.get('val', {}).get('macro_NDCG@50'):.4f}", flush=True)
                    except Exception as e:
                        elapsed = time.perf_counter() - start
                        log_run({
                            "phase": "dev", "model": model, "seed": dev_seed,
                            "lr": lr, "wd": wd, "elapsed": elapsed,
                            "status": "failed", "error": str(e),
                        })
                        print(f"  [FAIL] Failed in {elapsed:.1f}s: {e}", flush=True)


def select_best_configs():
    """Select the best configuration per model from the development log."""
    print("\n" + "=" * 60, flush=True)
    print("Selecting best configs from development grid", flush=True)
    print("=" * 60, flush=True)
    dev_results = []
    if LOG.exists():
        with LOG.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("phase") == "dev" and rec.get("status") == "success":
                    dev_results.append(rec)

    best = {}
    for model in MODELS:
        model_devs = [r for r in dev_results if r["model"] == model]
        if not model_devs:
            print(f"  {model}: no successful dev runs", flush=True)
            continue
        # rank configurations by mean validation NDCG over the two development seeds
        config_scores = {}
        for r in model_devs:
            key = (r["lr"], r["wd"])
            config_scores.setdefault(key, []).append(r.get("val_ndcg") or 0)
        config_means = {k: sum(v) / len(v) for k, v in config_scores.items()}
        # ties are broken towards the smaller lr, then the smaller wd (kept simple: max mean)
        best_key = max(config_means, key=config_means.get)
        best[model] = {"lr": best_key[0], "wd": best_key[1], "mean_val_ndcg": config_means[best_key]}
        print(f"  {model}: lr={best_key[0]}, wd={best_key[1]}, mean_val_NDCG@50={config_means[best_key]:.4f}", flush=True)
    return best


def run_formal(best_configs):
    """Formal comparison: 6 models x 10 formal seeds = 60 fits."""
    print("\n" + "=" * 60, flush=True)
    print("Formal Comparison: 60 fits", flush=True)
    print("=" * 60, flush=True)
    total = len(MODELS) * len(FORMAL_SEEDS)
    count = 0
    for model in MODELS:
        cfg = best_configs.get(model, {"lr": 0.001, "wd": 0.0})
        for seed in FORMAL_SEEDS:
            if result_complete(RESULTS / f"{model}_seed{seed}.json"):
                print(f"[skip] {model} seed={seed} already complete", flush=True)
                continue
            count += 1
            print(f"\n[{count}/{total}] {model} seed={seed} lr={cfg['lr']} wd={cfg['wd']}", flush=True)
            cmd = [
                sys.executable, "-m", "experiments.benchmark.train_unified",
                "--model", model,
                "--seed", str(seed),
                "--lr", str(cfg["lr"]),
                "--wd", str(cfg["wd"]),
                "--epochs", "150",
            ]
            start = time.perf_counter()
            try:
                run_cmd(cmd)
                elapsed = time.perf_counter() - start
                # read back the saved result file
                result_file = RESULTS / f"{model}_seed{seed}.json"
                with result_file.open(encoding="utf-8") as f:
                    result = json.load(f)
                log_run({
                    "phase": "formal", "model": model, "seed": seed,
                    "lr": cfg["lr"], "wd": cfg["wd"], "elapsed": elapsed,
                    "status": "success",
                    "test_ndcg": result.get("test", {}).get("macro_NDCG@50"),
                    "test_auroc": result.get("test_binary", {}).get("macro_AUROC"),
                })
                print(f"  [OK] Done in {elapsed:.1f}s, test_NDCG@50={result.get('test', {}).get('macro_NDCG@50'):.4f}", flush=True)
            except Exception as e:
                elapsed = time.perf_counter() - start
                log_run({
                    "phase": "formal", "model": model, "seed": seed,
                    "lr": cfg["lr"], "wd": cfg["wd"], "elapsed": elapsed,
                    "status": "failed", "error": str(e),
                })
                print(f"  [FAIL] Failed in {elapsed:.1f}s: {e}", flush=True)


def run_e2_permutations(best_configs):
    """Evidence permutation: 3 conditions x 10 seeds = 30 fits."""
    print("\n" + "=" * 60, flush=True)
    print("Evidence Permutation: 30 fits", flush=True)
    print("=" * 60, flush=True)
    modes = ["sign_permute", "strength_permute", "full_permute"]
    formal_seeds = list(range(101, 111))
    total = len(modes) * len(formal_seeds)
    count = 0
    for mode in modes:
        for seed in formal_seeds:
            if result_complete(RESULTS / f"full_seed{seed}_dev{seed}_{mode}.json"):
                print(f"[skip] full {mode} seed={seed} already complete", flush=True)
                continue
            count += 1
            cfg = best_configs.get("full", {"lr": 0.001, "wd": 0.00001})
            print(f"\n[{count}/{total}] full {mode} seed={seed}", flush=True)
            cmd = [
                sys.executable, "-m", "experiments.benchmark.permute_evidence",
                "--model", "full",
                "--seed", str(seed),
                "--dev-seed", str(seed),
                "--lr", str(cfg["lr"]),
                "--wd", str(cfg["wd"]),
                "--epochs", "150",
                "--mode", mode,
            ]
            start = time.perf_counter()
            try:
                run_cmd(cmd)
                elapsed = time.perf_counter() - start
                result_file = RESULTS / f"full_seed{seed}_dev{seed}_{mode}.json"
                with result_file.open(encoding="utf-8") as f:
                    result = json.load(f)
                log_run({
                    "phase": "e2", "model": "full", "seed": seed,
                    "mode": mode, "elapsed": elapsed, "status": "success",
                    "test_ndcg": result.get("test", {}).get("macro_NDCG@50"),
                })
                print(f"  [OK] Done in {elapsed:.1f}s", flush=True)
            except Exception as e:
                elapsed = time.perf_counter() - start
                log_run({
                    "phase": "e2", "model": "full", "seed": seed,
                    "mode": mode, "elapsed": elapsed,
                    "status": "failed", "error": str(e),
                })
                print(f"  [FAIL] Failed in {elapsed:.1f}s: {e}", flush=True)


def run_e3_missing(best_configs):
    """Missing-evidence robustness: 4 conditions x 10 seeds = 40 fits."""
    print("\n" + "=" * 60, flush=True)
    print("Missing Evidence Robustness: 40 fits", flush=True)
    print("=" * 60, flush=True)
    conditions = ["no_gwas", "no_meth", "no_pathway", "no_env"]
    formal_seeds = list(range(101, 111))
    total = len(conditions) * len(formal_seeds)
    count = 0
    for condition in conditions:
        for seed in formal_seeds:
            if result_complete(RESULTS / f"full_seed{seed}_dev{seed}_{condition}.json"):
                print(f"[skip] full {condition} seed={seed} already complete", flush=True)
                continue
            count += 1
            cfg = best_configs.get("full", {"lr": 0.001, "wd": 0.00001})
            print(f"\n[{count}/{total}] full {condition} seed={seed}", flush=True)
            cmd = [
                sys.executable, "-m", "experiments.benchmark.missing_evidence",
                "--model", "full",
                "--seed", str(seed),
                "--dev-seed", str(seed),
                "--lr", str(cfg["lr"]),
                "--wd", str(cfg["wd"]),
                "--epochs", "150",
                "--condition", condition,
            ]
            start = time.perf_counter()
            try:
                run_cmd(cmd)
                elapsed = time.perf_counter() - start
                result_file = RESULTS / f"full_seed{seed}_dev{seed}_{condition}.json"
                with result_file.open(encoding="utf-8") as f:
                    result = json.load(f)
                log_run({
                    "phase": "e3", "model": "full", "seed": seed,
                    "condition": condition, "elapsed": elapsed, "status": "success",
                    "test_ndcg": result.get("test", {}).get("macro_NDCG@50"),
                })
                print(f"  [OK] Done in {elapsed:.1f}s", flush=True)
            except Exception as e:
                elapsed = time.perf_counter() - start
                log_run({
                    "phase": "e3", "model": "full", "seed": seed,
                    "condition": condition, "elapsed": elapsed,
                    "status": "failed", "error": str(e),
                })
                print(f"  [FAIL] Failed in {elapsed:.1f}s: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["dev", "formal", "e2", "e3", "all"])
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.phase in ("dev", "all"):
        run_dev_grid()

    if args.phase in ("formal", "all"):
        best = select_best_configs()
        run_formal(best)

    if args.phase in ("e2", "all"):
        best = select_best_configs()
        run_e2_permutations(best)

    if args.phase in ("e3", "all"):
        best = select_best_configs()
        run_e3_missing(best)

    print("\n" + "=" * 60, flush=True)
    print("Phase complete. Check run_log.jsonl for details.", flush=True)
    print("=" * 60, flush=True)
