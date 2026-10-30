# -*- coding: utf-8 -*-
"""Unified training protocol: numerically stable BCE, checkpoint selection by
validation macro NDCG@50, and full audit fields in every saved run.

Usage: python -m experiments.benchmark.train_unified --model full --seed 101 --dev_seed 91 --lr 0.001 --wd 0.00001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.benchmark.build_graph_v2 import build_v2, OUT as GRAPH_PATH
from experiments.benchmark.models_new import MODELS, mp_edge_types
from experiments.benchmark.evaluate_unified import (
    standard_ranking_metrics, binary_metrics_by_phenotype, frequency_group_summary
)

HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "data_cache").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
RESULTS = ROOT / "results" / "benchmark"
G2P = ("gene", "assoc", "phenotype")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph_v2():
    """Load the rebuilt graph without caching splits; every call returns an
    unsplit copy."""
    if not GRAPH_PATH.exists():
        print("Graph v2 not found, building...")
        build_v2()
    data = torch.load(GRAPH_PATH, weights_only=False)
    # drop any stale split attributes
    store = data[G2P]
    for attr in ["train_mask", "valid_mask", "test_mask", "neg_train", "neg_valid", "neg_test"]:
        if hasattr(store, attr):
            delattr(store, attr)
    return data


def split_data_v2(data, seed, ratios=(0.8, 0.1, 0.1), neg_ratio=10):
    """Revised split protocol: target positives are deduplicated first, then
    partitioned by a fixed permutation; negatives are sampled without
    replacement and never overlap across splits."""
    rng = np.random.default_rng(seed)
    ei = data[G2P].edge_index.numpy()
    E = ei.shape[1]
    perm = rng.permutation(E)
    n_tr = int(E * ratios[0])
    n_va = int(E * ratios[1])
    tr_idx, va_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    nG, nP = int(data["gene"].num_nodes), int(data["phenotype"].num_nodes)
    pos_mask = np.zeros((nG, nP), dtype=bool)
    pos_mask[ei[0], ei[1]] = True

    def sample_neg(idx_pos, k, exclude_mask=None):
        ph = ei[1, idx_pos]
        w = np.bincount(ph, minlength=nP).astype(np.float64)
        w = w / w.sum()
        phn = rng.choice(nP, size=k, p=w)
        gn = rng.integers(0, nG, size=k)
        bad = pos_mask[gn, phn]
        if exclude_mask is not None:
            bad = bad | exclude_mask[gn, phn]
        tries = 0
        while bad.any() and tries < 20:
            fix = np.where(bad)[0]
            gn[fix] = rng.integers(0, nG, size=len(fix))
            bad = pos_mask[gn, phn]
            if exclude_mask is not None:
                bad = bad | exclude_mask[gn, phn]
            tries += 1
        keep = ~bad
        return np.stack([gn[keep], phn[keep]])

    ntr = len(tr_idx) * neg_ratio
    neg_tr = sample_neg(tr_idx, ntr)
    neg_va = sample_neg(va_idx, len(va_idx) * neg_ratio)
    neg_te = sample_neg(te_idx, len(te_idx) * neg_ratio)

    data[G2P].train_mask = torch.tensor(tr_idx, dtype=torch.long)
    data[G2P].valid_mask = torch.tensor(va_idx, dtype=torch.long)
    data[G2P].test_mask = torch.tensor(te_idx, dtype=torch.long)
    data[G2P].neg_train = torch.tensor(neg_tr, dtype=torch.long)
    data[G2P].neg_valid = torch.tensor(neg_va, dtype=torch.long)
    data[G2P].neg_test = torch.tensor(neg_te, dtype=torch.long)
    return data


def train_one(model, data, epochs=150, lr=0.001, wd=0.0, device="cpu", check_interval=25):
    """Train one model; the checkpoint is the epoch with the best validation
    macro NDCG@50."""
    model = model.to(device)
    started = time.perf_counter()
    store = data[G2P]
    tr = store.train_mask.to(device)
    pos_tr = store.edge_index[:, tr]
    neg_tr = store.neg_train.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best = {"ndcg": -1, "state": None, "epoch": 0}
    validation_history = []

    for ep in range(epochs):
        model.train()
        S = model.score_all()
        logits = model.logits
        # numerically stable BCE: log_sigmoid for positives, log_sigmoid(-logit) for negatives
        pos_logit = logits[pos_tr[0], pos_tr[1]]
        neg_logit = logits[neg_tr[0], neg_tr[1]]
        loss = -(Fn.logsigmoid(pos_logit).mean() + Fn.logsigmoid(-neg_logit).mean())

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if (ep + 1) % check_interval == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                Sv = model.score_all()
                val_logits = model.logits
                # validation: candidates = G - train_pos, evaluated on valid_pos
                m = standard_ranking_metrics(
                    val_logits,
                    store.edge_index[:, store.train_mask],
                    torch.zeros(2, 0, dtype=torch.long),  # valid positives are not excluded here
                    store.edge_index[:, store.valid_mask],
                )
            ndcg = m.get("macro_NDCG@50", 0) or 0
            validation_history.append({"epoch": ep + 1, "macro_NDCG@50": ndcg})
            if ndcg > best["ndcg"]:
                best = {"ndcg": ndcg,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                        "epoch": ep + 1}

    if best["state"]:
        model.load_state_dict(best["state"])
    elapsed = time.perf_counter() - started
    return model, {
        "best_epoch": best["epoch"],
        "best_val_macro_NDCG@50": best["ndcg"],
        "training_seconds": elapsed,
        "validation_history": validation_history,
    }


def evaluate_model(model, data, device="cpu"):
    """Final evaluation: FP32 logit ranking with macro NDCG@50 as the primary
    endpoint."""
    model = model.to(device)
    model.eval()
    store = data[G2P]
    with torch.no_grad():
        S = model.score_all()
        logits = model.logits.detach().cpu()

    # validation candidates = G - train_pos; test candidates = G - train_pos - valid_pos
    val_metrics = standard_ranking_metrics(
        logits,
        store.edge_index[:, store.train_mask],
        torch.zeros(2, 0, dtype=torch.long),  # valid positives are not excluded for validation
        store.edge_index[:, store.valid_mask],
    )
    test_metrics = standard_ranking_metrics(
        logits,
        store.edge_index[:, store.train_mask],
        store.edge_index[:, store.valid_mask],  # valid positives are excluded for testing
        store.edge_index[:, store.test_mask],
    )

    # binary classification metrics
    val_binary = binary_metrics_by_phenotype(
        logits, store.edge_index[:, store.valid_mask], store.neg_valid
    )
    test_binary = binary_metrics_by_phenotype(
        logits, store.edge_index[:, store.test_mask], store.neg_test
    )

    # head/tail frequency statistics
    freq = frequency_group_summary(
        store.edge_index[:, store.train_mask],
        store.edge_index[:, store.test_mask],
        store.neg_test,
    )

    return {
        "val": val_metrics,
        "test": test_metrics,
        "val_binary": val_binary,
        "test_binary": test_binary,
        "frequency": freq,
        "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
    }


def run_experiment(model_name, seed, dev_seed=None, lr=0.001, wd=0.0, epochs=150, hidden=64, layers=2, dropout=0.2):
    """Run a single experiment and return the result dictionary."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_name}] seed={seed} dev_seed={dev_seed} lr={lr} wd={wd} device={device}")

    data = load_graph_v2()
    if dev_seed is not None:
        # development mode: split by dev_seed
        data_dev = split_data_v2(data, seed=dev_seed)
    else:
        data_dev = split_data_v2(data, seed=seed)
    data_dev = data_dev.to(device)

    model_cls = MODELS[model_name]
    model = model_cls(data_dev, hidden=hidden, layers=layers, dropout=dropout)

    model, train_info = train_one(model, data_dev, epochs=epochs, lr=lr, wd=wd, device=device)
    eval_results = evaluate_model(model, data_dev, device=device)

    result = {
        "model": model_name,
        "seed": seed,
        "dev_seed": dev_seed,
        "lr": lr,
        "wd": wd,
        "epochs": epochs,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "device": device,
        **train_info,
        **eval_results,
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dev-seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    result = run_experiment(
        args.model, args.seed, args.dev_seed,
        args.lr, args.wd, args.epochs, args.hidden, args.layers, args.dropout
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    # save the run
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.model}_seed{args.seed}"
    if args.dev_seed is not None:
        run_id += f"_dev{args.dev_seed}"
    out_path = RESULTS / f"{run_id}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")
