# -*- coding: utf-8 -*-
"""Missing-evidence robustness: retrain after deleting an entire auxiliary
relation family (GWAS, methylation links, pathway links, or environment QTLs)
to measure how much each evidence family contributes."""
from __future__ import annotations
import argparse, json, copy, random, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.benchmark.train_unified import (
    load_graph_v2, split_data_v2, train_one, evaluate_model, set_seed, RESULTS, MODELS
)

def remove_edges(data, condition):
    # Relation names must match the actual edge types of graph_v2.pt; the
    # guards below prevent a silently empty removal.
    remove_map = {
        "no_gwas": [("snp", "gwas", "phenotype"), ("phenotype", "rev_gwas", "snp")],
        "no_meth": [("meth", "ewas", "phenotype"), ("phenotype", "rev_ewas", "meth"),
                    ("meth", "emqtl", "gene"), ("gene", "rev_emqtl", "meth"),
                    ("meth", "epathway", "pathway"), ("pathway", "rev_epathway", "meth")],
        "no_pathway": [("meth", "epathway", "pathway"), ("pathway", "rev_epathway", "meth"),
                       ("phenotype", "ppathway", "pathway"), ("pathway", "rev_ppathway", "phenotype")],
        "no_env": [("snp", "env_qtl", "env"), ("env", "rev_env_qtl", "snp")],
    }
    to_remove = set(remove_map[condition])
    types = set(data.edge_types)
    missing = to_remove - types
    if missing:
        raise ValueError(f"{condition}: edge types not present in graph: {sorted(missing)}")
    n_before = sum(data[et].edge_index.size(1) for et in data.edge_types)
    for et in list(data.edge_types):
        if et in to_remove:
            del data[et]
    n_after = sum(data[et].edge_index.size(1) for et in data.edge_types)
    if n_after >= n_before:
        raise ValueError(f"{condition}: no edges were removed (before={n_before}, after={n_after})")
    print(f"[{condition}] removed {n_before - n_after} edges "
          f"({n_before} -> {n_after})", flush=True)
    return data

def run_missing(model_name, seed, dev_seed, lr, wd, epochs, condition, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    data = load_graph_v2()
    data = remove_edges(data, condition)
    data = split_data_v2(data, dev_seed)
    data = data.to(device)
    model_cls = MODELS[model_name]
    model = model_cls(data, hidden=64, layers=2, dropout=0.2)
    model, train_info = train_one(model, data, epochs=epochs, lr=lr, wd=wd, device=device)
    metrics = evaluate_model(model, data, device=device)
    result = {"model": model_name, "seed": seed, "dev_seed": dev_seed, "lr": lr, "wd": wd,
              "epochs": epochs, "condition": condition, "device": device, **train_info,
              "val": metrics["val"], "test": metrics["test"],
              "val_binary": metrics["val_binary"], "test_binary": metrics["test_binary"],
              "frequency": metrics["frequency"]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{model_name}_seed{seed}_dev{dev_seed}_{condition}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="full", choices=list(MODELS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dev-seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--wd", type=float, default=0.0005)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--condition", required=True, choices=["no_gwas", "no_meth", "no_pathway", "no_env"])
    args = ap.parse_args()
    result = run_missing(args.model, args.seed, args.dev_seed or args.seed,
                        args.lr, args.wd, args.epochs, args.condition)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
