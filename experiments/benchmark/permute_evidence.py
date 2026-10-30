# -*- coding: utf-8 -*-
"""Evidence-attribute permutation: retrain with edge signs flipped randomly,
edge strengths shuffled, or both, to test how much the model relies on the
evidence attributes attached to auxiliary edges."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.benchmark.train_unified import (
    load_graph_v2, split_data_v2, train_one, evaluate_model, set_seed, RESULTS, MODELS
)

def permute_edge_attr(data, mode, seed):
    rng = np.random.RandomState(seed)
    for et in data.edge_types:
        store = data[et]
        if not hasattr(store, 'edge_attr') or store.edge_attr is None or store.edge_attr.size(0) == 0:
            continue
        ea = store.edge_attr.clone()
        if mode == "sign_permute":
            signs = ea[:, 0].clone()
            nz = signs != 0
            n = nz.sum().item()
            if n > 0:
                flips = torch.from_numpy(rng.choice([-1, 1], size=n)).float().to(signs.device)
                ea[nz, 0] = signs[nz] * flips
        elif mode == "strength_permute":
            strengths = ea[:, 1].clone()
            perm = torch.from_numpy(rng.permutation(len(strengths))).long()
            ea[:, 1] = strengths[perm]
        elif mode == "full_permute":
            signs = ea[:, 0].clone()
            strengths = ea[:, 1].clone()
            nz = signs != 0
            n = nz.sum().item()
            if n > 0:
                flips = torch.from_numpy(rng.choice([-1, 1], size=n)).float().to(signs.device)
                ea[nz, 0] = signs[nz] * flips
            perm = torch.from_numpy(rng.permutation(len(strengths))).long()
            ea[:, 1] = strengths[perm]
        store.edge_attr = ea
    return data

def run_permutation(model_name, seed, dev_seed, lr, wd, epochs, mode, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    data = load_graph_v2()
    data = permute_edge_attr(data, mode, seed)
    data = split_data_v2(data, dev_seed)
    data = data.to(device)
    model_cls = MODELS[model_name]
    model = model_cls(data, hidden=64, layers=2, dropout=0.2)
    model, train_info = train_one(model, data, epochs=epochs, lr=lr, wd=wd, device=device)
    metrics = evaluate_model(model, data, device=device)
    result = {"model": model_name, "seed": seed, "dev_seed": dev_seed, "lr": lr, "wd": wd,
              "epochs": epochs, "mode": mode, "device": device, **train_info,
              "val": metrics["val"], "test": metrics["test"],
              "val_binary": metrics["val_binary"], "test_binary": metrics["test_binary"],
              "frequency": metrics["frequency"]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{model_name}_seed{seed}_dev{dev_seed}_{mode}.json"
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
    ap.add_argument("--mode", required=True, choices=["sign_permute", "strength_permute", "full_permute"])
    args = ap.parse_args()
    result = run_permutation(args.model, args.seed, args.dev_seed or args.seed,
                            args.lr, args.wd, args.epochs, args.mode)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
