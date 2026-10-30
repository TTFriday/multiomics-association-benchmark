# -*- coding: utf-8 -*-
"""Benchmark heterogeneous graph: corrected parser (chromosome indexing,
CHH coverage, unknown methylation contexts, MAF taken from retained GWAS edges),
plus a JSON audit of inputs and graph statistics.

Consumes the relation tables in data_cache/ produced by src.data.loaders and
writes data_cache/graph_v2.pt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

HERE = Path(__file__).resolve().parent
# Locate the project root: walk up until a directory containing src/ and data_cache/ is found.
ROOT = HERE
while not (ROOT / "data_cache").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
CACHE = ROOT / "data_cache"
OUT = CACHE / "graph_v2.pt"
AUDIT = ROOT / "results" / "benchmark" / "graph_audit.json"

MAX_EDGES = {"ld": 100_000}
DEFAULT_CAP = 200_000

REL = {
    "gene_pheno":   ("gene", "assoc", "phenotype"),
    "snp_pheno":    ("snp", "gwas", "phenotype"),
    "snp_gene":     ("snp", "annot", "gene"),
    "snp_gene_cis": ("snp", "cis", "gene"),
    "snp_gene_trans": ("snp", "trans", "gene"),
    "snp_env":      ("snp", "env_qtl", "env"),
    "snp_snp":      ("snp", "ld", "snp"),
    "meth_pheno":   ("meth", "ewas", "phenotype"),
    "meth_gene":    ("meth", "emqtl", "gene"),
    "meth_pathway": ("meth", "epathway", "pathway"),
    "pheno_pathway": ("phenotype", "ppathway", "pathway"),
}
TYPES = ["gene", "phenotype", "snp", "meth", "pathway", "env"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_edge_table(name: str) -> pd.DataFrame:
    pkl = CACHE / f"{name}.pkl"
    df = pd.read_pickle(pkl)
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce").fillna(1.0)
    corr = pd.to_numeric(df["corr"], errors="coerce") if "corr" in df else pd.Series(np.zeros(len(df)))
    df["corr"] = corr.fillna(0.0)
    df = df.sort_values("strength", ascending=False).drop_duplicates(["src", "dst"], keep="first")
    cap = MAX_EDGES.get(name if name != "snp_snp" else "ld", DEFAULT_CAP)
    return df.head(cap)


def parse_meth_site_v2(s: str):
    """Return (chromosome, position, context); unknown contexts get 'unknown'
    instead of defaulting to CG."""
    m = re.match(r"^([1-5])_(\d+)_(CG|CHG|CHH)", s.upper())
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None, None, "unknown"


def build_v2():
    tables = {k: load_edge_table(k) for k in REL}

    # ---- node sets ----
    nodes = {t: set() for t in TYPES}
    for name, (s, r, d) in REL.items():
        df = tables[name]
        nodes[s] |= set(df["src"])
        nodes[d] |= set(df["dst"])
    idx = {t: {v: i for i, v in enumerate(sorted(nodes[t]))} for t in TYPES}
    print({t: len(nodes[t]) for t in TYPES})

    data = HeteroData()
    for t in TYPES:
        data[t].num_nodes = len(idx[t])

    for name, (s, r, d) in REL.items():
        df = tables[name]
        si = df["src"].map(idx[s]).to_numpy()
        di = df["dst"].map(idx[d]).to_numpy()
        ei = torch.tensor(np.stack([si, di]), dtype=torch.long)
        ea = torch.tensor(np.stack([
            np.sign(df["corr"]).to_numpy(dtype="float32"),
            np.log1p(df["strength"].to_numpy(dtype="float32")),
            (df["corr"] != 0).to_numpy(dtype="float32"),
        ], axis=1))
        data[s, r, d].edge_index = ei
        data[s, r, d].edge_attr = ea
        data[s, r, d].n_orig = len(df)
        if name != "gene_pheno":
            data[d, f"rev_{r}", s].edge_index = torch.tensor(np.stack([di, si]), dtype=torch.long)
            data[d, f"rev_{r}", s].edge_attr = ea.clone()

    # ---- node features ----
    def src_count(src_type, rel, dst_type):
        e = data[src_type, rel, dst_type]
        return np.bincount(e.edge_index[0].numpy(), minlength=data[src_type].num_nodes).astype("float32")

    def dst_count(src_type, rel, dst_type):
        e = data[src_type, rel, dst_type]
        return np.bincount(e.edge_index[1].numpy(), minlength=data[dst_type].num_nodes).astype("float32")

    for t in TYPES:
        n = data[t].num_nodes
        F = []
        if t == "gene":
            F += [np.log1p(dst_count("snp", "annot", "gene")),
                  np.log1p(dst_count("snp", "cis", "gene")),
                  np.log1p(dst_count("snp", "trans", "gene")),
                  np.log1p(dst_count("meth", "emqtl", "gene"))]
            st = np.zeros(n); pos = np.zeros(n); tot = np.zeros(n)
            for et in data.edge_types:
                if et[2] != "gene":
                    continue
                e = data[et]
                di = e.edge_index[1].numpy()
                wv = e.edge_attr[:, 1].numpy(); sg = e.edge_attr[:, 0].numpy()
                np.add.at(st, di, wv * sg); np.add.at(tot, di, wv)
                np.add.at(pos, di, np.maximum(sg, 0) * wv)
            F += [st / np.maximum(tot, 1), np.log1p(tot)]
        elif t == "phenotype":
            F += [np.log1p(dst_count("snp", "gwas", "phenotype")),
                  np.log1p(dst_count("meth", "ewas", "phenotype")),
                  np.log1p(src_count("phenotype", "ppathway", "pathway"))]
        elif t == "snp":
            arr = np.zeros((n, 9), dtype="float32")  # extra column flags unknown chromosomes
            pos_arr = np.zeros(n, dtype="float32")
            maf = np.zeros(n, dtype="float32")
            chr_ = np.zeros(n, dtype="int64")
            id2s = {i: v for v, i in idx["snp"].items()}
            unknown_count = 0
            for i, v in id2s.items():
                m = re.match(r"^chr([1-5cm])_(\d+)$", v, re.I)
                if m:
                    c = m.group(1).upper()
                    if c in "CM":
                        chr_[i] = {"C": 0, "M": 1}[c]
                    else:
                        chr_[i] = int(c)
                    pos_arr[i] = int(m.group(2))
                else:
                    unknown_count += 1
                    arr[i, 8] = 1.0  # unknown-chromosome flag
            arr[np.arange(n), np.clip(chr_, 0, 4)] = 1.0
            arr[:, 5] = np.log1p(pos_arr) / 20.0
            # MAF is taken only from the retained GWAS edges
            gdf = tables["snp_pheno"].copy()
            if "maf" in gdf:
                mm = gdf.groupby("src")["maf"].max()
                for k, v in mm.items():
                    if k in idx["snp"]:
                        maf[idx["snp"][k]] = float(v) if np.isfinite(v) else 0.0
            arr[:, 6] = np.log1p(maf * 1e4)
            arr[:, 7] = np.log1p(src_count("snp", "gwas", "phenotype") + src_count("snp", "annot", "gene"))
            F = [arr]
        elif t == "meth":
            arr = np.zeros((n, 10), dtype="float32")  # extra column flags unknown contexts
            id2m = {i: v for v, i in idx["meth"].items()}
            unknown_ctx = 0
            for i, v in id2m.items():
                c, p, ct = parse_meth_site_v2(v)
                if c:
                    arr[i, max(0, c - 1)] = 1.0
                    arr[i, 5] = np.log1p(p) / 20.0
                if ct == "unknown":
                    unknown_ctx += 1
                    arr[i, 9] = 1.0
                else:
                    arr[i, 6 + {"CG": 0, "CHG": 1, "CHH": 2}[ct]] = 1.0
            arr[:, 8] = np.log1p(src_count("meth", "emqtl", "gene") + src_count("meth", "ewas", "phenotype")
                                 + src_count("meth", "epathway", "pathway"))
            F = [arr]
        elif t == "pathway":
            F += [np.log1p(dst_count("meth", "epathway", "pathway")),
                  np.log1p(dst_count("phenotype", "ppathway", "pathway"))]
        elif t == "env":
            F += [np.log1p(dst_count("snp", "env_qtl", "env"))]
        X = np.stack([np.asarray(f, dtype="float32").reshape(n, -1) for f in F], axis=1)
        X = X.reshape(n, -1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        data[t].x = torch.tensor(X)

    data.node_index = {t: idx[t] for t in TYPES}
    torch.save(data, OUT)

    # ---- audit ----
    stats = {f"{s}-{r}-{d}": int(data[s, r, d].edge_index.size(1)) for s, r, d in data.edge_types}
    stats.update({t: int(data[t].num_nodes) for t in TYPES})
    stats["snp_unknown_chr"] = unknown_count
    stats["meth_unknown_context"] = unknown_ctx
    stats["gene_feature_dim"] = int(data["gene"].x.size(1))
    stats["snp_feature_dim"] = int(data["snp"].x.size(1))
    stats["meth_feature_dim"] = int(data["meth"].x.size(1))

    audit = {
        "graph_version": "v2",
        "output_path": str(OUT.relative_to(ROOT)) if OUT.is_relative_to(ROOT) else str(OUT),
        "output_sha256": sha256(OUT),
        "input_files": {name: sha256(CACHE / f"{name}.pkl") for name in REL},
        "stats": stats,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return data, audit


if __name__ == "__main__":
    build_v2()
