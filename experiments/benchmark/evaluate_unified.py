# -*- coding: utf-8 -*-
"""Unified evaluation protocol: FP32 logit ranking, validation/test candidate
separation, macro NDCG@50, and deterministic tie handling.
"""
from __future__ import annotations

import numpy as np
import torch


def _finite_scores(logits):
    """Return a finite FP32 score matrix [gene, phenotype]."""
    if isinstance(logits, torch.Tensor):
        if logits.is_complex():
            raise ValueError("logits must be real")
        logits = logits.detach().cpu().to(torch.float32).numpy()
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or not np.issubdtype(logits.dtype, np.number):
        raise ValueError("logits must be a real numeric [gene, phenotype] matrix")
    if not np.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")
    return logits


def _pair_array(pairs, name, shape=None):
    """Validate a [2, N] array of nonnegative integer pairs."""
    if isinstance(pairs, torch.Tensor):
        pairs = pairs.detach().cpu().numpy()
    pairs = np.asarray(pairs)
    if pairs.ndim == 1 and pairs.size == 0:
        pairs = np.empty((2, 0), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2, N]")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    if np.any(pairs < 0):
        raise ValueError(f"{name} indices must be nonnegative")
    if shape is not None and (np.any(pairs[0] >= shape[0]) or np.any(pairs[1] >= shape[1])):
        raise ValueError(f"{name} indices are outside the score matrix")
    return pairs.astype(np.int64, copy=False)


def _genes_by_phenotype(pairs):
    grouped = {}
    for gene, phenotype in pairs.T:
        grouped.setdefault(int(phenotype), set()).add(int(gene))
    return grouped


def standard_ranking_metrics(logits, train_pos, valid_pos, test_pos, ks=(10, 50), n_min=20):
    """Standard ranking metrics: FP32 logits, descending score, ties broken by
    ascending gene index (lexsort).

    Validation candidates = G - train_pos; all validation positives are kept and
    test labels are never used for filtering. Test candidates = G - train_pos -
    valid_pos; all test positives are kept. Equal scores are broken by the fixed
    gene-index order; the tie rate is reported. NDCG uses the per-phenotype
    number of test positives for the IDCG and averages equally over phenotypes
    that contain at least one test positive.

    Phenotypes are additionally stratified by their number of training
    positives (head: >= n_min, tail: < n_min); head/tail macro NDCG@k are
    reported over the test phenotypes of each group.
    """
    scores = _finite_scores(logits)
    n_genes = scores.shape[0]
    gene_ids = np.arange(n_genes, dtype=np.int64)

    train = _genes_by_phenotype(_pair_array(train_pos, "train_pos", scores.shape))
    valid = _genes_by_phenotype(_pair_array(valid_pos, "valid_pos", scores.shape))
    test_pairs = _pair_array(test_pos, "test_pos", scores.shape)
    test = _genes_by_phenotype(test_pairs)

    # test must not overlap train/valid
    for phenotype, genes in test.items():
        overlap = genes & (train.get(phenotype, set()) | valid.get(phenotype, set()))
        if overlap:
            raise ValueError(f"test_pos overlaps train/valid for phenotype {phenotype}")

    ks = tuple(dict.fromkeys(int(k) for k in ks if isinstance(k, (int, np.integer)) and k > 0))
    out = {"pair_MRR": None, "macro_MRR_first": None,
           "n_test_phenotypes": len(test), "n_test_pairs": int(test_pairs.shape[1]),
           "tie_rate": None, "tie_sensitivity_seed50": None,
           "n_head_phenotypes": 0, "n_tail_phenotypes": 0}
    for k in ks:
        out.update({f"pair_Hits@{k}": None, f"macro_Recall@{k}": None,
                    f"macro_NDCG@{k}": None,
                    f"head_macro_NDCG@{k}": None, f"tail_macro_NDCG@{k}": None})
    if not test:
        return out

    reciprocal_ranks, first_rr = [], []
    hits = {k: 0 for k in ks}
    recalls, ndcgs = {k: [] for k in ks}, {k: [] for k in ks}
    ndcg_by_ph = {k: [] for k in ks}
    tie_counts, total_candidates = 0, 0

    for phenotype, positives in test.items():
        blocked = train.get(phenotype, set()) | valid.get(phenotype, set())
        candidates = gene_ids[~np.isin(gene_ids, list(blocked))]
        # lexsort: ascending gene index first, then descending score,
        # so equal scores fall back to the fixed gene-index order
        order = candidates[np.lexsort((candidates, -scores[candidates, phenotype]))]
        rank_of = np.full(n_genes, -1, dtype=np.int64)
        rank_of[order] = np.arange(1, len(order) + 1)

        # tie detection
        candidate_scores = scores[order, phenotype]
        ties = np.sum(candidate_scores[:-1] == candidate_scores[1:]) if len(candidate_scores) > 1 else 0
        tie_counts += ties
        total_candidates += len(candidate_scores)

        ranks = rank_of[np.array(sorted(positives), dtype=np.int64)]
        reciprocal_ranks.extend((1.0 / ranks).tolist())
        first_rr.append(1.0 / int(ranks.min()))
        for k in ks:
            in_top_k = ranks[ranks <= k]
            hits[k] += len(in_top_k)
            recalls[k].append(len(in_top_k) / len(positives))
            dcg = np.sum(1.0 / np.log2(in_top_k + 1))
            ideal_ranks = np.arange(1, min(k, len(positives)) + 1)
            idcg = np.sum(1.0 / np.log2(ideal_ranks + 1))
            val = float(dcg / idcg)
            ndcgs[k].append(val)
            ndcg_by_ph[k].append((phenotype, val))

    out["pair_MRR"] = float(np.mean(reciprocal_ranks))
    out["macro_MRR_first"] = float(np.mean(first_rr))
    out["tie_rate"] = float(tie_counts / max(total_candidates - 1, 1))

    # frequency stratification (head/tail by training positives, independent of k)
    train_counts = {ph: len(genes) for ph, genes in train.items()}
    head_set = {ph for ph in test if train_counts.get(ph, 0) >= n_min}
    out["n_head_phenotypes"] = len(head_set)
    out["n_tail_phenotypes"] = len(test) - len(head_set)
    for k in ks:
        out[f"pair_Hits@{k}"] = hits[k] / out["n_test_pairs"]
        out[f"macro_Recall@{k}"] = float(np.mean(recalls[k]))
        out[f"macro_NDCG@{k}"] = float(np.mean(ndcgs[k]))
        head_vals = [v for ph, v in ndcg_by_ph[k] if ph in head_set]
        tail_vals = [v for ph, v in ndcg_by_ph[k] if ph not in head_set]
        out[f"head_macro_NDCG@{k}"] = float(np.mean(head_vals)) if head_vals else None
        out[f"tail_macro_NDCG@{k}"] = float(np.mean(tail_vals)) if tail_vals else None
    return out


def binary_metrics_by_phenotype(logits, pos, neg):
    """Phenotype-equal-weight macro AUROC / macro AUPRC; only phenotypes with
    both positive and negative samples are included."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    scores = _finite_scores(logits)
    pos_array = _pair_array(pos, "pos", scores.shape)
    neg_array = _pair_array(neg, "neg", scores.shape)
    positive = _genes_by_phenotype(pos_array)
    negative = _genes_by_phenotype(neg_array)
    phenotypes = sorted(positive.keys() & negative.keys())
    aucs, aps, n_pairs = [], [], 0
    for phenotype in phenotypes:
        if positive[phenotype] & negative[phenotype]:
            raise ValueError("pos and neg overlap for the same phenotype")
        pos_genes = pos_array[0, pos_array[1] == phenotype].tolist()
        neg_genes = neg_array[0, neg_array[1] == phenotype].tolist()
        labels = np.concatenate([np.ones(len(pos_genes)), np.zeros(len(neg_genes))])
        values = scores[pos_genes + neg_genes, phenotype]
        aucs.append(float(roc_auc_score(labels, values)))
        aps.append(float(average_precision_score(labels, values)))
        n_pairs += len(labels)
    return {"macro_AUROC": float(np.mean(aucs)) if aucs else None,
            "macro_AUPRC": float(np.mean(aps)) if aps else None,
            "n_phenotypes": len(phenotypes), "n_pairs": n_pairs}


def frequency_group_summary(train_pos, test_pos, neg_test, n_min=20):
    """Split phenotypes into head (>= n_min training positives) and tail groups
    and return group counts (no model metrics are computed here)."""
    train = _genes_by_phenotype(_pair_array(train_pos, "train_pos"))
    positive = _genes_by_phenotype(_pair_array(test_pos, "test_pos"))
    neg_array = _pair_array(neg_test, "neg_test")
    negative = _genes_by_phenotype(neg_array)
    result = {name: {"n_phenotypes": 0, "n_test_positives": 0,
                     "n_test_negatives": 0, "prevalence": None}
              for name in ("head", "tail")}
    for phenotype in sorted(positive.keys() | negative.keys()):
        pos_genes, neg_genes = positive.get(phenotype, set()), negative.get(phenotype, set())
        name = "head" if len(train.get(phenotype, set())) >= n_min else "tail"
        group = result[name]
        group["n_phenotypes"] += 1
        group["n_test_positives"] += len(pos_genes)
        group["n_test_negatives"] += int(np.count_nonzero(neg_array[1] == phenotype))
    for group in result.values():
        total = group["n_test_positives"] + group["n_test_negatives"]
        if total:
            group["prevalence"] = group["n_test_positives"] / total
    return result
