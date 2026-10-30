# -*- coding: utf-8 -*-
"""The six benchmark models: the full evidence-aware HiGNN, its no-evidence
ablation, HGT-style and SimpleHGN-style attention baselines, the DistMult
entity-embedding baseline, and a feature+ID MLP without message passing.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as Fn
from torch.nn import Parameter, Linear, Dropout, Embedding


def mp_edge_types(data):
    """Relations used for message passing: the target gene-phenotype layer is
    excluded."""
    out = []
    for s, r, d in data.edge_types:
        if "assoc" in r:
            continue
        out.append((s, r, d))
    return out


class HiGNNFull(torch.nn.Module):
    """Full HiGNN: neighbour attention, relation-level fusion, and
    phenotype-conditioned task pooling."""

    def __init__(self, data, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.data = data
        self.node_types = list(data.node_types)
        self.edge_types = mp_edge_types(data)
        self.hidden = hidden
        self.inp = torch.nn.ModuleDict({t: Linear(data[t].x.size(1), hidden) for t in self.node_types})
        self.base = torch.nn.ParameterDict({
            t: Parameter(torch.randn(int(data[t].num_nodes), hidden) * 0.1) for t in self.node_types})

        key = lambda et: f"{et[0]}__{et[1]}__{et[2]}"
        self.convs = torch.nn.ModuleList()
        for _ in range(layers):
            conv = RelAttnLayer(self.edge_types, self.node_types, hidden, key, dropout)
            self.convs.append(conv)

        self.out_g = Linear(hidden, hidden)
        self.out_p = Linear(hidden, hidden)
        self.task_ctx = Linear(hidden, hidden)
        self.task_lin = Linear(hidden, hidden)
        self.drop = Dropout(dropout)

    def encode(self):
        h = {t: torch.tanh(self.inp[t](self.data[t].x)) + self.base[t] for t in self.node_types}
        for c in self.convs:
            hn = c(h, self.data)
            h = {t: Fn.elu(hn[t]) for t in self.node_types}
        return h

    def phenotype_context(self, h):
        g = self.task_lin(h["gene"])
        ctx = self.task_ctx(h["phenotype"])
        logit = ctx @ g.t() / (self.hidden ** 0.5)
        gamma = torch.softmax(logit, dim=1)
        self.gamma = gamma.detach()
        z = gamma @ h["gene"] + h["phenotype"]
        return z

    def score_all(self):
        h = self.encode()
        Hg = self.out_g(h["gene"])
        z = self.phenotype_context(h)
        Hp = self.out_p(z)
        self.logits = (Hg @ Hp.t()) / (self.hidden ** 0.5)
        return torch.sigmoid(self.logits)


class RelAttnLayer(torch.nn.Module):
    """Neighbour-level attention per relation plus relation-level semantic
    fusion of the per-relation aggregates."""

    def __init__(self, edge_types, node_types, hidden, key_fn, dropout=0.0):
        super().__init__()
        self.edge_types = edge_types
        self.node_types = node_types
        self.src_lin = torch.nn.ModuleDict({key_fn(et): Linear(hidden, hidden) for et in edge_types})
        self.att = Parameter(torch.randn(len(edge_types), hidden) * 0.1)
        self.ev_w = Parameter(torch.zeros(len(edge_types)))
        self.sg_w = Parameter(torch.zeros(len(edge_types)))
        self.sg_m = Parameter(torch.zeros(len(edge_types)))
        self.rel_att = Parameter(torch.randn(len(edge_types), hidden) * 0.1)
        self.self_lin = torch.nn.ModuleDict({t: Linear(hidden, hidden) for t in node_types})
        self.drop = Dropout(dropout)

    def forward(self, h, data):
        out = {t: self.self_lin[t](h[t]) for t in self.node_types}
        incoming = {t: [] for t in self.node_types}
        rel_pos = {t: [] for t in self.node_types}
        for i, et in enumerate(self.edge_types):
            s, r, d = et
            store = data[et]
            ei = store.edge_index
            ea = store.edge_attr
            n_d = h[d].size(0)
            q = self.src_lin[f"{s}__{r}__{d}"](h[s])[ei[0]]
            logit = (q * self.att[i]).sum(-1)
            logit = logit + self.ev_w[i] * ea[:, 1]
            logit = logit + self.sg_w[i] * ea[:, 0] * ea[:, 1]
            alpha = _softmax_scatter(logit, ei[1], n_d)
            msg = q * alpha.unsqueeze(-1)
            msg = msg * (1.0 + torch.tanh(self.sg_m[i]) * ea[:, 0]).unsqueeze(-1)
            agg = torch.zeros_like(h[d]).index_add_(0, ei[1], self.drop(msg))
            incoming[d].append(agg)
            rel_pos[d].append(i)
        for t, aggs in incoming.items():
            if not aggs:
                continue
            A = torch.stack(aggs, dim=1)
            if A.size(1) > 1:
                idx = torch.tensor(rel_pos[t], device=A.device)
                score = (A * self.rel_att[idx].unsqueeze(0)).sum(-1)
                beta = torch.softmax(score, dim=1)
                mix = (A * beta.unsqueeze(-1)).sum(1)
            else:
                mix = A[:, 0]
            out[t] = out[t] + mix
        return out


def _softmax_scatter(logit, index, n):
    m = torch.zeros(n, device=logit.device).index_reduce_(0, index, logit, "amax", include_self=False)
    e = torch.exp(logit - m.index_select(0, index))
    z = torch.zeros(n, device=logit.device).index_add_(0, index, e)
    return e / (z.index_select(0, index) + 1e-12)


class HiGNNNoInfo(torch.nn.Module):
    """No-evidence ablation: identical backbone, but the edge sign/strength
    inputs are disabled.

    The two evidence-summary channels of the gene features (the sign-weighted
    strength aggregate and the log total evidence weight) are removed as well,
    so that no evidence-derived inputs remain; the features of all other node
    types are unchanged. Unused parameters are removed rather than counted as
    capacity.
    """

    def __init__(self, data, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.data = data
        self.node_types = list(data.node_types)
        self.edge_types = mp_edge_types(data)
        self.hidden = hidden

        # only the last two gene-feature columns are evidence summaries;
        # every other node type keeps its full features
        self.inp = torch.nn.ModuleDict()
        for t in self.node_types:
            orig_dim = data[t].x.size(1)
            new_dim = orig_dim - 2 if t == "gene" else orig_dim
            self.inp[t] = Linear(new_dim, hidden)

        self.base = torch.nn.ParameterDict({
            t: Parameter(torch.randn(int(data[t].num_nodes), hidden) * 0.1) for t in self.node_types})

        key = lambda et: f"{et[0]}__{et[1]}__{et[2]}"
        self.convs = torch.nn.ModuleList()
        for _ in range(layers):
            conv = RelAttnLayerNoInfo(self.edge_types, self.node_types, hidden, key, dropout)
            self.convs.append(conv)

        self.out_g = Linear(hidden, hidden)
        self.out_p = Linear(hidden, hidden)
        self.task_ctx = Linear(hidden, hidden)
        self.task_lin = Linear(hidden, hidden)
        self.drop = Dropout(dropout)

    def encode(self):
        h = {}
        for t in self.node_types:
            x = self.data[t].x[:, :-2] if t == "gene" else self.data[t].x
            h[t] = torch.tanh(self.inp[t](x)) + self.base[t]
        for c in self.convs:
            hn = c(h, self.data)
            h = {t: Fn.elu(hn[t]) for t in self.node_types}
        return h

    def phenotype_context(self, h):
        g = self.task_lin(h["gene"])
        ctx = self.task_ctx(h["phenotype"])
        logit = ctx @ g.t() / (self.hidden ** 0.5)
        gamma = torch.softmax(logit, dim=1)
        self.gamma = gamma.detach()
        z = gamma @ h["gene"] + h["phenotype"]
        return z

    def score_all(self):
        h = self.encode()
        Hg = self.out_g(h["gene"])
        z = self.phenotype_context(h)
        Hp = self.out_p(z)
        self.logits = (Hg @ Hp.t()) / (self.hidden ** 0.5)
        return torch.sigmoid(self.logits)


class RelAttnLayerNoInfo(torch.nn.Module):
    """No-evidence variant of the attention layer: no edge sign/strength, only
    topology and relation types."""

    def __init__(self, edge_types, node_types, hidden, key_fn, dropout=0.0):
        super().__init__()
        self.edge_types = edge_types
        self.node_types = node_types
        self.src_lin = torch.nn.ModuleDict({key_fn(et): Linear(hidden, hidden) for et in edge_types})
        self.att = Parameter(torch.randn(len(edge_types), hidden) * 0.1)
        # no evidence-weight, sign-weight or sign-gate parameters
        self.rel_att = Parameter(torch.randn(len(edge_types), hidden) * 0.1)
        self.self_lin = torch.nn.ModuleDict({t: Linear(hidden, hidden) for t in node_types})
        self.drop = Dropout(dropout)

    def forward(self, h, data):
        out = {t: self.self_lin[t](h[t]) for t in self.node_types}
        incoming = {t: [] for t in self.node_types}
        rel_pos = {t: [] for t in self.node_types}
        for i, et in enumerate(self.edge_types):
            s, r, d = et
            store = data[et]
            ei = store.edge_index
            n_d = h[d].size(0)
            q = self.src_lin[f"{s}__{r}__{d}"](h[s])[ei[0]]
            logit = (q * self.att[i]).sum(-1)
            # no sign/strength bias terms
            alpha = _softmax_scatter(logit, ei[1], n_d)
            msg = q * alpha.unsqueeze(-1)
            # no sign gate
            agg = torch.zeros_like(h[d]).index_add_(0, ei[1], self.drop(msg))
            incoming[d].append(agg)
            rel_pos[d].append(i)
        for t, aggs in incoming.items():
            if not aggs:
                continue
            A = torch.stack(aggs, dim=1)
            if A.size(1) > 1:
                idx = torch.tensor(rel_pos[t], device=A.device)
                score = (A * self.rel_att[idx].unsqueeze(0)).sum(-1)
                beta = torch.softmax(score, dim=1)
                mix = (A * beta.unsqueeze(-1)).sum(1)
            else:
                mix = A[:, 0]
            out[t] = out[t] + mix
        return out


class HGTModel(torch.nn.Module):
    """HGT-style baseline: multi-head attention with per-relation key/query/
    value transforms, adapted to the gene-phenotype decoding head."""

    def __init__(self, data, hidden=64, layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.data = data
        self.node_types = list(data.node_types)
        self.edge_types = mp_edge_types(data)
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        self.inp = torch.nn.ModuleDict({t: Linear(data[t].x.size(1), hidden) for t in self.node_types})
        self.base = torch.nn.ParameterDict({
            t: Parameter(torch.randn(int(data[t].num_nodes), hidden) * 0.1) for t in self.node_types})

        key = lambda et: f"{et[0]}__{et[1]}__{et[2]}"
        self.k_lin = torch.nn.ModuleDict({key(et): Linear(hidden, hidden) for et in self.edge_types})
        self.q_lin = torch.nn.ModuleDict({key(et): Linear(hidden, hidden) for et in self.edge_types})
        self.v_lin = torch.nn.ModuleDict({key(et): Linear(hidden, hidden) for et in self.edge_types})
        self.out_lin = torch.nn.ModuleDict({t: Linear(hidden, hidden) for t in self.node_types})

        self.rel_att = Parameter(torch.randn(len(self.edge_types), num_heads) * 0.1)
        self.drop = Dropout(dropout)

        self.out_g = Linear(hidden, hidden)
        self.out_p = Linear(hidden, hidden)
        self.task_ctx = Linear(hidden, hidden)
        self.task_lin = Linear(hidden, hidden)

    def encode(self):
        h = {t: torch.tanh(self.inp[t](self.data[t].x)) + self.base[t] for t in self.node_types}
        for _ in range(2):  # fixed two layers
            out = {t: self.out_lin[t](h[t]) for t in self.node_types}
            for i, et in enumerate(self.edge_types):
                s, r, d = et
                store = self.data[et]
                ei = store.edge_index
                k = self.k_lin[f"{s}__{r}__{d}"](h[s])[ei[0]].view(-1, self.num_heads, self.head_dim)
                q = self.q_lin[f"{s}__{r}__{d}"](h[s])[ei[0]].view(-1, self.num_heads, self.head_dim)
                v = self.v_lin[f"{s}__{r}__{d}"](h[s])[ei[0]].view(-1, self.num_heads, self.head_dim)
                att = (k * q).sum(-1) / math.sqrt(self.head_dim)
                att = att + self.rel_att[i].unsqueeze(0)
                att = torch.softmax(att, dim=1)
                msg = (att.unsqueeze(-1) * v).view(-1, self.hidden)  # [E, num_heads*head_dim]
                agg = torch.zeros_like(h[d]).index_add_(0, ei[1], self.drop(msg))
                out[d] = out[d] + agg
            h = {t: Fn.elu(out[t]) for t in self.node_types}
        return h

    def phenotype_context(self, h):
        g = self.task_lin(h["gene"])
        ctx = self.task_ctx(h["phenotype"])
        logit = ctx @ g.t() / (self.hidden ** 0.5)
        gamma = torch.softmax(logit, dim=1)
        z = gamma @ h["gene"] + h["phenotype"]
        return z

    def score_all(self):
        h = self.encode()
        Hg = self.out_g(h["gene"])
        z = self.phenotype_context(h)
        Hp = self.out_p(z)
        self.logits = (Hg @ Hp.t()) / (self.hidden ** 0.5)
        return torch.sigmoid(self.logits)


class SimpleHGNModel(torch.nn.Module):
    """SimpleHGN-style baseline: edge-type embeddings, residual connections,
    and relation-level softmax fusion."""

    def __init__(self, data, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.data = data
        self.node_types = list(data.node_types)
        self.edge_types = mp_edge_types(data)
        self.hidden = hidden

        self.inp = torch.nn.ModuleDict({t: Linear(data[t].x.size(1), hidden) for t in self.node_types})
        self.base = torch.nn.ParameterDict({
            t: Parameter(torch.randn(int(data[t].num_nodes), hidden) * 0.1) for t in self.node_types})

        key = lambda et: f"{et[0]}__{et[1]}__{et[2]}"
        self.src_lin = torch.nn.ModuleList([
            torch.nn.ModuleDict({key(et): Linear(hidden, hidden) for et in self.edge_types})
            for _ in range(layers)])
        self.self_lin = torch.nn.ModuleList([
            torch.nn.ModuleDict({t: Linear(hidden, hidden) for t in self.node_types})
            for _ in range(layers)])
        self.edge_type_emb = torch.nn.ParameterList([
            Parameter(torch.randn(len(self.edge_types), hidden) * 0.1) for _ in range(layers)])
        self.rel_att = torch.nn.ParameterList([
            Parameter(torch.randn(len(self.edge_types), hidden) * 0.1) for _ in range(layers)])
        self.drop = Dropout(dropout)

        self.out_g = Linear(hidden, hidden)
        self.out_p = Linear(hidden, hidden)
        self.task_ctx = Linear(hidden, hidden)
        self.task_lin = Linear(hidden, hidden)

    def encode(self):
        h = {t: torch.tanh(self.inp[t](self.data[t].x)) + self.base[t] for t in self.node_types}
        for li in range(len(self.src_lin)):
            out = {t: self.self_lin[li][t](h[t]) for t in self.node_types}
            incoming, rel_pos = {t: [] for t in self.node_types}, {t: [] for t in self.node_types}
            for i, et in enumerate(self.edge_types):
                s, r, d = et
                store = self.data[et]
                ei = store.edge_index
                m = self.src_lin[li][f"{s}__{r}__{d}"](h[s])[ei[0]]
                m = m + self.edge_type_emb[li][i].unsqueeze(0)
                agg = torch.zeros_like(h[d]).index_add_(0, ei[1], self.drop(m))
                incoming[d].append(agg)
                rel_pos[d].append(i)
            for t, aggs in incoming.items():
                if not aggs:
                    continue
                A = torch.stack(aggs, dim=1)
                if A.size(1) > 1:
                    idx = torch.tensor(rel_pos[t], device=A.device)
                    score = (A * self.rel_att[li][idx].unsqueeze(0)).sum(-1)
                    beta = torch.softmax(score, dim=1)
                    out[t] = out[t] + (A * beta.unsqueeze(-1)).sum(1)
                else:
                    out[t] = out[t] + A[:, 0]
            h = {t: Fn.elu(out[t]) for t in self.node_types}
        return h

    def phenotype_context(self, h):
        g = self.task_lin(h["gene"])
        ctx = self.task_ctx(h["phenotype"])
        logit = ctx @ g.t() / (self.hidden ** 0.5)
        gamma = torch.softmax(logit, dim=1)
        z = gamma @ h["gene"] + h["phenotype"]
        return z

    def score_all(self):
        h = self.encode()
        Hg = self.out_g(h["gene"])
        z = self.phenotype_context(h)
        Hp = self.out_p(z)
        self.logits = (Hg @ Hp.t()) / (self.hidden ** 0.5)
        return torch.sigmoid(self.logits)


class DistMultModel(torch.nn.Module):
    """DistMult baseline: trainable entity-identifier embeddings with a
    bilinear (diagonal) scorer."""

    def __init__(self, data, hidden=64, **kw):
        super().__init__()
        self.data = data
        self.hidden = hidden
        self.g_emb = Embedding(int(data["gene"].num_nodes), hidden)
        self.p_emb = Embedding(int(data["phenotype"].num_nodes), hidden)
        torch.nn.init.xavier_uniform_(self.g_emb.weight)
        torch.nn.init.xavier_uniform_(self.p_emb.weight)

    def score_all(self):
        hg = self.g_emb(torch.arange(self.data["gene"].num_nodes, device=self.g_emb.weight.device))
        hp = self.p_emb(torch.arange(self.data["phenotype"].num_nodes, device=self.p_emb.weight.device))
        self.logits = (hg @ hp.t()) / (self.hidden ** 0.5)
        return torch.sigmoid(self.logits)


class FeatureIDMLP(torch.nn.Module):
    """Feature+ID MLP baseline: no message passing; uses graph-derived node
    features plus trainable node identity embeddings."""

    def __init__(self, data, hidden=64, **kw):
        super().__init__()
        self.data = data
        self.g = Linear(data["gene"].x.size(1), hidden)
        self.p = Linear(data["phenotype"].x.size(1), hidden)
        self.gb = Parameter(torch.randn(int(data["gene"].num_nodes), hidden) * 0.1)
        self.pb = Parameter(torch.randn(int(data["phenotype"].num_nodes), hidden) * 0.1)
        self.hidden = hidden

    def score_all(self):
        hg = torch.tanh(self.g(self.data["gene"].x)) + self.gb
        hp = torch.tanh(self.p(self.data["phenotype"].x)) + self.pb
        self.logits = (hg @ hp.t()) / (self.hidden ** 0.5)
        return torch.sigmoid(self.logits)


MODELS = {
    "full": HiGNNFull,
    "no_info": HiGNNNoInfo,
    "hgt": HGTModel,
    "simplehgn": SimpleHGNModel,
    "distmult": DistMultModel,
    "mlp": FeatureIDMLP,
}
