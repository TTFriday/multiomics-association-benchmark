# Multi-omics association completion benchmark

Companion code for the benchmarking study "Entity features dominate message
passing in multi-omics association completion: a systematic benchmarking study".
The benchmark task is gene–phenotype association completion on AtMAD
(Arabidopsis thaliana), formulated as transductive link prediction over a
signed, evidence-weighted multi-relational graph.

Raw association tables are not included (see Data).

## Layout

```
experiments/benchmark  graph build, the six benchmark models, unified training
                       and evaluation, evidence permutation and deletion
                       ablations, statistical analysis
src/data               raw AtMAD table parsing and ID normalisation; writes the
                       data_cache/*.pkl relation tables consumed by the graph builder
provenance/            SHA256 manifest of the local AtMAD input files
results/benchmark      scalar per-run results behind the paper tables (see below)
```

## Environment

Tested with Python 3.10.9, PyTorch 2.5.1+cu121 and PyTorch Geometric 2.8.1 on
Windows and an NVIDIA RTX 3090. CPU is enough for data preparation and the
analysis; full-graph training needs a CUDA GPU (one benchmark fit takes about
one minute on that device). GPU scatter-add is not bitwise deterministic, so
rerunning an identical seed gives smallly different numbers; the recorded
per-run JSONs are the reference.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Data

AtMAD source paper: Lan Y, Sun R, Ouyang J, Ding W, Kim MJ, Wu J, Li Y, Shi T (2021)
AtMAD: Arabidopsis thaliana multi-omics association database. Nucleic Acids Research
49(D1):D1445–D1451. https://doi.org/10.1093/nar/gkaa1042

Download the ten tables from http://www.megabionet.org/atmad (download page:
http://119.3.41.228/atmad/download.php): `ciseqtl.xls`, `transeqtl.xls`,
`environmenteqtl.xlsx`, `gwaseqtl.csv`, `emqtl.csv`, `epathway.csv`,
`phenotypepathway.csv`, `gwas.csv`, `twas.csv`, `ewas.csv`.

Free download access is not a redistribution licence, so the tables are not shipped
here. `provenance/AtMAD_input_manifest.json` records the SHA256 of the exact local
copies used for the reported graph. The page label "June 2020" describes the server's
summary, not our snapshot, and the original acquisition date was not recorded. As of
October 2026 the current remote file lengths differ from the local ones, so a fresh
download will not reproduce these numbers byte for byte; rebuild the graph and rerun
if you need results from the live snapshot.

Point the code at your own copy:

```bash
export HIGNN_RAW=/absolute/path/to/AtMAD_inputs      # default: ../rawdata
```

Generated caches go to `data_cache/`, which is git-ignored. Only load `.pt`/`.pkl`
files you produced yourself.

## Benchmark study

All commands run from the repository root with the environment above.

### Graph

```bash
python -m src.data.loaders                           # parse the ten AtMAD tables into data_cache/*.pkl
python -m experiments.benchmark.build_graph_v2       # build data_cache/graph_v2.pt + results/benchmark/graph_audit.json
```

The loader standardises entity IDs (TAIR gene identifiers, SNP loci) and converts
p values/FDR to signed log-strength edge attributes. The builder assembles the
heterogeneous graph with the corrected parser (chromosome indexing, CHH coverage,
unknown methylation contexts, MAF taken from retained GWAS edges). The graph
carries 21 directed relations over six node types; the target `assoc` layer is
excluded from message passing, leaving 20 auxiliary relations.
`results/benchmark/graph_audit.json` records node/edge counts, feature
dimensions and input hashes.

### Models

| Name | What it actually is |
|---|---|
| `distmult` | trainable entity embeddings with bilinear scoring; no message passing, no node features |
| `mlp` | node features plus trainable identity embeddings; no message passing |
| `hgt` | HGT-style control: type-specific K/Q/V projections and multi-head attention (adapted rewrite) |
| `simplehgn` | SimpleHGN-style control: edge-type embeddings plus relation-level semantic attention (adapted rewrite) |
| `full` | HiGNN: neighbour attention using edge sign and strength, relation mixing, phenotype-conditioned task pooling |
| `no_info` | HiGNN ablation: edge attributes and the two gene evidence-summary feature channels removed; topology and types retained |

Baseline names refer to adapted re-implementations, not to the published
architectures. The `no_info` control keeps everything else identical to `full`
(same layers, parameters and schedule) so the comparison isolates the evidence
inputs; dropped parameters are removed, not zeroed, and do not count as capacity.

### Protocol

- Positive target associations are deduplicated and split 8:1:1 into
  train/validation/test by a fixed permutation per seed.
- Negative pairs are sampled without replacement and without overlap between
  splits, with phenotype drawn from the split's positive frequency and gene
  uniform. The training loss averages log-sigmoid terms over positives and over
  sampled negatives (balanced 1:1); binary AUROC/AUPRC use 10 sampled negatives
  per positive.
- Ranking metrics (macro NDCG@10/50, pair MRR) are computed over full candidate
  sets in FP32: all gene–phenotype pairs minus training positives, and minus
  validation positives for the test evaluation, with lexicographic tie-breaking.
- Head/tail stratified NDCG@50 groups phenotypes by that seed's training
  frequency: head ≥ 20 training positives, tail < 20.
- Model selection: validation macro NDCG@50 at 25-epoch checkpoints (and the
  final epoch), best state restored. 150 epochs, hidden width 64, 2 layers,
  message dropout 0.2, Adam, gradient-norm clip 5.0.
- Hyperparameters were selected on a development grid (6 models × learning
  rates {1e-3, 5e-3} × weight decays {0, 1e-5, 5e-4} × dev seeds {91, 92});
  the selected (lr, wd) per model is recorded in `results/benchmark/run_log.jsonl`.
- Formal comparison: 6 models × seeds 101–110.

### Running

```bash
python -m experiments.benchmark.run_all --phase dev      # development grid, 72 fits
python -m experiments.benchmark.run_all --phase formal   # six models x ten seeds, 60 fits
python -m experiments.benchmark.run_all --phase e2       # evidence permutation, 30 fits
python -m experiments.benchmark.run_all --phase e3       # evidence deletion, 40 fits
python -m experiments.benchmark.analyze_results          # summary tables, Wilcoxon + Holm, bootstrap CIs
```

Each fit is a separate process that writes one JSON; completed fits are skipped
on re-invocation (a fit counts as complete when its record contains the
frequency-stratified test metrics), so an interrupted batch resumes where it
stopped. E2 modes are `sign_permute`, `strength_permute` and `full_permute`
(edge attributes shuffled, topology preserved). E3 conditions are `no_gwas`,
`no_meth`, `no_pathway` and `no_env` (all edges of one biological evidence
category removed, forward and reverse; the runner aborts if a condition matches
nothing).

`analyze_results.py` writes `summary_statistics.csv` (mean ± standard deviation
per model) and `model_comparisons.csv` (Wilcoxon signed-rank tests with Holm
correction, full model against each other model, seeds matched).

### Result files

```
results/benchmark/
  graph_audit.json                            graph construction audit
  best_configs.json                           selected (lr, wd) per model
  <model>_seed<seed>.json                     formal per-run records, seeds 101-110
  <model>_seed<seed>_dev<seed>.json           development-grid records
  full_seed<seed>_dev<seed>_<mode>.json       E2 permutation records
  full_seed<seed>_dev<seed>_<condition>.json  E3 deletion records
  run_log.jsonl                               batch log: phase, hyperparameters, status
  summary_statistics.csv                      per-model mean ± std
  model_comparisons.csv                       paired tests with Holm correction
```

Each JSON keeps the full configuration, the validation history, validation and
test ranking metrics, binary metrics, frequency-stratified metrics, and the
SHA256 of the test logits.

### Headline results

Means ± population standard deviation over the ten formal seeds; the full
metric set is in `summary_statistics.csv` and the per-run JSONs are the
reference:

| Model | macro NDCG@50 | macro AUROC |
|---|---|---|
| `distmult` | 0.3835 ± 0.0153 | 0.9249 ± 0.0105 |
| `mlp` | 0.2843 ± 0.0119 | 0.9228 ± 0.0090 |
| `simplehgn` | 0.1355 ± 0.0252 | 0.8745 ± 0.0441 |
| `full` (HiGNN) | 0.1038 ± 0.0138 | 0.7507 ± 0.0408 |
| `hgt` | 0.0815 ± 0.0153 | 0.8627 ± 0.0264 |
| `no_info` | 0.0905 ± 0.0164 | 0.8606 ± 0.0411 |

Entity-only models lead by a wide margin: the best relational model reaches
about a third of the DistMult score. The evidence ablations point the same way.
Permuting edge attributes (E2) changes macro NDCG@50 by at most 4.4% and
deleting whole edge categories (E3) by at most 8.8%, in every case matching or
exceeding the baseline; none of the permutation effects is significant after
Holm correction. Frequency stratification reproduces the head/tail gap: all six
models rank head phenotypes (≥ 20 training positives) far above tail ones, and
the absolute head−tail difference is 0.17–0.28 for the entity-only models
versus 0.07–0.09 for the relational variants.

## Notes

- Figures in the paper are produced by a separate local plotting script that is not part
  of this release.

## License

To be selected by the authors before publication. Until then no reuse rights are
granted. Please cite the paper if you build on this work.
