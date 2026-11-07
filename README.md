# Learning-Based MILP Solver with Graph Neural Networks

A GNN that predicts solutions to multi-dimensional knapsack problems, trained on
Gurobi-optimal labels, with a feasibility-repair step and a benchmark against Gurobi
and a greedy baseline.

## Problem

Multi-dimensional knapsack (5 constraints, 30–60 items):

```
maximize    sum_i v_i x_i
subject to  sum_i w_ji x_i <= c_j   for j = 1..5
            x_i in {0, 1}
```

Instances follow the Chu & Beasley (1998) generator: `w_ji ~ U{0..1000}`,
`c_j = 0.5 * sum_i w_ji`, `v_i = mean_j w_ji + U{0..500}`.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `step1_generate_data.py` | Generates 10,000 instances and solves each with Gurobi for labels |
| 2 | `step2_build_graphs.py` | Converts each instance to a variable–constraint bipartite graph |
| 3 | `step3_train_gnn.py` | Trains a message-passing GNN to predict per-item take probabilities |
| 4 | `step4_repair.py` | Turns probabilities into feasible solutions (greedy fill, best of 5 cutoffs) |
| 5 | `step5_benchmark.py` | Benchmarks against Gurobi and a greedy efficiency baseline |

**Graph.** One node per item, one node per constraint, and an edge for every
non-zero `w_ji`. Item features: relative value, relative efficiency, mean share of
capacity used. Constraint features: tightness, problem size. Edge feature: scaled `w_ji / c_j`.

**Model.** Two rounds of item→constraint→item message passing with mean aggregation,
64-dim hidden states, trained with binary cross-entropy on the Gurobi labels
(8,000 train / 1,000 validation / 1,000 test).

**Repair.** Items are added in order of predicted probability and skipped if they would
violate any constraint, so every output is feasible. The best-of-5 variant also tries
"items with probability ≥ T first, then fill by efficiency" for T ∈ {0.3, 0.5, 0.7, 0.9}
and keeps the best.

## Results

1,000 held-out test instances. Every method uses 1 CPU thread and is timed end to end
(graph construction + inference + repair for the GNN).

| Method | Mean gap | Max gap | Optimal | Time / instance |
|---|---|---|---|---|
| Gurobi (exact) | 0% | 0% | 1000/1000 | 590 ms |
| Greedy by efficiency | 4.29% | 16.7% | 8 | 0.14 ms |
| GNN + repair (batched) | 2.46% | 18.9% | 27 | 2.7 ms |
| **GNN + best-of-5 repair (batched)** | **1.91%** | 9.3% | 37 | **3.3 ms** |
| Gurobi, stopped at the GNN's objective | 1.15% | 6.6% | 95 | 10.4 ms |

- GNN item-level accuracy: 89.3%.
- The GNN's gap is less than half of greedy's (1.91% vs 4.29%).
- At equal solution quality (Gurobi stopped as soon as it matches the GNN's objective
  via `BestObjStop`), the GNN is about **3x faster** than Gurobi. Gurobi's time to match
  varied between runs (10–15 ms), so ~3x is the robust figure. The ~180x figure against
  a full Gurobi solve is not a like-for-like comparison, since Gurobi also proves optimality.
- All GNN and greedy solutions were checked to satisfy every constraint.

## Running it

```bash
pip install -r requirements.txt
python step1_generate_data.py     # ~1 hour (Gurobi), writes knapsack_data.pkl
python step2_build_graphs.py      # writes knapsack_graphs.pkl
python step3_train_gnn.py         # ~30 min on CPU, writes gnn_model.pt
python step5_benchmark.py         # ~15 min
```

A trained `gnn_model.pt` is included. The generated `.pkl` data files are not committed;
step 1 regenerates them deterministically (seed 42). Gurobi's free pip license is enough,
since these models are well under its 2,000-variable limit.

## Limitations

- Trained and tested on one instance distribution; larger or structurally different
  problems would need retraining.
- No optimality guarantee: worst-case gap is 9.3%.
- CPU only; batched inference would be faster on a GPU.
