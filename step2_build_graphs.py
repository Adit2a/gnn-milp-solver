# Step 2: bipartite graph with one node per item, one node per constraint,
# and an edge between every item and every constraint carrying weight[j][i].
# Output: knapsack_graphs.pkl

import pickle
import numpy as np

NUM_VAR_FEATURES = 3
NUM_CON_FEATURES = 2
NUM_EDGE_FEATURES = 1


def efficiency_scores(weights, values, capacities):
    # value / sum_j (weight[j][i] / capacity[j]), the standard greedy score for MKP
    weights = np.array(weights, dtype=np.float64)
    capacities = np.array(capacities, dtype=np.float64)
    share_of_each_limit = weights / capacities[:, None]
    share_used = share_of_each_limit.sum(axis=0)
    share_used = np.maximum(share_used, 1e-9)
    return np.array(values, dtype=np.float64) / share_used


def build_graph(weights, values, capacities):
    weights = np.array(weights, dtype=np.float64)
    values = np.array(values, dtype=np.float64)
    capacities = np.array(capacities, dtype=np.float64)
    num_constraints = weights.shape[0]
    num_items = weights.shape[1]

    # share of constraint j used by item i, scaled by num_items so it stays around 1
    share = weights / capacities[:, None] * num_items

    efficiency = efficiency_scores(weights, values, capacities)

    # item features: relative value, relative efficiency, mean share used
    var_features = np.zeros((num_items, NUM_VAR_FEATURES), dtype=np.float32)
    var_features[:, 0] = values / values.max()
    var_features[:, 1] = efficiency / efficiency.max()
    var_features[:, 2] = share.mean(axis=0)

    # constraint features: tightness, problem size
    con_features = np.zeros((num_constraints, NUM_CON_FEATURES), dtype=np.float32)
    con_features[:, 0] = capacities / np.maximum(weights.sum(axis=1), 1e-9)
    con_features[:, 1] = num_items / 100.0

    # edges ordered constraint by constraint, matching share.reshape(-1)
    edge_var = np.tile(np.arange(num_items, dtype=np.int64), num_constraints)
    edge_con = np.repeat(np.arange(num_constraints, dtype=np.int64), num_items)
    edge_features = share.reshape(-1, 1).astype(np.float32)

    graph = {
        "var_features": var_features,
        "con_features": con_features,
        "edge_var": edge_var,
        "edge_con": edge_con,
        "edge_features": edge_features,
    }
    return graph


def main():
    with open("knapsack_data.pkl", "rb") as f:
        dataset = pickle.load(f)

    graphs = []
    for problem in dataset:
        graph = build_graph(problem["weights"], problem["values"], problem["capacities"])
        graph["label"] = problem["solution"].astype(np.float32)
        graphs.append(graph)

    with open("knapsack_graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)

    example = graphs[0]
    print("Saved", len(graphs), "graphs to knapsack_graphs.pkl")
    print("Example graph 0:")
    print("  variable nodes  :", example["var_features"].shape[0])
    print("  constraint nodes:", example["con_features"].shape[0])
    print("  edges           :", example["edge_var"].shape[0])
    print("  first 3 variable feature rows:")
    print(example["var_features"][:3])


if __name__ == "__main__":
    main()
