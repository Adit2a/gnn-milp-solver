# Step 5: benchmark on the 1000 test instances.
# All methods run single-threaded and are timed end to end
# (graph building + network + repair for the GNN).

import pickle
import time
import numpy as np
import torch

from step1_generate_data import solve_with_gurobi
from step2_build_graphs import build_graph
from step3_train_gnn import KnapsackGNN, make_batch, NUM_TRAIN, NUM_VAL
from step4_repair import repair_with_probabilities, repair_best_of_tries, greedy_efficiency


def gap_percent(best_value, found_value):
    return 100.0 * (best_value - found_value) / best_value


def summarize(name, gaps, total_seconds, num_problems, gurobi_ms):
    gaps = np.array(gaps)
    optimal_count = 0
    for g in gaps:
        if g < 1e-9:
            optimal_count = optimal_count + 1
    ms = 1000.0 * total_seconds / num_problems
    print(name.ljust(32),
          "| mean gap", str(round(gaps.mean(), 3)).rjust(6), "%",
          "| max gap", str(round(gaps.max(), 3)).rjust(6), "%",
          "| optimal", str(optimal_count).rjust(4), "/", num_problems,
          "| time", str(round(ms, 3)).rjust(8), "ms",
          "| speedup vs Gurobi", str(round(gurobi_ms / ms, 1)).rjust(6), "x")


def predict_batched(model, problems):
    # one forward pass for all problems, split back into one array per problem
    graphs = []
    for problem in problems:
        graphs.append(build_graph(problem["weights"], problem["values"], problem["capacities"]))
    with torch.no_grad():
        all_probabilities = torch.sigmoid(model(make_batch(graphs))).numpy()

    per_problem = []
    position = 0
    for problem in problems:
        num_items = len(problem["values"])
        per_problem.append(all_probabilities[position:position + num_items])
        position = position + num_items
    return per_problem


def main():
    torch.set_num_threads(1)

    with open("knapsack_data.pkl", "rb") as f:
        dataset = pickle.load(f)
    test_problems = dataset[NUM_TRAIN + NUM_VAL:]
    num_problems = len(test_problems)
    print("Test problems:", num_problems)

    model = KnapsackGNN(hidden_size=64, num_rounds=2)
    model.load_state_dict(torch.load("gnn_model.pt"))
    model.eval()

    best_values = []
    for problem in test_problems:
        best_values.append(problem["best_value"])

    # Gurobi to proven optimality
    print("Timing Gurobi (1 thread)...")
    start = time.perf_counter()
    for problem in test_problems:
        solve_with_gurobi(problem["weights"], problem["values"], problem["capacities"], threads=1)
    gurobi_seconds = time.perf_counter() - start
    gurobi_ms = 1000.0 * gurobi_seconds / num_problems

    # Greedy
    greedy_gaps = []
    start = time.perf_counter()
    for k in range(num_problems):
        p = test_problems[k]
        solution, value = greedy_efficiency(p["weights"], p["values"], p["capacities"])
        greedy_gaps.append(gap_percent(best_values[k], value))
    greedy_seconds = time.perf_counter() - start

    # warm-up call, not timed
    predict_batched(model, test_problems[:1])

    # GNN + repair, one instance at a time
    single_gaps = []
    start = time.perf_counter()
    for k in range(num_problems):
        p = test_problems[k]
        probabilities = predict_batched(model, [p])[0]
        solution, value = repair_with_probabilities(p["weights"], p["values"], p["capacities"], probabilities)
        single_gaps.append(gap_percent(best_values[k], value))
    single_seconds = time.perf_counter() - start

    # GNN + repair, batched
    batched_gaps = []
    start = time.perf_counter()
    all_probabilities = predict_batched(model, test_problems)
    for k in range(num_problems):
        p = test_problems[k]
        solution, value = repair_with_probabilities(p["weights"], p["values"], p["capacities"], all_probabilities[k])
        batched_gaps.append(gap_percent(best_values[k], value))
    batched_seconds = time.perf_counter() - start

    # GNN + best-of-5 repair, batched
    tries_gaps = []
    tries_values = []
    start = time.perf_counter()
    all_probabilities = predict_batched(model, test_problems)
    for k in range(num_problems):
        p = test_problems[k]
        solution, value = repair_best_of_tries(p["weights"], p["values"], p["capacities"], all_probabilities[k])
        tries_gaps.append(gap_percent(best_values[k], value))
        tries_values.append(value)
    tries_seconds = time.perf_counter() - start

    # Gurobi stopped at the first solution at least as good as the GNN's,
    # i.e. the time Gurobi needs to reach the same quality
    print("Timing Gurobi stopped at the GNN's solution value...")
    match_gaps = []
    start = time.perf_counter()
    for k in range(num_problems):
        p = test_problems[k]
        solution, value, solve_time = solve_with_gurobi(p["weights"], p["values"], p["capacities"],
                                                        threads=1, stop_at_value=tries_values[k])
        match_gaps.append(gap_percent(best_values[k], value))
    match_seconds = time.perf_counter() - start

    correct = 0
    total = 0
    for k in range(num_problems):
        predicted = all_probabilities[k] > 0.5
        correct = correct + (predicted == (test_problems[k]["solution"] == 1)).sum()
        total = total + len(predicted)

    print()
    print("GNN per-item accuracy:", round(100.0 * correct / total, 2), "%")
    print()
    summarize("Gurobi (exact, 1 thread)", np.zeros(num_problems), gurobi_seconds, num_problems, gurobi_ms)
    summarize("Greedy by efficiency", greedy_gaps, greedy_seconds, num_problems, gurobi_ms)
    summarize("GNN + repair (one by one)", single_gaps, single_seconds, num_problems, gurobi_ms)
    summarize("GNN + repair (batched)", batched_gaps, batched_seconds, num_problems, gurobi_ms)
    summarize("GNN + best-of-5 repair (batched)", tries_gaps, tries_seconds, num_problems, gurobi_ms)
    summarize("Gurobi stopped at GNN's value", match_gaps, match_seconds, num_problems, gurobi_ms)

    same_quality_speedup = match_seconds / tries_seconds
    print()
    print("Same-quality speedup (Gurobi stopped at GNN's value / GNN best-of-5 batched):",
          round(same_quality_speedup, 1), "x")


if __name__ == "__main__":
    main()
