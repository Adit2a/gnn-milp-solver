# Step 4: feasibility repair. Add items in a given order, skipping any that would
# break a constraint, so every output is feasible.
# repair_best_of_tries also tries "confident items first, then greedy fill" at a few
# probability cutoffs and keeps the best result.

import numpy as np
from step2_build_graphs import efficiency_scores

CONFIDENCE_CUTOFFS = [0.3, 0.5, 0.7, 0.9]


def fill_in_order(weights, values, capacities, order):
    num_constraints = weights.shape[0]
    num_items = weights.shape[1]

    # plain lists are much faster than numpy element access in a Python loop
    item_weights = np.array(weights).T.tolist()
    item_values = np.array(values).tolist()
    limits = np.array(capacities).tolist()
    order = np.array(order).tolist()

    taken = [0] * num_items
    used = [0] * num_constraints
    total_value = 0

    for i in order:
        if taken[i] == 1:
            continue
        w = item_weights[i]
        fits = True
        for j in range(num_constraints):
            if used[j] + w[j] > limits[j]:
                fits = False
                break
        if fits:
            taken[i] = 1
            for j in range(num_constraints):
                used[j] = used[j] + w[j]
            total_value = total_value + item_values[i]

    return np.array(taken, dtype=np.int64), total_value


def order_by_score(scores):
    return np.argsort(-np.array(scores), kind="stable")


def greedy_efficiency(weights, values, capacities):
    scores = efficiency_scores(weights, values, capacities)
    return fill_in_order(weights, values, capacities, order_by_score(scores))


def repair_with_probabilities(weights, values, capacities, probabilities):
    return fill_in_order(weights, values, capacities, order_by_score(probabilities))


def repair_best_of_tries(weights, values, capacities, probabilities):
    probabilities = np.array(probabilities)
    efficiency = efficiency_scores(weights, values, capacities)
    probability_order = order_by_score(probabilities)
    efficiency_order = order_by_score(efficiency)

    best_solution, best_value = fill_in_order(weights, values, capacities, probability_order)

    for cutoff in CONFIDENCE_CUTOFFS:
        order = []
        for i in probability_order:
            if probabilities[i] >= cutoff:
                order.append(i)
        for i in efficiency_order:
            order.append(i)   # already-taken items are skipped by fill_in_order

        solution, value = fill_in_order(weights, values, capacities, order)
        if value > best_value:
            best_solution = solution
            best_value = value

    return best_solution, best_value


def main():
    weights = np.array([[5, 4, 3, 2],
                        [1, 4, 4, 3]])
    values = np.array([10, 9, 6, 3])
    capacities = np.array([8, 7])
    probabilities = [0.9, 0.2, 0.8, 0.4]
    print("Greedy:           ", greedy_efficiency(weights, values, capacities))
    print("GNN single try:   ", repair_with_probabilities(weights, values, capacities, probabilities))
    print("GNN best of tries:", repair_best_of_tries(weights, values, capacities, probabilities))


if __name__ == "__main__":
    main()
