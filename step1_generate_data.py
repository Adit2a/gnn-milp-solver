# Step 1: multi-dimensional knapsack instances solved with Gurobi.
#
#   maximize    sum_i value[i] * x[i]
#   subject to  sum_i weight[j][i] * x[i] <= capacity[j]   for each constraint j
#               x[i] in {0, 1}
#
# Generator follows Chu & Beasley (1998):
#   weight[j][i] ~ U{0..1000}, capacity[j] = 0.5 * sum_i weight[j][i],
#   value[i] = mean_j weight[j][i] + U{0..500}
#
# Output: knapsack_data.pkl

import random
import pickle
import time
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB

NUM_CONSTRAINTS = 5
TIGHTNESS = 0.5


def generate_instance(num_items, num_constraints):
    weights = np.zeros((num_constraints, num_items), dtype=np.int64)
    for j in range(num_constraints):
        for i in range(num_items):
            weights[j, i] = random.randint(0, 1000)

    values = np.zeros(num_items, dtype=np.int64)
    for i in range(num_items):
        average_weight = weights[:, i].sum() / num_constraints
        values[i] = int(average_weight + random.randint(0, 500))

    capacities = np.zeros(num_constraints, dtype=np.int64)
    for j in range(num_constraints):
        capacities[j] = int(TIGHTNESS * weights[j, :].sum())

    return weights, values, capacities


def solve_with_gurobi(weights, values, capacities, threads=None, stop_at_value=None):
    # stop_at_value: stop at the first solution at least this good (used in step 5)
    num_constraints = weights.shape[0]
    num_items = weights.shape[1]

    model = gp.Model()
    model.Params.OutputFlag = 0
    if threads is not None:
        model.Params.Threads = threads
    if stop_at_value is not None:
        model.Params.BestObjStop = stop_at_value

    x = model.addVars(num_items, vtype=GRB.BINARY)
    model.setObjective(gp.quicksum(int(values[i]) * x[i] for i in range(num_items)), GRB.MAXIMIZE)
    for j in range(num_constraints):
        model.addConstr(gp.quicksum(int(weights[j, i]) * x[i] for i in range(num_items)) <= int(capacities[j]))

    start = time.time()
    model.optimize()
    solve_time = time.time() - start

    if model.Status != GRB.OPTIMAL and model.Status != GRB.USER_OBJ_LIMIT:
        return None, None, solve_time

    solution = np.zeros(num_items, dtype=np.int64)
    for i in range(num_items):
        if x[i].X > 0.5:
            solution[i] = 1
    return solution, model.ObjVal, solve_time


def main():
    random.seed(42)
    num_instances = 10000
    min_items = 30
    max_items = 60
    if len(sys.argv) > 1:
        num_instances = int(sys.argv[1])

    dataset = []
    start_all = time.time()
    for k in range(num_instances):
        num_items = random.randint(min_items, max_items)
        weights, values, capacities = generate_instance(num_items, NUM_CONSTRAINTS)
        solution, best_value, solve_time = solve_with_gurobi(weights, values, capacities)

        problem = {
            "weights": weights,           # (num_constraints, num_items)
            "values": values,
            "capacities": capacities,
            "solution": solution,
            "best_value": best_value,
            "solve_time": solve_time,
        }
        dataset.append(problem)

        if (k + 1) % 500 == 0:
            print("Solved", k + 1, "problems,", round(time.time() - start_all), "s so far")

    with open("knapsack_data.pkl", "wb") as f:
        pickle.dump(dataset, f)

    times = []
    for p in dataset:
        times.append(p["solve_time"])
    times = np.array(times)
    print("Saved", len(dataset), "problems to knapsack_data.pkl")
    print("Gurobi solve time: mean", round(times.mean() * 1000, 2), "ms, max", round(times.max() * 1000, 2), "ms")


if __name__ == "__main__":
    main()
