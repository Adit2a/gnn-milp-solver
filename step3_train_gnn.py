# Step 3: train a bipartite GNN to predict per-item take probabilities.
# Output: gnn_model.pt

import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn

from step2_build_graphs import NUM_VAR_FEATURES, NUM_CON_FEATURES, NUM_EDGE_FEATURES

NUM_TRAIN = 8000
NUM_VAL = 1000
# the remaining 1000 graphs are the test set (step 5)


def make_batch(graph_list):
    var_feature_list = []
    con_feature_list = []
    edge_var_list = []
    edge_con_list = []
    edge_feature_list = []
    label_list = []

    var_offset = 0
    con_offset = 0
    for graph in graph_list:
        var_feature_list.append(graph["var_features"])
        con_feature_list.append(graph["con_features"])
        edge_var_list.append(graph["edge_var"] + var_offset)
        edge_con_list.append(graph["edge_con"] + con_offset)
        edge_feature_list.append(graph["edge_features"])
        if "label" in graph:
            label_list.append(graph["label"])
        var_offset = var_offset + graph["var_features"].shape[0]
        con_offset = con_offset + graph["con_features"].shape[0]

    batch = {
        "var_features": torch.tensor(np.concatenate(var_feature_list)),
        "con_features": torch.tensor(np.concatenate(con_feature_list)),
        "edge_var": torch.tensor(np.concatenate(edge_var_list)),
        "edge_con": torch.tensor(np.concatenate(edge_con_list)),
        "edge_features": torch.tensor(np.concatenate(edge_feature_list)),
    }
    if len(label_list) > 0:
        batch["label"] = torch.tensor(np.concatenate(label_list))
    return batch


def mean_aggregate(messages, target_index, num_targets):
    hidden_size = messages.shape[1]
    total = torch.zeros(num_targets, hidden_size)
    total.index_add_(0, target_index, messages)

    count = torch.zeros(num_targets)
    count.index_add_(0, target_index, torch.ones(len(target_index)))
    count = count.clamp(min=1.0)

    return total / count.unsqueeze(1)


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_size, edge_size):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + edge_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, sender_states, receiver_states, edge_sender, edge_receiver, edge_features):
        message_input = torch.cat([sender_states[edge_sender],
                                   receiver_states[edge_receiver],
                                   edge_features], dim=1)
        messages = self.message_mlp(message_input)
        incoming = mean_aggregate(messages, edge_receiver, receiver_states.shape[0])
        return self.update_mlp(torch.cat([receiver_states, incoming], dim=1))


class KnapsackGNN(nn.Module):
    def __init__(self, hidden_size=64, num_rounds=2):
        super().__init__()
        self.num_rounds = num_rounds
        self.var_embed = nn.Sequential(nn.Linear(NUM_VAR_FEATURES, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.con_embed = nn.Sequential(nn.Linear(NUM_CON_FEATURES, hidden_size), nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.var_to_con = nn.ModuleList()
        self.con_to_var = nn.ModuleList()
        for r in range(num_rounds):
            self.var_to_con.append(MessagePassingLayer(hidden_size, NUM_EDGE_FEATURES))
            self.con_to_var.append(MessagePassingLayer(hidden_size, NUM_EDGE_FEATURES))
        self.output_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                         nn.Linear(hidden_size, 1))

    def forward(self, batch):
        var_states = self.var_embed(batch["var_features"])
        con_states = self.con_embed(batch["con_features"])
        edge_var = batch["edge_var"]
        edge_con = batch["edge_con"]
        edge_features = batch["edge_features"]

        for r in range(self.num_rounds):
            con_states = self.var_to_con[r](var_states, con_states, edge_var, edge_con, edge_features)
            var_states = self.con_to_var[r](con_states, var_states, edge_con, edge_var, edge_features)

        return self.output_head(var_states).squeeze(1)


def evaluate(model, graphs, loss_function, batch_size=256):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = make_batch(graphs[start:start + batch_size])
            logits = model(batch)
            num_items = len(batch["label"])
            total_loss = total_loss + loss_function(logits, batch["label"]).item() * num_items
            predictions = (logits > 0).float()
            total_correct = total_correct + (predictions == batch["label"]).sum().item()
            total_items = total_items + num_items
    return total_loss / total_items, total_correct / total_items


def main():
    random.seed(0)
    torch.manual_seed(0)

    with open("knapsack_graphs.pkl", "rb") as f:
        graphs = pickle.load(f)
    train_graphs = graphs[:NUM_TRAIN]
    val_graphs = graphs[NUM_TRAIN:NUM_TRAIN + NUM_VAL]
    print("Train graphs:", len(train_graphs), " Validation graphs:", len(val_graphs))

    model = KnapsackGNN(hidden_size=64, num_rounds=2)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    batch_size = 64
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        order = list(range(len(train_graphs)))
        random.shuffle(order)

        for start in range(0, len(order), batch_size):
            batch_graphs = []
            for index in order[start:start + batch_size]:
                batch_graphs.append(train_graphs[index])
            batch = make_batch(batch_graphs)

            loss = loss_function(model(batch), batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_graphs, loss_function)
        val_loss, val_acc = evaluate(model, val_graphs, loss_function)
        print("Epoch", epoch + 1,
              "| train loss", round(train_loss, 4), "acc", round(train_acc, 4),
              "| val loss", round(val_loss, 4), "acc", round(val_acc, 4),
              "|", round(time.time() - start_time, 1), "s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "gnn_model.pt")

    print("Best validation loss:", round(best_val_loss, 4), "-> saved to gnn_model.pt")


if __name__ == "__main__":
    main()
