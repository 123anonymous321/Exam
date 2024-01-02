import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid

from data_visualization import plot_loss_and_accuracies
from gin_model import GIN
from train_node_classification import HYPERPARAMS_PATH


def compute_confidence(accuracies: list):
    std = np.std(accuracies)
    conf = 1.96 * std
    return conf


if __name__ == "__main__":
    files = list(HYPERPARAMS_PATH.glob("*.json"))

    # store test accuracies for all trainings i, only best epoch, for confidence intervalls
    model_test_accuracies = {}

    # store the following only in the first training, for all epochs, for plotting
    all_train_losses = {}
    all_train_accuracies = {}
    all_val_accuracies = {}

    for hyperparam_file in files:
        print(f"hyperparam_file = {hyperparam_file}")

        with open(hyperparam_file, "r") as f:
            hyperparams = json.load(f)
            print(f"loaded_hp = {hyperparams}")

        if hyperparams["dataset"] == "Cora":
            dataset = Planetoid(root="/tmp/Cora", name="Cora")
        elif hyperparams["dataset"] == "CiteSeer":
            dataset = Planetoid(root="/tmp/Citeseer", name="CiteSeer")
        data = dataset[0]

        params = {
            "input_features": dataset.num_features,
            "hidden_features": hyperparams["hidden_dim"],
            "num_layers": hyperparams["n_layers"],
            "num_mlp_layers": 3,
            "learning_rate": hyperparams["learning_rate"],
            "drop_ratio": hyperparams["drop_ratio"],
            "weight_decay": hyperparams["weight_decay"],
            "num_epochs": 200,
            "num_classes": dataset.num_classes,  # 2
            "patience": 15,
        }
        gr = hyperparams["global_readout"]
        ds = hyperparams["dataset"]
        model_name = f"GIN GR {ds}" if gr else f"GIN {ds}"
        print(f"model_name = {model_name}")

        train_losses = []
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []

        for i in range(100):
            print(f"\ni = {i}")
            model = GIN(
                num_layers=params["num_layers"],
                input_dim=params["input_features"],
                hidden_dim=params["hidden_features"],
                num_classes=params["num_classes"],
                node_classifier=True,
                global_readout=hyperparams["global_readout"],
            )

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )
            best_acc = -1.0
            patience = params["patience"]
            curr_patience = 0
            test_acc = -1.0

            for epoch in range(params["num_epochs"]):
                model.train()
                optimizer.zero_grad()
                y_pred = model(data.x, data.edge_index)
                y_true_one_hot = F.one_hot(
                    input=data.y, num_classes=dataset.num_classes
                ).to(torch.float)

                loss = loss_fn(
                    y_pred[data.train_mask],
                    y_true_one_hot[data.train_mask].to(torch.float),
                )
                loss.backward()
                optimizer.step()

                model.eval()
                # compute accuracies
                y_pred_labels = torch.argmax(y_pred, dim=1)
                y_true_labels = data.y

                train_acc = accuracy_score(
                    y_true_labels[data.train_mask], y_pred_labels[data.train_mask]
                )

                val_acc = accuracy_score(
                    y_true_labels[data.val_mask], y_pred_labels[data.val_mask]
                )
                if i == 0:
                    train_losses.append(loss)
                    train_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)

                if val_acc > best_acc:
                    best_acc = val_acc
                    curr_patience = 0
                    best_epoch = epoch
                    test_acc = accuracy_score(
                        y_true_labels[data.test_mask], y_pred_labels[data.test_mask]
                    )
                else:
                    curr_patience += 1
                    if curr_patience == patience:
                        break

            print(f"epoch = {epoch}")

            test_accuracies.append(test_acc)
            print(f"best_acc = {best_acc}")

        model_test_accuracies[model_name] = test_accuracies

        all_train_losses[model_name] = [x.detach().numpy() for x in train_losses]
        all_train_accuracies[model_name] = [x for x in train_accuracies]
        all_val_accuracies[model_name] = [x for x in val_accuracies]

    for model, test_accuracies in model_test_accuracies.items():
        confidence = compute_confidence(test_accuracies)
        print(f"model = {model}")
        print(f"np.mean(test_accuracies) = {np.mean(test_accuracies)}")
        print(f"confidence = {confidence}")

    plot_loss_and_accuracies(
        losses=all_train_losses,
        train_accuracies=all_train_accuracies,
        val_accuracies=all_val_accuracies,
    )
