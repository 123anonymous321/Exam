import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid

from gin_model import GIN

MODEL_DATA_PATH = Path(__file__).parent.parent / "model_data"
FINAL_MODELS_PATH = MODEL_DATA_PATH / "final_models"
HYPERPARAMS_PATH = MODEL_DATA_PATH / "hyperparameters"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on a specified dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help='Name of the dataset to train on: "cora", "citeseer"',
    )
    args = parser.parse_args()

    if args.dataset == "cora":
        dataset = Planetoid(root="/tmp/Cora", name="Cora")
    elif args.dataset == "citeseer":
        dataset = Planetoid(root="/tmp/Citeseer", name="CiteSeer")
    else:
        raise ValueError(
            "The given dataset is not valid. Valid options are 'cora' and 'citeseer'."
        )
    data = dataset[0]

    best_models = {}
    for global_readout in [True, False]:
        print(
            "\n\n\n\n\n--------------------------------------------------------------------------------------------------------"
        )
        print(f"GR = {global_readout}")

        key = f"GR_{dataset.name}" if global_readout else f"{dataset.name}"
        best_models[key] = {}
        best_model_acc = -1.0
        for weight_decay in [0, 1e-3, 1e-4]:
            for learning_rate in [1e-3, 1e-4]:
                for n_layers in [4, 5, 6]:
                    for hidden_dim in [128]:
                        for drop_ratio in [0, 0.1, 0.2]:
                            print(
                                f"\nweight_decay = {weight_decay}, learning_rate = {learning_rate}, n_layers = {n_layers}, hidden_dim = {hidden_dim}, drop_ratio = {drop_ratio}"
                            )
                            params = {
                                "input_features": dataset.num_features,
                                "hidden_features": hidden_dim,
                                "num_layers": n_layers,
                                "num_mlp_layers": 3,
                                "learning_rate": learning_rate,
                                "drop_ratio": drop_ratio,
                                "weight_decay": weight_decay,
                                "num_epochs": 200,
                                "num_classes": dataset.num_classes,  # 2
                                "patience": 15,
                            }

                            model = GIN(
                                num_layers=params["num_layers"],
                                input_dim=params["input_features"],
                                hidden_dim=params["hidden_features"],
                                num_classes=params["num_classes"],
                                node_classifier=True,
                                global_readout=global_readout,
                            )

                            loss_fn = nn.CrossEntropyLoss()
                            optimizer = torch.optim.AdamW(
                                model.parameters(),
                                lr=params["learning_rate"],
                                weight_decay=params["weight_decay"],
                            )
                            losses = []
                            train_accuracies = []
                            val_accuracies = []

                            best_acc = -1.0
                            final_test_acc = -1.0
                            patience = params["patience"]
                            curr_patience = 0
                            test_acc = -1.0

                            for epoch in range(params["num_epochs"]):
                                # training
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
                                losses.append(loss)

                                # evaluation
                                model.eval()

                                y_pred_labels = torch.argmax(y_pred, dim=1)
                                y_true_labels = data.y

                                train_acc = accuracy_score(
                                    y_true_labels[data.train_mask],
                                    y_pred_labels[data.train_mask],
                                )
                                train_accuracies.append(train_acc)

                                val_acc = accuracy_score(
                                    y_true_labels[data.val_mask],
                                    y_pred_labels[data.val_mask],
                                )
                                val_accuracies.append(val_acc)

                                if val_acc > best_acc:
                                    best_acc = val_acc
                                    curr_patience = 0
                                    best_epoch = epoch
                                    test_acc = accuracy_score(
                                        y_true_labels[data.test_mask],
                                        y_pred_labels[data.test_mask],
                                    )
                                    if val_acc > best_model_acc:
                                        best_model_acc = val_acc
                                        best_models[key]["val_acc"] = val_acc
                                        best_models[key]["test_acc"] = test_acc
                                        best_models[key][
                                            "global_readout"
                                        ] = global_readout
                                        best_models[key]["dataset"] = dataset.name
                                        best_models[key]["n_layers"] = n_layers
                                        best_models[key]["hidden_dim"] = hidden_dim
                                        best_models[key][
                                            "learning_rate"
                                        ] = learning_rate
                                        best_models[key]["weight_decay"] = weight_decay
                                        best_models[key]["drop_ratio"] = drop_ratio
                                        best_models[key][
                                            "global_readout"
                                        ] = global_readout

                                        file_name = (
                                            f"GIN_GR_{dataset.name}.pth"
                                            if global_readout
                                            else f"GIN_{dataset.name}.pth"
                                        )
                                        torch.save(
                                            model.state_dict(),
                                            FINAL_MODELS_PATH / file_name,
                                        )

                                else:
                                    curr_patience += 1
                                    if curr_patience == patience:
                                        break

                            print(
                                f"best_acc = {best_acc}, test_acc = {test_acc}, n_epoch = {epoch}"
                            )

    for key, val in best_models.items():
        with open(HYPERPARAMS_PATH / f"GIN_{key}.json", "w") as f:
            json.dump(val, f, indent=4)
        print(f"\nkey = {key}")
        for k, v in val.items():
            print(f"\t{k}: {v}")
