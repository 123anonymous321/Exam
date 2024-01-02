import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, Sequential, Linear


class MLPModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        act_fn=nn.ReLU,
    ):
        super(MLPModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.act_fn = act_fn

        hidden_dims = (num_layers - 1) * [hidden_dim]
        dims = [input_dim, *hidden_dims, output_dim]

        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(act_fn())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GIN(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        drop_ratio: float = 0.0,
        node_classifier: bool = False,
        num_mlp_layers: int = 2,
        global_readout: bool = False,
    ) -> None:
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.global_readout = global_readout

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.fn = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        self.fc_gr = torch.nn.ModuleList()

        for layer in range(num_layers):
            nn_seq = Sequential(
                "x",
                [
                    (torch.nn.Linear(hidden_dim, 2 * hidden_dim), "x -> x"),
                    (torch.nn.BatchNorm1d(2 * hidden_dim), "x -> x"),
                    (torch.nn.ReLU(), "x -> x"),
                    (torch.nn.Linear(2 * hidden_dim, hidden_dim), "x -> x"),
                ],
            )
            self.convs.append(GINConv(nn_seq))

            # linear layer for global readout
            self.fc_gr.append(Linear(hidden_dim, hidden_dim))

        self.mlp_head = None
        if node_classifier:
            self.mlp_head = MLPModule(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=num_classes,
                num_layers=num_mlp_layers,
                act_fn=nn.ReLU,
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_list = [self.fn(x)]
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index)
            if self.global_readout:
                global_readout = self.fc_gr[layer](h_list[layer])
                h = h + global_readout

            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            h_list.append(h)

        x = h_list[-1]
        if self.mlp_head is not None:
            x = self.mlp_head(x)
        return x
