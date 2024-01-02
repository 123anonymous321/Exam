import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

if __name__ == "__main__":
    # Load the dataset
    datasets = [
        # Planetoid(root="/tmp/Cora", name="Cora"),
        Planetoid(root="/tmp/Citeseer", name="CiteSeer"),
        # Planetoid(root="/tmp/PubMed", name="PubMed"),
        # Coauthor(root="/tmp/CoauthorCS", name="CS"),
        # Coauthor(root="/tmp/CoauthorPhysics", name="Physics"),
        # PPI(root="/tmp/PPI"),
    ]

    for data in datasets:
        d = data[0]
        print(f"d.num_classes = {data.num_classes}")
        print(f"d.num_nodes = {d.num_nodes}")
        print(f"d.num_edges = {d.num_edges}")
        print(f"d.num_features = {d.num_features}")

        print(f"\ndata = {data}")
        num_connected_components = []
        for d in data:
            g = to_networkx(data[0])
            g_undirected = g.to_undirected()

            num_connected_components.append(
                nx.number_connected_components(g_undirected)
            )

        print(f"len(data) = {len(data)}")
        print(
            f"sum(num_connected_components) / len(data) = {sum(num_connected_components) / len(data)}"
        )
