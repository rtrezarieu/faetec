import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx

def pyg2_data_transform(data: Data):
    # convert pyg data to Data class for version compatibility (2.0)
    source = data.__dict__
    if "_store" in source:
        source = source["_store"]
    return Data(**{k: v for k, v in source.items() if v is not None})


def visualize_graph_as_3D_structure(data, x_list, y_list, z_list):
    vis = to_networkx(data)

    pos_3d = {}
    for node, x, y, z in zip(vis.nodes(), x_list, y_list, z_list):
        pos_3d[node] = (x, z, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for node, (x, y, z) in pos_3d.items():
        ax.scatter(x, z, y, s=120, c='b', depthshade=True)
        for neighbor in vis.neighbors(node):
            x2, y2, z2 = pos_3d[neighbor]
            ax.plot([x, x2], [z, z2], [y, y2], c='k')

    print('mass', data.x[:10][8])
    print('loads', data.x[:10][9]) # Chargement unitaire en tête de structure? Dans quelle direction?

    ax.set_title('3D Visualization of Graph')
    plt.tight_layout()
    plt.show()


# Adapted from torchvision.transforms - avoid importing the whole module
class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string