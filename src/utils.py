import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import numpy as np

def pyg2_data_transform(data: Data):
    # convert pyg data to Data class for version compatibility (2.0)
    source = data.__dict__
    if "_store" in source:
        source = source["_store"]
    return Data(**{k: v for k, v in source.items() if v is not None})

# for one unique 3D structure
def visualize_graph_as_3D_structure(data, x_list, y_list, z_list, color):
    vis = to_networkx(data)

    pos_3d = {}
    for node, x, y, z in zip(vis.nodes(), x_list, y_list, z_list):
        pos_3d[node] = (x, -z, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for node, (x, y, z) in pos_3d.items():
        ax.scatter(x, y, z, s=120, c='b', depthshade=True)
        for neighbor in vis.neighbors(node):
            x2, y2, z2 = pos_3d[neighbor]
            ax.plot([x, x2], [y, y2], [z, z2], c=color)

    ax.axes.set_aspect('equal')

    ax.set_title('3D Visualization of Graph')
    plt.tight_layout()
    plt.show()

class GraphVisualizer:
    def __init__(self, data, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2, x_pred_list, y_pred_list, z_pred_list, show_vectors):
        self.data = data
        self.x_list1 = x_list1
        self.y_list1 = y_list1
        self.z_list1 = z_list1
        self.x_list2 = x_list2
        self.y_list2 = y_list2
        self.z_list2 = z_list2
        self.x_pred_list = x_pred_list
        self.y_pred_list = y_pred_list
        self.z_pred_list = z_pred_list
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.sc1 = None
        self.sc2 = None

        self.toggle1_visible = True
        self.toggle2_visible = False
        self.toggle3_visible = False

        self.draw_plots()
        self.add_buttons()

    def draw_plots(self):
        self.ax.clear()
        # Plot the first dataset
        vis1 = to_networkx(self.data)
        pos_3d1 = {node: (x, -z, y) for node, x, y, z in zip(vis1.nodes(), self.x_list1, self.y_list1, self.z_list1)}
        if self.toggle1_visible:
            self.sc1 = self.ax.scatter(*zip(*pos_3d1.values()), s=120, c='b', depthshade=True)
            for node in vis1.nodes():
                for neighbor in vis1.neighbors(node):
                    x1, y1, z1 = pos_3d1[node]
                    x2, y2, z2 = pos_3d1[neighbor]
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], c='b')

        vis2 = to_networkx(self.data)
        pos_3d2 = {node: (x, -z, y) for node, x, y, z in zip(vis2.nodes(), self.x_list2, self.y_list2, self.z_list2)}
        if self.toggle2_visible:
            self.sc2 = self.ax.scatter(*zip(*pos_3d2.values()), s=120, c='r', depthshade=True)
            for node in vis2.nodes():
                for neighbor in vis2.neighbors(node):
                    x1, y1, z1 = pos_3d2[node]
                    x2, y2, z2 = pos_3d2[neighbor]
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], c='r')

        vis3 = to_networkx(self.data)
        pos_3d3 = {node: (x, -z, y) for node, x, y, z in zip(vis3.nodes(), self.x_pred_list, self.y_pred_list, self.z_pred_list)}
        if self.toggle3_visible:
            self.sc3 = self.ax.scatter(*zip(*pos_3d3.values()), s=120, c='g', depthshade=True)
            for node in vis3.nodes():
                for neighbor in vis3.neighbors(node):
                    x1, y1, z1 = pos_3d3[node]
                    x2, y2, z2 = pos_3d3[neighbor]
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], c='g')       

        self.ax.set_title('3D Visualization of Graph')
        # self.ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        self.ax.axes.set_aspect('equal')
        plt.tight_layout()
        plt.draw()  # Redraw the updated figure

    def add_buttons(self):
        ax_button1 = plt.axes([0.1, 0.01, 0.1, 0.05])
        self.button1 = Button(ax_button1, 'Base')
        self.button1.on_clicked(self.toggle_data1)

        ax_button2 = plt.axes([0.22, 0.01, 0.1, 0.05])
        self.button2 = Button(ax_button2, 'Target')
        self.button2.on_clicked(self.toggle_data2)

        ax_button3 = plt.axes([0.34, 0.01, 0.1, 0.05])
        self.button3 = Button(ax_button3, 'Prediction')
        self.button3.on_clicked(self.toggle_data3)       

    def toggle_data1(self, event):
        self.toggle1_visible = not self.toggle1_visible
        self.draw_plots()

    def toggle_data2(self, event):
        self.toggle2_visible = not self.toggle2_visible
        self.draw_plots()

    def toggle_data3(self, event):
        self.toggle3_visible = not self.toggle3_visible
        self.draw_plots()

def visualize_graphs(sample_data1, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2, x_pred_list, y_pred_list, z_pred_list, show_vectors=False):
    visualizer = GraphVisualizer(sample_data1, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2, x_pred_list, y_pred_list, z_pred_list, show_vectors=show_vectors)
    plt.show(block=True)




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