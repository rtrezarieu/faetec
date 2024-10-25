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
    def __init__(self, data, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2, x_pred_list, y_pred_list, z_pred_list, x_forces_list, y_forces_list, z_forces_list, supports_list):
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
        self.x_forces_list = x_forces_list
        self.y_forces_list = y_forces_list
        self.z_forces_list = z_forces_list
        self.supports_list = supports_list
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.sc1 = None
        self.sc2 = None
        self.sc3 = None
        self.sc4 = None

        self.toggle1_visible = True
        self.toggle2_visible = False
        self.toggle3_visible = False
        self.toggle4_visible = False
        self.toggle5_visible = False

        self.draw_plots()
        self.add_buttons()

    def draw_plots(self):
        self.ax.clear()
        # Plot the first dataset
        vis = to_networkx(self.data)
        pos_3d1 = {node: (x, -z, y) for node, x, y, z in zip(vis.nodes(), self.x_list1, self.y_list1, self.z_list1)}
        if self.toggle1_visible:
            self.sc1 = self.ax.scatter(*zip(*pos_3d1.values()), s=120, c='b', depthshade=True)
            for node in vis.nodes():
                for neighbor in vis.neighbors(node):
                    x1, y1, z1 = pos_3d1[node]
                    x2, y2, z2 = pos_3d1[neighbor]
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], c='b')

        pos_3d2 = {node: (x, -z, y) for node, x, y, z in zip(vis.nodes(), self.x_list2, self.y_list2, self.z_list2)}
        if self.toggle2_visible:
            self.sc2 = self.ax.scatter(*zip(*pos_3d2.values()), s=120, c='r', depthshade=True)
            for node in vis.nodes():
                for neighbor in vis.neighbors(node):
                    x1, y1, z1 = pos_3d2[node]
                    x2, y2, z2 = pos_3d2[neighbor]
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], c='r')

        if self.x_pred_list is not None:
            pos_3d3 = {node: (x, -z, y) for node, x, y, z in zip(vis.nodes(), self.x_pred_list, self.y_pred_list, self.z_pred_list)}
            if self.toggle3_visible:
                self.sc3 = self.ax.scatter(*zip(*pos_3d3.values()), s=120, c='g', depthshade=True)
                for node in vis.nodes():
                    for neighbor in vis.neighbors(node):
                        x1, y1, z1 = pos_3d3[node]
                        x2, y2, z2 = pos_3d3[neighbor]
                        self.ax.plot([x1, x2], [y1, y2], [z1, z2], c='g')         

        if self.x_forces_list is not None and self.y_forces_list is not None and self.z_forces_list is not None:
            pos_3d4 = {node: (x, -z, y) for node, x, y, z in zip(vis.nodes(), self.x_list1, self.y_list1, self.z_list1)}
            if self.toggle4_visible:
                self.sc4 = self.ax.scatter(*zip(*pos_3d4.values()), s=120, c='orange', depthshade=True)

                max_amplitude = max((fx**2 + fy**2 + fz**2)**0.5 for fx, fy, fz in zip(self.x_forces_list, self.y_forces_list, self.z_forces_list))

                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                zlim = self.ax.get_zlim()
                plot_size = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
                desired_max_length = 5e-3
                scaling_factor = desired_max_length / max_amplitude

                for node in vis.nodes():
                    x, y, z = pos_3d4[node]
                    fx, fy, fz = self.x_forces_list[node], -self.z_forces_list[node], self.y_forces_list[node]
                    amplitude = (fx**2 + fy**2 + fz**2)**0.5
                    normalized_length = amplitude * scaling_factor
                    self.ax.quiver(x, y, z, fx, fy, fz, color='orange', length=normalized_length, normalize=False)

        if self.supports_list is not None:
            pos_3d5 = {node: (x, -z, y) for node, x, y, z in zip(vis.nodes(), self.x_list1, self.y_list1, self.z_list1)}
            if self.toggle5_visible:
                support_nodes = [node for node in vis.nodes() if self.supports_list[node]]
                support_positions = [pos_3d5[node] for node in support_nodes]
                self.sc5 = self.ax.scatter(*zip(*support_positions), s=120, c='purple', depthshade=False)
                # # Annotate the support nodes
                # for node in support_nodes:
                #     x, y, z = pos_3d5[node]
                #     self.ax.text(x, y, z, f'{node}', color='purple')

        self.ax.set_title('3D Visualization of Graph', fontsize=16)
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        self.ax.axes.set_aspect('equal')
        plt.tight_layout()
        plt.draw()

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

        ax_button4 = plt.axes([0.46, 0.01, 0.1, 0.05])
        self.button4 = Button(ax_button4, 'Forces')
        self.button4.on_clicked(self.toggle_data4)

        ax_button5 = plt.axes([0.58, 0.01, 0.1, 0.05])
        self.button5 = Button(ax_button5, 'Supports')
        self.button5.on_clicked(self.toggle_data5)       

    def toggle_data1(self, event):
        self.toggle1_visible = not self.toggle1_visible
        self.draw_plots()

    def toggle_data2(self, event):
        self.toggle2_visible = not self.toggle2_visible
        self.draw_plots()

    def toggle_data3(self, event):
        self.toggle3_visible = not self.toggle3_visible
        self.draw_plots()

    def toggle_data4(self, event):
        self.toggle4_visible = not self.toggle4_visible
        self.draw_plots()
    
    def toggle_data5(self, event):
        self.toggle5_visible = not self.toggle5_visible
        self.draw_plots()

def visualize_graphs(sample_data1, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2, x_pred_list, y_pred_list, z_pred_list, x_forces_list, y_forces_list, z_forces_list, supports_list):
    visualizer = GraphVisualizer(sample_data1, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2, x_pred_list, y_pred_list, z_pred_list, x_forces_list, y_forces_list, z_forces_list, supports_list)
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