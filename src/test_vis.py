import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

# Mock function to convert data to a networkx graph (replace with actual implementation)
def to_networkx(data):
    G = nx.Graph()
    G.add_edges_from(data)  # Assuming `data` is in edge list format
    return G

class GraphVisualizer:
    def __init__(self, data, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2):
        self.data = data
        self.x_list1 = x_list1
        self.y_list1 = y_list1
        self.z_list1 = z_list1
        self.x_list2 = x_list2
        self.y_list2 = y_list2
        self.z_list2 = z_list2
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.sc1 = None
        self.sc2 = None

        self.toggle1_visible = True
        self.toggle2_visible = True  # Start with both datasets visible

        self.draw_plots()
        self.add_buttons()  # Add buttons after plotting

    def draw_plots(self):
        self.ax.clear()  # Clear the axes for redrawing
        vis1 = to_networkx(self.data)
        pos_3d1 = {node: (x, -z, y) for node, x, y, z in zip(vis1.nodes(), self.x_list1, self.y_list1, self.z_list1)}
        # Dataset 1
        if self.toggle1_visible:
            self.sc1 = self.ax.scatter(*zip(*pos_3d1.values()), s=120, c='b', depthshade=True)
            for node in vis1.nodes():
                for neighbor in vis1.neighbors(node):
                    x1, y1, z1 = pos_3d1[node]
                    x2, y2, z2 = pos_3d1[neighbor]
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], c='b')

        vis2 = to_networkx(self.data)
        pos_3d2 = {node: (x, -z, y) for node, x, y, z in zip(vis2.nodes(), self.x_list2, self.y_list2, self.z_list2)}
        # Dataset 2
        if self.toggle2_visible:
            self.sc2 = self.ax.scatter(*zip(*pos_3d2.values()), s=120, c='r', depthshade=True)
            for node in vis2.nodes():
                for neighbor in vis2.neighbors(node):
                    x1, y1, z1 = pos_3d2[node]
                    x2, y2, z2 = pos_3d2[neighbor]
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2], c='r')

        self.ax.set_title('3D Visualization of Graph')
        self.ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        # self.ax.axes.set_aspect('equal')
        plt.tight_layout()
        plt.draw()  # Redraw the updated figure

    def add_buttons(self):
        # Define buttons after setting up the figure
        ax_button1 = plt.axes([0.1, 0.01, 0.1, 0.05])
        self.button1 = Button(ax_button1, 'Toggle Data 1')  # Store reference
        self.button1.on_clicked(self.toggle_data1)

        ax_button2 = plt.axes([0.22, 0.01, 0.1, 0.05])
        self.button2 = Button(ax_button2, 'Toggle Data 2')  # Store reference
        self.button2.on_clicked(self.toggle_data2)

    def toggle_data1(self, event):
        print("Toggling data 1")
        self.toggle1_visible = not self.toggle1_visible
        self.draw_plots()  # Redraw after toggling visibility

    def toggle_data2(self, event):
        print("Toggling data 2")
        self.toggle2_visible = not self.toggle2_visible
        self.draw_plots()  # Redraw after toggling visibility

def visualize_graphs(sample_data1, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2):
    visualizer = GraphVisualizer(sample_data1, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2)
    plt.show(block=True)  # Make sure to keep the plot open

# Sample data for testing
sample_data1 = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Example edges
x_list1 = [0, 1, 1, 0]
y_list1 = [0, 0, 1, 1]
z_list1 = [0, 0, 0, 0]
x_list2 = [0.5, 1.5, 1.5, 0.5]
y_list2 = [0.5, 0.5, 1.5, 1.5]
z_list2 = [1, 1, 1, 1]

# Call the visualization function
visualize_graphs(sample_data1, x_list1, y_list1, z_list1, x_list2, y_list2, z_list2)
