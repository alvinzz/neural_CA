from parameter import Parameter

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import RegularPolygon

import numpy as np

class Grid_Layer_Visualizer(Parameter):
    param_path = "neural_CA.grid_layer_visualizer"
    param_name = "Grid_Layer_Visualizer"

    def update_parameters(self):
        self.params = {
            "param_path": Grid_Layer_Visualizer.param_path,
            "param_name": Grid_Layer_Visualizer.param_name,
        }

    def visualize(self, layer_state):
        raise NotImplementedError

    def init_plot(self, layer_state):
        raise NotImplementedError

    def render_t(self, ax, t, t_layer_state):
        raise NotImplementedError

class Rect_Grid_Layer_Visualizer(Grid_Layer_Visualizer):
    param_path = "neural_CA.grid_layer_visualizer"
    param_name = "Rect_Grid_Layer_Visualizer"

    def __init__(self):
        self.grid_height = 3
        self.grid_width = 3

        self.cmap = get_cmap('bwr')

    def update_parameters(self):
        self.params = {
            "param_path": Rect_Grid_Layer_Visualizer.param_path,
            "param_name": Rect_Grid_Layer_Visualizer.param_name,
        
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
        }

    def visualize(self, layer_state):
        plt.ion()

        fig, ax = self.init_plot(layer_state)

        assert layer_state.shape[1] == self.grid_height * self.grid_width, \
            "expected layer_state of shape [time, {}]".format(self.grid_height * self.grid_width)

        min_val = np.min(layer_state)
        max_val = np.max(layer_state)

        layer_state -= min_val
        layer_state /= (max_val - min_val)
        layer_state *= 255
        layer_state = layer_state.astype(np.int32)

        for t in range(layer_state.shape[0]):
            self.render_t(ax, t, layer_state[t])
            
            fig.canvas.draw()
            plt.pause(0.2)

        plt.close()

    def init_plot(self, layer_state):
        fig = plt.figure()
        ax = plt.axes()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # hack to get colorbar
        min_val = np.min(layer_state)
        max_val = np.max(layer_state)

        im = ax.scatter([0, 0], [0, 0], c=[min_val, max_val], cmap=self.cmap)
        fig.colorbar(im, ax=ax)

        # add polygons to axes
        for h in range(self.grid_height):
            for w in range(self.grid_width):
                ax.add_patch(
                    RegularPolygon(
                        (w, -h), 4, np.sqrt(2)/2, orientation=np.pi/4,
                        color=self.cmap(127)))

        ax.axis('equal')

        return fig, ax

    def render_t(self, ax, t, t_layer_state):
        for h in range(self.grid_height):
            for w in range(self.grid_width):
                i = h * self.grid_width + w
                ax.patches[i].set_color(self.cmap(t_layer_state[i]))

        ax.set_title('t={}'.format(t))

class Hex_Grid_Layer_Visualizer(Grid_Layer_Visualizer):
    param_path = "neural_CA.grid_layer_visualizer"
    param_name = "Hex_Grid_Layer_Visualizer"

    def __init__(self):
        self.grid_radius = 3

        self.cmap = get_cmap('bwr')

        self.hex_grid_height = 1.5/np.sqrt(3)
        self.hex_grid_width = 0.5
        self.hex_size = 1/np.sqrt(3)

    def update_parameters(self):
        self.params = {
            "param_path": Hex_Grid_Layer_Visualizer.param_path,
            "param_name": Hex_Grid_Layer_Visualizer.param_name,
        
            "grid_radius": self.grid_radius,
        }

    def get_coords(self):
        n_rows = 2 * self.grid_radius - 1
        self.n_cells = 3 * self.grid_radius * (self.grid_radius - 1) + 1
        self.coords = []
        row = -(self.grid_radius - 1)
        column = -(self.grid_radius - 1)
        for i in range(self.n_cells):
            self.coords.append((row, column))
            column += 2
            if column + abs(row) > 2 * (self.grid_radius - 1):
                row += 1
                column = -2 * (self.grid_radius - 1) + abs(row)

    def visualize(self, layer_state, pause=False):
        plt.ion()

        fig, ax = self.init_plot(layer_state)

        assert layer_state.shape[1] == self.n_cells, \
            "expected layer_state of shape [time, {}]".format(self.n_cells)

        min_val = np.min(layer_state)
        max_val = np.max(layer_state)

        layer_state -= min_val
        layer_state /= (max_val - min_val)
        layer_state *= 255
        layer_state = layer_state.astype(np.int32)

        for t in range(layer_state.shape[0]):
            self.render_t(ax, t, layer_state[t])
            
            fig.canvas.draw()
            if pause:
                input()
            else:
                plt.pause(0.2)


        plt.close()

    def init_plot(self, layer_state):
        if not hasattr(self, "coords"):
            self.get_coords()

        fig = plt.figure()
        ax = plt.axes()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # hack to get colorbar
        min_val = np.min(layer_state)
        max_val = np.max(layer_state)

        im = ax.scatter([0, 0], [0, 0], c=[min_val, max_val], cmap=self.cmap)
        fig.colorbar(im, ax=ax)

        # add polygons to axes
        for i in range(self.n_cells):
            y, x = self.coords[i]
            ax.add_patch(
                RegularPolygon(
                    (x * self.hex_grid_width, -y * self.hex_grid_height), 6, self.hex_size,
                    color=self.cmap(127)))

        ax.axis('equal')

        return fig, ax

    def render_t(self, ax, t, t_layer_state):
        for i in range(self.n_cells):
            ax.patches[i].set_color(self.cmap(t_layer_state[i]))

        ax.set_title('t={}'.format(t))

if __name__ == "__main__":
    visualizer = Hex_Grid_Layer_Visualizer()
    visualizer.grid_radius = 3
    visualizer.visualize(np.linspace(-19, 19, 9*19).reshape([9, 19]).astype(np.float32))

    visualizer = Rect_Grid_Layer_Visualizer()
    visualizer.grid_height = 4
    visualizer.grid_width = 4
    visualizer.visualize(np.linspace(-19, 19, 16*9).reshape([9, 16]).astype(np.float32))
