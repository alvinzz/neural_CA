from parameter import Parameter

import matplotlib.pyplot as plt

import numpy as np

class Grid_Visualizer(Parameter):
    param_path = "neural_CA.grid_visualizer"
    param_name = "Grid_Visualizer"

    def __init__(self):
        self.grid_layer_visualizer = None

    def update_parameters(self):
        self.params = {
            "param_path": Grid_Visualizer.param_path,
            "param_name": Grid_Visualizer.param_name,

            "grid_layer_visualizer": self.grid_layer_visualizer,
        }

    def visualize(self, state):
        raise NotImplementedError

class All_Layers_Grid_Visualizer(Grid_Visualizer):
    param_path = "neural_CA.grid_visualizer"
    param_name = "All_Layers_Grid_Visualizer"

    def visualize(self, state, pause=False):
        assert len(state.shape) == 3, "expected state of shape [time, n_cells, obs_dim]"

        plt.ion()

        figs, axes = [], []

        for dim in range(state.shape[2]):
            fig, ax = self.grid_layer_visualizer.init_plot(state[:, :, dim])
            figs.append(fig)
            axes.append(ax)

            min_val = np.min(state[:, :, dim])
            max_val = np.max(state[:, :, dim])

            state[:, :, dim] -= min_val
            state[:, :, dim] /= (max_val - min_val)
            state[:, :, dim] *= 255
        state = state.astype(np.int32)

        for t in range(state.shape[0]):
            for dim in range(state.shape[2]):
                self.grid_layer_visualizer.render_t(axes[dim], t, state[t, :, dim])
                axes[dim].set_title("d={}, ".format(dim) + axes[dim].get_title())
                figs[dim].canvas.draw()
            
            if pause:
                input()
            else:
                plt.pause(0.2)

        for fig in figs:
            plt.close(fig)

if __name__ == "__main__":
    visualizer = All_Layers_Grid_Visualizer()
    from grid_layer_visualizer import *
    visualizer.grid_layer_visualizer = Hex_Grid_Layer_Visualizer()
    visualizer.grid_layer_visualizer.grid_radius = 3

    state = np.stack([
        np.linspace(-19, 19, 9*19).reshape([9, 19]),
        np.linspace(19, -19, 9*19).reshape([9, 19]),
        np.linspace(-0, 0, 9*19).reshape([9, 19])], 2)

    visualizer.visualize(state)