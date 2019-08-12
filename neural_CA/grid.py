import numpy as np

from parameter import Parameter

class Grid(Parameter):
    param_path = "neural_CA.grid"
    param_name = "Grid"

    def __init__(self):
        self.global_params = [
            "grid_hidden_dim",
        ]
        self.params = [
            "cell_ruleset",
        ]
        self.shared_params = [
            "obs_dim",

            "neighbor_rule",
        ]

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.state_dim = self.obs_dim + self.grid_hidden_dim
        self.n_cells = self.neighbor_rule.n_cells
        self.A = self.neighbor_rule.A

        self.set_random_state()

        self.cells_neighbors = []
        for i in range(self.n_cells):
            cell_neighbors = []
            for j in range(self.n_cells):
                if self.A[i, j]:
                    cell_neighbors.append(j)
            self.cells_neighbors.append(cell_neighbors)

    def set_random_state(self):
        self.cell_states = np.array([self.cell_ruleset.random_state()
            for _ in range(self.n_cells)])
        self.next_cell_states = np.array([np.zeros(self.state_dim) for _ in range(self.n_cells)])

    def update(self):
        for i in range(self.n_cells):
            cell_state = self.cell_states[i].copy()
            neighbor_states = self.cell_states[self.cells_neighbors[i]].copy()
            self.next_cell_states[i] = self.cell_ruleset.compute_next(cell_state, neighbor_states)

        self.cell_states = self.next_cell_states.copy()

    def visualize(self, grid_visualizer):
        state = self.get_state()
        grid_visualizer.visualize(state, pause=True)

    def visualize_dim(self, dim, grid_layer_visualizer):
        state = self.get_state()
        layer_state = state[:, dim]
        grid_layer_visualizer.visualize(state, pause=True)

    def get_obs(self):
        return np.expand_dims(self.cell_states[:, :self.obs_dim].astype(np.float32), 0)

    def get_state(self):
        return np.expand_dims(self.cell_states.astype(np.float32), 0)