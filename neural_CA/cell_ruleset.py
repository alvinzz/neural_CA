import numpy as np

from parameter import Parameter

class Cell_Ruleset(Parameter):
    param_path = "neural_CA.cell_ruleset"
    param_name = "Cell_Ruleset"

    def __init__(self):
        self.global_params = set([])
        self.params = set([])
        self.shared_params = set([
            "obs_dim",
            "grid_hidden_dim",
        ])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        pass

    def random_state(self):
        raise NotImplementedError

    def compute_next(self, cell_state, neighbor_states):
        raise NotImplementedError
        # self.next_state = self.state
        # tot_effect = 0
        # for neighbor in self.neighbors:
        #     tot_effect += self.interact_fn(neighbor.state, self.state)
        # self.next_state += self.apply_fn(self.state, tot_effect)

class XOR_Cell_Ruleset(Cell_Ruleset):
    param_path = "neural_CA.cell_ruleset"
    param_name = "XOR_Cell_Ruleset"

    def __init__(self):
        self.global_params = set([])
        self.params = set([])
        self.shared_params = set([
            "obs_dim",
            "grid_hidden_dim",
        ])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        assert self.obs_dim == 1, "XOR_Cell_Ruleset requires obs_dim = 1"

        self.state_dim = self.obs_dim + self.grid_hidden_dim

    def random_state(self):
        return np.random.randint(2, size=(self.state_dim))

    def compute_next(self, cell_state, neighbor_states):
        cell_state = cell_state.astype(np.int32)
        neighbor_states = neighbor_states.astype(np.int32)

        cell_next_state = np.zeros_like(cell_state)

        cell_next_state[1:] = cell_state[:-1]

        cell_next_state[0] = cell_state[0]
        for past in cell_state[1:]:
            cell_next_state[0] ^= past
        for neighbor in neighbor_states:
            for other in neighbor:
                cell_next_state[0] ^= other

        return cell_next_state
