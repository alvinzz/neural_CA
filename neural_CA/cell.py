import numpy as np

class Cell(object):
    def __init__(self, obs_dim, hidden_dim):
        self.neighbors = []
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.dim = self.obs_dim + self.hidden_dim
        
        self.state = np.zeros(self.dim)

        self.interact_fn = None
        self.apply_fn = None

    def set_state(self, state):
        self.state = state

    def set_random_state(self):
        raise NotImplementedError

    def get_obs(self):
        return self.state[:self.obs_dim]

    def get_hidden(self):
        return self.state[self.obs_dim:]

    def compute_next(self):
        raise NotImplementedError
        # self.next_state = self.state
        # tot_effect = 0
        # for cell in self.neighbors:
        #     tot_effect += self.interact_fn(cell.state, self.state)
        # self.next_state += self.apply_fn(self.state, tot_effect)

    def update(self):
        self.state = self.next_state