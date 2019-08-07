import numpy as np
import tensorflow as tf

from cell import Cell

from parameter import Parameter

class Grid(object):
    def __init__(self, n_cells):
        self.n_cells = n_cells
        self.A = np.zeros((self.n_cells, self.n_cells))
        self.cells = [Cell()]

        self.obs_dim = self.cells[0].obs_dim
        self.hidden_dim = self.cells[0].hidden_dim
        self.dim = self.obs_dim + self.hidden_dim

    def set_cell_neighbors(self):
        for i in range(self.n_cells):
            cell = self.cells[i]
            cell.neighbors = []
            for j in self.A[i]:
                if j:
                    cell.neighbors.append(self.cells[j])

    def get_A_from_cell_neighbors(self):
        self.A = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            cell = self.cells[i]
            for j in cell.neighbors:
                self.A[i, j] = 1

    def update(self):
        for cell in self.cells:
            cell.compute_next()
        for cell in self.cells:
            cell.update()

    def visualize(self):
        pass

    def visualize_n(self, n):
        pass

class TF_Grid(Parameter):
    def __init__(self):
        self.state_dim = 2
        self.n_cells = 9

        self.A = np.zeros((self.n_cells, self.n_cells), dtype=np.int32)
        d = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        for (cell, neighbors_list) in d.items():
            for neighbor in neighbors_list:
                self.A[cell, neighbor] = 1
        self.interaction_inds = np.nonzero(self.A)

        # self.cell_states = tf.Variable(shape=(self.state_dim, self.n_cells))
        self.cell_states = tf.reshape(tf.constant([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=tf.float32), [-1])

    def update(self):
        # raise NotImplementedError
        interaction_matrix = tf.nn.embedding_lookup(self.cell_states, self.interaction_inds)
        cells = interaction_matrix[0]
        neighbors = interaction_matrix[1]
        # effects = MLP(tf.concatenate([cells, neighbors, MLP(cells)*MLP(neighbors)], axis=-1))
        effects = neighbors
        tot_effects = tf.math.segment_sum(effects, self.interaction_inds[0])
        # self.cell_states = MLP(self.cell_states, tot_effects)
        self.cell_states = tot_effects

if __name__ == '__main__':
    grid = TF_Grid()
    print(tf.reshape(grid.cell_states, [3, 3]))
    grid.update()
    print(tf.reshape(grid.cell_states, [3, 3]))
    grid.update()
    print(tf.reshape(grid.cell_states, [3, 3]))
    grid.update()
    print(tf.reshape(grid.cell_states, [3, 3]))
