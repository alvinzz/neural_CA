import numpy as np
import tensorflow as tf

from cell import Cell

from parameter import Parameter
from models.model import Model

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

class TF_Grid(Model):
    param_path = "neural_CA/grid"
    param_name = "TF_Grid"

    def __init__(self):
        self.n_cells = 9

        self.obs_dim = 1
        self.hidden_dim = 0

        self.time_horizon = 1

        self.neighbor_rule = None

        self.tf_grid_model = None

    def update_parameters(self):
        self.params = {
            "param_path": Experiment.param_path,
            "param_name": Experiment.param_name,

            "n_cells": self.n_cells,

            "obs_dim": self.obs_dim,
            "hidden_dim": self.hidden_dim,

            "time_horizon": self.time_horizon,

            "neighbor_rule": self.neighbor_rule,

            "tf_grid_model": self.tf_grid_model,
        }

    # calculates adjacency matrix for the grid based on the Grid.neighbor_rule
    def get_A(self):
        if not hasattr(self, 'A'):
            self.A = np.zeros((self.n_cells, self.n_cells), dtype=np.int32)

            for i in range(self.n_cells):
                for j in range(self.n_cells):
                    if self.neighbor_rule(i, j):
                        self.A[i, j] = 1

            self.effect_inds = np.nonzero(self.A)

    #TODO: returns keras model....
    def build_tf_model(self):
        if not hasattr(self, 'tf_model'):
            self.tf_model = self.tf_grid_model(self)

    def predict(self, inputs):
        if not hasattr(self, 'A'):
            self.get_A()

        if not hasattr(self, 'tf_model'):
            self.build_tf_model()

        pred = self.tf_model(inputs)

        return pred

    def visualize(self):
        raise NotImplementedError

    def visualize_n(self, n):
        raise NotImplementedError

if __name__ == '__main__':
    from tf_grid_model import TF_Grid_Model_v1

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
    def neighbor_rule(i, j):
        return j in d[i]

    grid = TF_Grid()
    grid.neighbor_rule = neighbor_rule
    grid.tf_grid_model = TF_Grid_Model_v1
    grid.hidden_dim = 1
    grid.time_horizon = 10

    inputs = {"grid_obs": tf.constant([
        [[0],[1],[2],[3],[4],[5],[6],[7],[8]],
        [[10],[11],[12],[13],[14],[15],[16],[17],[18]],
        [[20],[21],[22],[23],[24],[25],[26],[27],[28]]], dtype=tf.float32)}

    #with tf.GradientTape() as grad_tape:
    #    pred = grid.predict(inputs)
    #    loss = tf.linalg.norm(pred)
    #gradients = grad_tape.gradient(loss, grid.tf_grid_model.trainable_variables)

    #print(list(grid.tf_grid_model.trainable_variables))

    pred = grid.predict(inputs)
    print(pred[0])
    print(pred[1])
    print(pred[2])
