import numpy as np

from cell import Cell

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
