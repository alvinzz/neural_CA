from parameter import Parameter

import numpy as np

class Neighbor_Rule(Parameter):
    param_path = "neural_CA.neighbor_rule"
    param_name = "Neighbor_Rule"

    def __init__(self):
        self.global_params = set([])
        self.params = set([])
        self.shared_params = set([])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def is_neighbor(self, i, j):
        raise NotImplementedError

    def _build(self):
        self.n_cells = 0
        self.get_A()
        raise NotImplementedError

    # calculates adjacency matrix
    def get_A(self):
        self.A = np.zeros((self.n_cells, self.n_cells), dtype=np.int32)

        for i in range(self.n_cells):
            for j in range(self.n_cells):
                if self.is_neighbor(i, j):
                    self.A[i, j] = 1

class Rect_Neighbor_Rule(Neighbor_Rule):
    param_path = "neural_CA.neighbor_rule"
    param_name = "Rect_Neighbor_Rule"

    def __init__(self):
        self.global_params = set([])
        self.params = set([
            "grid_height",
            "grid_width",
            "grid_connectivity",
        ])
        self.shared_params = set([])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.n_cells = self.grid_width * self.grid_height

        self.get_A()

    def is_neighbor(self, i, j):
        self.n_cells = self.grid_height * self.grid_width

        assert max(i, j) < self.n_cells and min(i, j) >= 0, \
            "query ({}, {}) not in bounds for grid with {} cells".format(i, j, self.n_cells)

        assert self.grid_connectivity == 4 or self.grid_connectivity == 8, \
            "grid_connectivity for rectangular grid must be 4 or 8"

        dist = abs(i - j)

        # check horizontal neighbors
        if (i // self.grid_width) == (j // self.grid_width) and dist == 1:
            return True

        # check vertical neighbors
        if dist == self.grid_width:
            return True

        # check diagonal neighbors
        if self.grid_connectivity == 8:
            if abs(i // self.grid_width - j // self.grid_width) == 1 \
            and (dist == self.grid_width + 1 or dist == self.grid_width - 1):
                return True

        return False

class Hex_Neighbor_Rule(Neighbor_Rule):
    param_path = "neural_CA.neighbor_rule"
    param_name = "Hex_Neighbor_Rule"

    def __init__(self):
        self.global_params = set([])
        self.params = set([
            "grid_radius",
        ])
        self.shared_params = set([])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.n_cells = 3 * self.grid_radius * (self.grid_radius - 1) + 1

        self.coords = []
        n_rows = 2 * self.grid_radius - 1
        row = -(self.grid_radius - 1)
        column = -(self.grid_radius - 1)
        for i in range(self.n_cells):
            self.coords.append((row, column))
            column += 2
            if column + abs(row) > 2 * (self.grid_radius - 1):
                row += 1
                column = -2 * (self.grid_radius - 1) + abs(row)

        self.get_A()

    def is_neighbor(self, i, j):
        assert max(i, j) < self.n_cells and min(i, j) >= 0, \
            "query ({}, {}) not in bounds for grid with {} cells".format(i, j, self.n_cells)

        if self.coords[i][0] == self.coords[j][0] and abs(self.coords[i][1] - self.coords[j][1]) == 2:
            return True

        if abs(self.coords[i][0] - self.coords[j][0]) == 1 and abs(self.coords[i][1] - self.coords[j][1]) == 1:
            return True

        return False

def test_rect_neighbor_rule():
    import numpy as np
    rule = Rect_Neighbor_Rule()

    for h in range(1, 5):
        for w in range(1, h+1):
            print(np.arange(h*w).reshape([h, w]))
            rule.grid_height = h
            rule.grid_width = w

            rule.grid_connectivity = 4
            for i in range(h*w):
                neighbors = []
                for j in range(h*w):
                    if rule.is_neighbor(i, j):
                        neighbors.append(j)
                print(i, neighbors)

            input()

            print(np.arange(h*w).reshape([h, w]))
            rule.grid_connectivity = 8
            for i in range(h*w):
                neighbors = []
                for j in range(h*w):
                    if rule.is_neighbor(i, j):
                        neighbors.append(j)
                print(i, neighbors)

            input()

def test_hex_neighbor_rule():
    rule = Hex_Neighbor_Rule()

    for grid_radius in range(1, 5):
        rule.grid_radius = grid_radius

        rule.get_coords()
        str_len = len(str(3 * rule.grid_radius * (rule.grid_radius - 1)))
        l = [['_' * str_len for _ in range(4 * (rule.grid_radius - 1) + 1)] for _ in range(2 * rule.grid_radius - 1)]
        for (i, (r, c)) in enumerate(rule.coords):
            l[r + rule.grid_radius - 1][c + 2 * (rule.grid_radius - 1)] = str(i).zfill(str_len)
        for row in l:
            res = ''
            for s in row:
                res += s
            print(res)

        for i in range(rule.n_cells):
            neighbors = []
            for j in range(rule.n_cells):
                if rule.is_neighbor(i, j):
                    neighbors.append(j)
            print(i, neighbors)

        input()

if __name__ == "__main__":
    test_rect_neighbor_rule()
    test_hex_neighbor_rule()
