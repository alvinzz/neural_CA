from parameter import Parameter

class Neighbor_Rule(Parameter):
	param_path = "neural_CA/neighbor_rule"
	param_name = "Neighbor_Rule"

	def update_parameters(self):
		self.params = {
			"param_path": Neighbor_Rule.param_path,
			"param_name": Neighbor_Rule.param_name,
		}

	def is_neighbor(self, i, j):
		raise NotImplementedError

class Rect_Neighbor_Rule(Neighbor_Rule):
	param_path = "neural_CA/neighbor_rule"
	param_name = "Rect_Neighbor_Rule"

	def __init__(self):
		self.height = 3
		self.width = 3

		self.connectivity = 4

	def update_parameters(self):
		self.params = {
			"param_path": Rect_Neighbor_Rule.param_path,
			"param_name": Rect_Neighbor_Rule.param_name,

			"height": self.height,
			"width": self.width,

			"connectivity": self.connectivity,
		}

	def is_neighbor(self, i, j):
		self.n_cells = self.height * self.width

		assert max(i, j) < self.n_cells and min(i, j) >= 0, \
			"query ({}, {}) not in bounds for grid with {} cells".format(i, j, self.n_cells)

		assert self.connectivity == 4 or self.connectivity == 8, \
			"connectivity for rectangular grid must be 4 or 8"

		dist = abs(i - j)

		# check horizontal neighbors
		if (i // self.width) == (j // self.width) and dist == 1:
			return True

		# check vertical neighbors
		if dist == self.width:
			return True

		# check diagonal neighbors
		if self.connectivity == 8:
			if abs(i // self.width - j // self.width) == 1 \
			and (dist == self.width + 1 or dist == self.width - 1):
				return True

		return False

class Hex_Neighbor_Rule(Neighbor_Rule):
	param_path = "neural_CA/neighbor_rule"
	param_name = "Hex_Neighbor_Rule"

	def __init__(self):
		self.radius = 3

	def update_parameters(self):
		self.params = {
			"param_path": Hex_Neighbor_Rule.param_path,
			"param_name": Hex_Neighbor_Rule.param_name,

			"radius": self.radius,
		}

	def get_coords(self):
		n_rows = 2 * self.radius - 1
		self.n_cells = 3 * self.radius * (self.radius + 1) + 1
		self.coords = []
		row = -self.radius
		column = -self.radius
		for i in range(self.n_cells):
			self.coords.append((row, column))
			column += 2
			if column + abs(row) > 2 * self.radius:
				row += 1
				column = -2 * self.radius + abs(row)

	def is_neighbor(self, i, j):
		if not hasattr(self, "coords"):
			self.get_coords()

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
			rule.height = h
			rule.width = w

			rule.connectivity = 4
			for i in range(h*w):
				neighbors = []
				for j in range(h*w):
					if rule.is_neighbor(i, j):
						neighbors.append(j)
				print(i, neighbors)

			input()

			print(np.arange(h*w).reshape([h, w]))
			rule.connectivity = 8
			for i in range(h*w):
				neighbors = []
				for j in range(h*w):
					if rule.is_neighbor(i, j):
						neighbors.append(j)
				print(i, neighbors)

			input()

def test_hex_neighbor_rule():
	rule = Hex_Neighbor_Rule()

	for radius in range(1, 4):
		rule.radius = radius

		rule.get_coords()
		str_len = len(str(3 * rule.radius * (rule.radius + 1)))
		l = [['_' * str_len for _ in range(4 * rule.radius + 1)] for _ in range(2 * rule.radius + 1)]
		for (i, (r, c)) in enumerate(rule.coords):
			l[r + rule.radius][c + 2 * rule.radius] = str(i).zfill(str_len)
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

if __name__ == '__main__':
	test_rect_neighbor_rule()
	test_hex_neighbor_rule()
