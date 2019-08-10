from parameter import Parameter

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import RegularPolygon

import numpy as np

class Grid_Layer_Visualizer(Parameter):
	param_path = "neural_CA/grid_layer_visualizer"
	param_name = "Grid_Layer_Visualizer"

	def update_parameters(self):
		self.params = {
			"param_path": Grid_Layer_Visualizer.param_path,
			"param_name": Grid_Layer_Visualizer.param_name,
		}

	def visualize(self, layer_state):
		raise NotImplementedError

class Rect_Grid_Layer_Visualizer(Grid_Layer_Visualizer):
	param_path = "neural_CA/grid_layer_visualizer"
	param_name = "Rect_Grid_Layer_Visualizer"

	def __init__(self):
		self.height = 3
		self.width = 3

		self.cmap = get_cmap('bwr')

	def update_parameters(self):
		self.params = {
			"param_path": Rect_Grid_Layer_Visualizer.param_path,
			"param_name": Rect_Grid_Layer_Visualizer.param_name,
		
			"height": self.height,
			"width": self.width,
		}

	# layer_state is [time_horizon, Grid.n_cells]
	def visualize(self, layer_state):
		assert layer_state.shape[1] == self.height * self.width, \
			"expected layer_state of shape [time_horizon, {}]".format(self.height * self.width)

		min_val = np.min(layer_state)
		max_val = np.max(layer_state)

		layer_state -= min_val
		layer_state /= (max_val - min_val)
		layer_state *= 255
		layer_state = layer_state.astype(np.int32)

		plt.ion()

		fig = plt.figure()
		ax = plt.axes()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# hack to get colorbar
		im = ax.scatter([0,0], [0,0], c=[min_val,max_val], cmap=self.cmap)
		fig.colorbar(im, ax=ax)

		for t in range(layer_state.shape[0]):
			for i in range(layer_state.shape[1]):
				y, x = i // self.width, i % self.width
				if t == 0:
					ax.add_patch(
						RegularPolygon(
							(x, -y), 4, np.sqrt(2)/2, orientation=np.pi/4,
							color=self.cmap(layer_state[t, i])))
				else:
					ax.patches[i].set_color(self.cmap(layer_state[t, i]))

			ax.set_title('t={}'.format(t))
			if t == 0:
				ax.axis('equal')
			
			fig.canvas.draw()
			plt.pause(0.2)

		plt.close()

class Hex_Grid_Layer_Visualizer(Grid_Layer_Visualizer):
	param_path = "neural_CA/grid_layer_visualizer"
	param_name = "Hex_Grid_Layer_Visualizer"

	def __init__(self):
		self.radius = 3

		self.cmap = get_cmap('bwr')

		self.hex_height = 1.5/np.sqrt(3)
		self.hex_width = 0.5
		self.hex_size = 1/np.sqrt(3)

	def update_parameters(self):
		self.params = {
			"param_path": Hex_Grid_Layer_Visualizer.param_path,
			"param_name": Hex_Grid_Layer_Visualizer.param_name,
		
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

	# layer_state is [time_horizon, Grid.n_cells]
	def visualize(self, layer_state):
		self.init_plot()

		assert layer_state.shape[1] == self.n_cells, \
			"expected layer_state of shape [time_horizon, {}]".format(self.n_cells)

		min_val = np.min(layer_state)
		max_val = np.max(layer_state)

		layer_state -= min_val
		layer_state /= (max_val - min_val)
		layer_state *= 255
		layer_state = layer_state.astype(np.int32)

		for t in range(layer_state.shape[0]):
			for i in range(self.n_cells):
				y, x = self.coords[i]

					ax.patches[i].set_color(self.cmap(layer_state[t, i]))

			ax.set_title('t={}'.format(t))
			if t == 0:
				ax.axis('equal')
			
			fig.canvas.draw()
			plt.pause(0.2)

		plt.close()

	def init_plot(self, layer_state):
		if not hasattr(self, "coords"):
			self.get_coords()

		plt.ion()

		self.fig = plt.figure()
		self.ax = plt.axes()
		self.ax.get_xaxis().set_visible(False)
		self.ax.get_yaxis().set_visible(False)

		# hack to get colorbar
		im = self.ax.scatter([0,0], [0,0], c=[min_val,max_val], cmap=self.cmap)
		self.fig.colorbar(im, ax=self.ax)

		for i in range(self.n_cells):
			y, x = self.coords[i]
			self.ax.add_patch(
				RegularPolygon(
					(x * self.hex_width, -y * self.hex_height), 6, self.hex_size,
					color=self.cmap(127)))

if __name__ == '__main__':
	visualizer = Hex_Grid_Layer_Visualizer()
	visualizer.radius = 2
	visualizer.visualize(np.linspace(-19, 19, 9*19).reshape([9, 19]).astype(np.float32))

	visualizer = Rect_Grid_Layer_Visualizer()
	visualizer.height = 4
	visualizer.width = 4
	visualizer.visualize(np.linspace(-19, 19, 16*9).reshape([9, 16]).astype(np.float32))
