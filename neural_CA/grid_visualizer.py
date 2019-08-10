from parameter import Parameter

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import RegularPolygon

import numpy as np

class Grid_Visualizer(Parameter):
	param_path = "neural_CA/grid_visualizer"
	param_name = "Grid_Visualizer"

	def __init__(self):


	def update_parameters(self):
		self.params = {
			"param_path": Grid_Visualizer.param_path,
			"param_name": Grid_Visualizer.param_name,

			"layer_visualizer": self.layer_visualizer,
		}

	def visualize(self, state):
		raise NotImplementedError