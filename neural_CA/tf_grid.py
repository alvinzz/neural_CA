import numpy as np
import tensorflow as tf

from models.model import Model

from utils import create_MLP

class TF_Grid(Model):
    param_path = "neural_CA/tf_grid"
    param_name = "TF_Grid"

    def __init__(self):
        super(TF_Grid, self).__init__()

        # grid parameters
        self.n_cells = 9

        self.obs_dim = 1
        self.hidden_dim = 1

        self.neighbor_rule = None

    def update_parameters(self):
        self.params = {
            "param_path": TF_Grid_v1.param_path,
            "param_name": TF_Grid_v1.param_name,

            "n_cells": self.n_cells,

            "obs_dim": self.obs_dim,
            "hidden_dim": self.hidden_dim,

            "neighbor_rule": self.neighbor_rule,
        }

    def init_tf_vars(self):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError

    def predict(self, inputs):
        return self.__call__(inputs)

    # calculates adjacency matrix for the grid based on the Grid.neighbor_rule
    def get_A(self):
        if not hasattr(self, "A"):
            self.A = np.zeros((self.n_cells, self.n_cells), dtype=np.int32)

            for i in range(self.n_cells):
                for j in range(self.n_cells):
                    if self.neighbor_rule.is_neighbor(i, j):
                        self.A[i, j] = 1

            self.effect_inds = np.array(np.nonzero(self.A))
            self.n_effects = self.effect_inds.shape[1]

    def visualize(self, grid_visualizer):
        raise NotImplementedError

    def visualize_n(self, n, grid_layer_visualizer):
        raise NotImplementedError


# cannot inherit tf.keras.Model from TF_Grid due to TF
class TF_Grid_v1(tf.keras.Model, TF_Grid):
    param_path = "neural_CA/tf_grid"
    param_name = "TF_Grid_v1"

    def __init__(self):
        super(TF_Grid_v1, self).__init__()

        # grid parameters
        self.n_cells = 9

        self.obs_dim = 1
        self.hidden_dim = 1

        self.neighbor_rule = None

        # nn parameters
        self.effect_dotp_dim = 1
        self.effect_dotp_MLP_hidden_sizes = [20, 20]

        self.effect_dim = 1
        self.effect_MLP_hidden_sizes = [20, 20]

        self.apply_dotp_dim = 1
        self.apply_dotp_MLP_hidden_sizes = [20, 20]

        self.apply_MLP_hidden_sizes = [20, 20]

        self.activation = "relu"

        # for convenience
        self.state_dim = self.obs_dim + self.hidden_dim

    def update_parameters(self):
        self.params = {
            "param_path": TF_Grid_v1.param_path,
            "param_name": TF_Grid_v1.param_name,

            "n_cells": self.n_cells,

            "obs_dim": self.obs_dim,
            "hidden_dim": self.hidden_dim,

            "neighbor_rule": self.neighbor_rule,

            "effect_dotp_dim": self.effect_dotp_dim,
            "effect_dotp_MLP_hidden_sizes": self.effect_dotp_MLP_hidden_sizes,

            "effect_dim": self.effect_dim,
            "effect_MLP_hidden_sizes": self.effect_MLP_hidden_sizes,

            "apply_dotp_dim": self.apply_dotp_dim,
            "apply_dotp_MLP_hidden_sizes": self.apply_dotp_MLP_hidden_sizes,

            "apply_MLP_hidden_sizes": self.apply_MLP_hidden_sizes,

            "activation": self.activation,
        }

    def init_tf_vars(self):
        # init Variables for tf.keras.Model
        self.get_A()
        self.build_MLPs()
        if self.hidden_dim:
            self.init_hidden_state = tf.Variable(tf.ones((self.n_cells, self.hidden_dim)), dtype=tf.float32, trainable=True, name="init_hidden")

    def call(self, inputs):
        """
        pseudocode:
        for cell in cells:
            # calculate total effect of neighbors on cell
            tot_effect = 0
            for neighbor in get_neighbors(cell):
                cell_effect_transform = MLP(cell.state)
                neighbor_effect_transform = MLP(neighbor.state)
                effect_dotp = cell_effect_transform (*) neighbor_effect_transform # (*) denotes elementwise multiplication
                effect = MLP(cell.state, neighbor.state, effect_dotp)
                tot_effect += effect

            # apply total effect to cell
            apply_cell_transform = MLP(cell.state)
            apply_effect_transform = MLP(tot_effect)
            apply_dotp = apply_cell_transform (*) apply_effect_transform
            cell.state = MLP(cell.state, tot_effect, apply_dotp)
        """
        time_horizon, grid_obs = inputs["time_horizon"], inputs["grid_obs"]
        batch_size = tf.shape(grid_obs)[0]

        preds = tf.TensorArray(tf.float32, time_horizon, dynamic_size=False, tensor_array_name="preds", infer_shape=True)

        if self.hidden_dim:
            batch_cells = tf.concat([grid_obs, tf.tile(tf.expand_dims(self.init_hidden_state, 0), [batch_size, 1, 1])], -1)
        else:
            batch_cells = grid_obs
        cells = tf.reshape(batch_cells, [batch_size * self.n_cells, self.state_dim])

        for t in tf.range(time_horizon):
            effect_matrix = tf.map_fn(lambda cells: tf.nn.embedding_lookup(cells, self.effect_inds), batch_cells)
            effect_cells = tf.reshape(effect_matrix[:, 0], [batch_size * self.n_effects, self.state_dim])
            effect_neighbors = tf.reshape(effect_matrix[:, 1], [batch_size * self.n_effects, self.state_dim])

            if self.effect_dotp_dim:
                 cell_effect_transform = self.cell_effect_transform_MLP(effect_cells)
                 neighbor_effect_transform = self.neighbor_effect_transform_MLP(effect_neighbors)
                 effect_in = tf.concat(
                     [effect_cells, effect_neighbors, cell_effect_transform * neighbor_effect_transform], -1)
            else:
                 effect_in = tf.concat(
                     [effect_cells, effect_neighbors], -1)

            effects = self.effect_MLP(effect_in)

            batch_effects = tf.reshape(effects, [batch_size, self.n_effects, self.effect_dim])
            tot_effect = tf.map_fn(lambda effects: tf.math.segment_sum(effects, self.effect_inds[0]), batch_effects)
            tot_effect = tf.reshape(tot_effect, [batch_size * self.n_cells, self.effect_dim])

            if self.apply_dotp_dim:
                cell_apply_transform = self.cell_apply_transform_MLP(cells)
                effect_apply_transform = self.effect_apply_transform_MLP(tot_effect)
                apply_in = tf.concat(
                    [cells, tot_effect, cell_apply_transform * effect_apply_transform], -1)
            else:
                apply_in = tf.concat(
                    [cells, tot_effect], -1)

            cells = self.apply_MLP(apply_in)
            batch_cells = tf.reshape(cells, [batch_size, self.n_cells, self.state_dim])

            preds = preds.write(t, batch_cells[:, :, :self.obs_dim])

        preds = preds.stack()
        preds = tf.transpose(preds, [1, 0, 2, 3])

        return preds

    # cannot be inherited due to TF
    def predict(self, inputs):
        return self.__call__(inputs)

    def visualize(self, grid_visualizer):
        raise NotImplementedError

    def visualize_n(self, n, grid_layer_visualizer):
        raise NotImplementedError

    def build_MLPs(self):
        if self.effect_dotp_dim:
            self.cell_effect_transform_MLP = create_MLP(
                self.state_dim, self.effect_dotp_dim,
                self.effect_dotp_MLP_hidden_sizes, self.activation)
            self.neighbor_effect_transform_MLP = create_MLP(
                self.state_dim, self.effect_dotp_dim,
                self.effect_dotp_MLP_hidden_sizes, self.activation)

        self.effect_MLP = create_MLP(
            2 * self.state_dim + self.effect_dotp_dim, self.effect_dim,
            self.effect_MLP_hidden_sizes, self.activation)

        if self.apply_dotp_dim:
            self.cell_apply_transform_MLP = create_MLP(
                self.state_dim, self.apply_dotp_dim,
                self.apply_dotp_MLP_hidden_sizes, self.activation)
            self.effect_apply_transform_MLP = create_MLP(
                self.effect_dim, self.apply_dotp_dim,
                self.apply_dotp_MLP_hidden_sizes, self.activation)

        self.apply_MLP = create_MLP(
            self.state_dim + self.effect_dim + self.apply_dotp_dim, self.state_dim,
            self.apply_MLP_hidden_sizes, self.activation)


if __name__ == '__main__':
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

    grid = TF_Grid_v1()
    grid.neighbor_rule = neighbor_rule
    grid.init_tf_vars()
    optimizer = tf.keras.optimizers.Adam(0.01)
    
    @tf.function
    def f(): 
        inputs = {"grid_obs": tf.constant([
            [[0],[1],[2],[3],[4],[5],[6],[7],[8]],
            [[10],[11],[12],[13],[14],[15],[16],[17],[18]],
            [[20],[21],[22],[23],[24],[25],[26],[27],[28]]], dtype=tf.float32), 
            "batch_size": 3,
            "time_horizon": 5,
        }
        # inputs = {"grid_obs": tf.constant([
        #     [[0],[1],[2],[3],[4],[5],[6],[7],[8]],
        #     [[0],[1],[2],[3],[4],[5],[6],[7],[8]],
        #     [[0],[1],[2],[3],[4],[5],[6],[7],[8]]], dtype=tf.float32)}

        with tf.GradientTape() as grad_tape:
            pred = grid.predict(inputs)
            loss = tf.reduce_mean(tf.square(pred - 1))
        gradients = grad_tape.gradient(loss, grid.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients, grid.trainable_variables))

        return (loss, pred)

    print(f()[1][2])
    losses = []
    for i in range(100):
        losses.append(f()[0])
    print(losses[::10])
    print(f()[1][2])

    print(grid.init_hidden_state)
