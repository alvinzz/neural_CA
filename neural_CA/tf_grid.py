import numpy as np
import tensorflow as tf

from models.model import Model

class TF_Grid(Model):
    param_path = "neural_CA/tf_grid"
    param_name = "TF_Grid"

    def __init__(self):
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

        # build tf_model
        self.build_tf_model()


    def update_parameters(self):
        self.params = {
            "param_path": TF_Grid.param_path,
            "param_name": TF_Grid.param_name,

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
        self.get_A()

        if not hasattr(self, 'tf_model'):
            grid_obs = tf.keras.layers.Input(shape=[self.n_cells, self.obs_dim])
            print(grid_obs)
            time_horizon = tf.keras.layers.Input(shape=(), dtype=tf.int32)

            self.build_MLPs()
            
            if self.hidden_dim:
                self.start_hidden_state = tf.Variable(tf.zeros((self.n_cells, self.hidden_dim)), dtype=tf.float32)
            
            preds = []

            batch_size = tf.shape(grid_obs)[0]

            if self.hidden_dim:
                batch_cells = tf.concat([grid_obs, tf.tile(tf.expand_dims(self.start_hidden_state, 0), [batch_size, 1, 1])], -1)
            else:
                batch_cells = grid_obs
            cells = tf.reshape(batch_cells, [batch_size * self.n_cells, self.state_dim])

            for t in range(self.time_horizon):
                effect_matrix = tf.stack(list(
                    map(lambda cells: tf.nn.embedding_lookup(cells, self.effect_inds), tf.unstack(batch_cells))))
                effect_cells = tf.reshape(effect_matrix[:, 0], [batch_size * self.n_effects, self.state_dim])
                effect_neighbors = tf.reshape(effect_matrix[:, 1], [batch_size * self.n_effects, self.state_dim])

                if self.effect_dotp_dim:
                     effect_dotp_cells = self.effect_dotp_cell_MLP(effect_cells)
                     effect_dotp_neighbors = self.effect_dotp_neighbor_MLP(effect_neighbors)
                     effect_in = tf.concat(
                         [effect_cells, effect_neighbors, effect_dotp_cells * effect_dotp_neighbors], -1)
                else:
                     effect_in = tf.concat(
                         [effect_cells, effect_neighbors], -1)

                effects = self.effect_MLP(effect_in)
                batch_effects = tf.reshape(effects, [batch_size, self.n_effects, self.effect_dim])
                stack_tot_effects = tf.stack(list(
                    map(lambda effects: tf.math.segment_sum(effects, self.effect_inds[0]), tf.unstack(batch_effects))))
                tot_effects = tf.reshape(stack_tot_effects, [batch_size * self.n_cells, self.effect_dim])

                if self.apply_dotp_dim:
                    apply_dotp_cells = self.apply_dotp_cell_MLP(cells)
                    apply_dotp_effects = self.apply_dotp_effect_MLP(tot_effects)
                    apply_in = tf.concat(
                        [cells, tot_effects, apply_dotp_cells * apply_dotp_effects], -1)
                else:
                    apply_in = tf.concat(
                        [cells, tot_effects], -1)

                cells = self.apply_MLP(apply_in)

                batch_cells = tf.reshape(cells, [batch_size, self.n_cells, self.state_dim])

                preds.append(batch_cells[:, :, :self.obs_dim])

            preds = tf.stack(preds, 1)

            self.tf_model = tf.keras.Model(inputs=[grid_obs, time_horizon], outputs=preds)

    def build_MLPs(self):
        if self.effect_dotp_dim:
            self.effect_dotp_cell_MLP = self.create_MLP(
                self.state_dim, self.effect_dotp_dim,
                self.effect_dotp_MLP_hidden_sizes, self.activation)
            self.effect_dotp_neighbor_MLP = self.create_MLP(
                self.state_dim, self.effect_dotp_dim,
                self.effect_dotp_MLP_hidden_sizes, self.activation)

        self.effect_MLP = self.create_MLP(
            2 * self.state_dim + self.effect_dotp_dim, self.effect_dim,
            self.effect_MLP_hidden_sizes, self.activation)

        if self.apply_dotp_dim:
            self.apply_dotp_cell_MLP = self.create_MLP(
                self.state_dim, self.apply_dotp_dim,
                self.apply_dotp_MLP_hidden_sizes, self.activation)
            self.apply_dotp_effect_MLP = self.create_MLP(
                self.effect_dim, self.apply_dotp_dim,
                self.apply_dotp_MLP_hidden_sizes, self.activation)

        self.apply_MLP = self.create_MLP(
            self.state_dim + self.effect_dim + self.apply_dotp_dim, self.state_dim,
            self.apply_MLP_hidden_sizes, self.activation)

    def create_MLP(self, in_size, out_size, hidden_sizes, activation):
        mlp = []

        for layer in range(len(hidden_sizes) + 1):
            if layer == 0:
                mlp.append(tf.keras.layers.Dense(hidden_sizes[0],
                    activation=activation, input_shape=[in_size]))
            elif layer == len(hidden_sizes):
                mlp.append(tf.keras.layers.Dense(out_size))
            else:
                mlp.append(tf.keras.layers.Dense(hidden_sizes[layer],
                    activation=activation))

        mlp = tf.keras.Sequential(mlp)

        return mlp

    def MLP_forward(self, mlp, x):
        for layer in mlp:
            x = layer(x)
        return x

    def predict(self, inputs):
        if not hasattr(self, 'A'):
            self.get_A()

        if not hasattr(self, 'tf_model'):
            self.build_tf_model()

        pred = self.tf_model(inputs["grid_obs"], inputs["time_horizon"])

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
    # grid.tf_grid_model = TF_Grid_Model_v1
    # grid.hidden_dim = 1
    # grid.time_horizon = 10

    inputs = {"grid_obs": tf.constant([
        [[0],[1],[2],[3],[4],[5],[6],[7],[8]],
        [[10],[11],[12],[13],[14],[15],[16],[17],[18]],
        [[20],[21],[22],[23],[24],[25],[26],[27],[28]]], dtype=tf.float32),
        "time_horizon": 2}

    #with tf.GradientTape() as grad_tape:
    #    pred = grid.predict(inputs)
    #    loss = tf.linalg.norm(pred)
    #gradients = grad_tape.gradient(loss, grid.tf_grid_model.trainable_variables)

    #print(list(grid.tf_grid_model.trainable_variables))

    pred = grid.predict(inputs)
    print(pred[0])
    print(pred[1])
    print(pred[2])
