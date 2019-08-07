import tensorflow as tf

from parameter import Parameter

class TF_Grid_Model_v1(tf.keras.Model, Parameter):
    param_path = "neural_CA/tf_grid_model"
    param_name = "TF_Grid_Model_v1"

    def __init__(self, grid):
        super(TF_Grid_Model_v1, self).__init__()

        # parameters
        self.effect_dotp_dim = 1
        self.effect_dotp_MLP_hidden_sizes = [20, 20]

        self.effect_dim = 1
        self.effect_MLP_hidden_sizes = [20, 20]

        self.apply_dotp_dim = 1
        self.apply_dotp_MLP_hidden_sizes = [20, 20]

        self.apply_MLP_hidden_sizes = [20, 20]

        self.activation = "relu"

        # store grid params
        self.grid = grid

        self.n_cells = self.grid.n_cells
        self.effect_inds = self.grid.effect_inds

        self.obs_dim = self.grid.obs_dim
        self.hidden_dim = self.grid.hidden_dim
        self.state_dim = self.obs_dim + self.hidden_dim

        self.time_horizon = self.grid.time_horizon

        # build Layers and start_hidden_state Variable
        self.build_MLPs()
        if self.hidden_dim:
            self.start_hidden_state = tf.Variable((self.n_cells, self.hidden_dim), dtype=tf.float32)        

    def update_paramters(self):
        self.params = {
            "param_path": TF_Grid_Model_v1.param_path,
            "param_name": TF_Grid_Model_v1.param_name,

            "effect_dotp_dim": self.effect_dotp_dim,
            "effect_dotp_MLP_hidden_sizes": self.effect_dotp_MLP_hidden_sizes,

            "effect_dim": self.effect_dim,
            "effect_MLP_hidden_sizes": self.effect_MLP_hidden_sizes,

            "apply_dotp_dim": self.apply_dotp_dim,
            "apply_dotp_MLP_hidden_sizes": self.apply_dotp_MLP_hidden_sizes,

            "apply_MLP_hidden_sizes": self.apply_MLP_hidden_sizes,

            "activation": self.activation,
        }

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

    def call(self, inputs):
        preds = []

        if self.hidden_dim:
            cells = tf.concat([inputs["grid_obs"], tf.tile(self.start_hidden_state)], -1)
        else:
            cells = inputs["grid_obs"]
       
        for t in range(self.time_horizon):
            effect_matrix = tf.nn.embedding_lookup(cells, self.effect_inds)
            effect_cells = effect_matrix[0]
            effect_neighbors = effect_matrix[1]

            if self.effect_dotp_dim:
                 effect_dotp_cells = self.effect_dotp_cell_MLP(effect_cells)
                 effect_dotp_neighbors = self.effect_dotp_neighbor_MLP(effect_neighbors)
                 effect_in = tf.concat(
                     [effect_cells, effect_neighbors, effect_dotp_cells * effect_dotp_neighbors], -1)
            else:
                 effect_in = tf.concat(
                     [effect_cells, effect_neighbors], -1)

            effects = self.effect_MLP(effect_in)
            tot_effects = tf.math.segment_sum(effects, self.effect_inds[0])

            if self.apply_dotp_dim:
                apply_dotp_cells = self.apply_dotp_cell_MLP(cells)
                apply_dotp_effects = self.apply_dotp_effect_MLP(tot_effects)
                apply_in = tf.concat(
                    [cells, tot_effects, apply_dotp_cells * apply_dotp_effects], -1)
            else:
                apply_in = tf.concat(
                    [cells, tot_effects], -1)

            cells_pred = self.apply_MLP(apply_in)

            preds.append(cells_pred)

        preds = tf.stack(preds)

        return preds
