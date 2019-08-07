import tensorflow as tf

from parameter import Parameter

class TF_Grid_Model_v1(tf.keras.Model, Parameter):
    param_path = "neural_CA/tf_grid_model"
    param_name = "TF_Grid_Model_v1"

    def __init__(self, grid):
        super(TF_Grid_Model_v1, self).__init__()

        # parameters
        self.effect_dotp_dim = grid.state_dim
        self.effect_dotp_MLP_hidden_sizes = [20, 20]

        self.effect_dim = grid.state_dim
        self.effect_MLP_hidden_sizes = [20, 20]

        self.apply_dotp_dim = grid.state_dim
        self.apply_dotp_MLP_hidden_sizes = [20, 20]

        self.apply_MLP_hidden_sizes = [20, 20]

        self.activation = "relu"

        # store grid & create Layers
        self.grid = grid
        self.build_MLPs()

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
                self.grid.state_dim, self.effect_dotp_dim,
                self.effect_dotp_MLP_hidden_sizes, self.activation)
            self.effect_dotp_neighbor_MLP = self.create_MLP(
                self.grid.state_dim, self.effect_dotp_dim,
                self.effect_dotp_MLP_hidden_sizes, self.activation)

        self.effect_MLP = self.create_MLP(
            2 * self.grid.state_dim + self.effect_dotp_dim, self.effect_dim,
            self.effect_MLP_hidden_sizes, self.activation)

        if self.apply_dotp_dim:
            self.apply_dotp_cell_MLP = self.create_MLP(
                self.grid.state_dim, self.apply_dotp_dim,
                self.apply_dotp_MLP_hidden_sizes, self.activation)
            self.apply_dotp_effect_MLP = self.create_MLP(
                self.effect_dim, self.apply_dotp_dim,
                self.apply_dotp_MLP_hidden_sizes, self.activation)

        self.apply_MLP = self.create_MLP(
            self.grid.state_dim + self.effect_dim + self.apply_dotp_dim, self.grid.state_dim,
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
        #interaction_matrix = tf.stack(list(
        #    map(
        #        lambda input: tf.nn.embedding_lookup(input, self.grid.interaction_inds),
        #        tf.unstack(inputs)
        #    )
        #))
        interaction_matrix = tf.nn.embedding_lookup(inputs, self.grid.interaction_inds)
        cells = interaction_matrix[0]
        neighbors = interaction_matrix[1]

        if self.effect_dotp_dim:
             effect_dotp_cells = self.effect_dotp_cell_MLP(cells)
             effect_dotp_neighbors = self.effect_dotp_neighbor_MLP(neighbors)
             effect_in = tf.concat(
                 [cells, neighbors, effect_dotp_cells * effect_dotp_neighbors], -1)
        else:
             effect_in = tf.concat(
                 [cells, neighbors], -1)

        effects = self.effect_MLP(effect_in)
        tot_effects = tf.math.segment_sum(effects, self.grid.interaction_inds[0])

        if self.apply_dotp_dim:
            apply_dotp_cells = self.apply_dotp_cell_MLP(inputs)
            apply_dotp_effects = self.apply_dotp_effect_MLP(tot_effects)
            apply_in = tf.concat(
                [inputs, tot_effects, apply_dotp_cells * apply_dotp_effects], -1)
        else:
            apply_in = tf.concat(
                [inputs, tot_effects], -1)

        cells_pred = self.apply_MLP(apply_in)

        return cells_pred
