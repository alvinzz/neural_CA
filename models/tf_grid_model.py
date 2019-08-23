import numpy as np
import tensorflow as tf

from models.model import Model

from models.tf_utils import create_MLP

class TF_Grid_Model(Model):
    param_path = "models.tf_grid_model"
    param_name = "TF_Grid_Model"

    def __init__(self):
        self.global_params = []

        self.params = [
            "model_hidden_dim",
            "warmup_time",
        ]

        self.shared_params = [
            "neighbor_rule",
        ]

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.A = self.neighbor_rule.A
        self.get_effect_inds()
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError
        # return self.__call__(inputs)

    # calculates effect_inds based on the Grid.neighbor_rule
    def get_effect_inds(self):
        self.effect_inds = np.array(np.nonzero(self.A))
        self.n_effects = self.effect_inds.shape[1]

    def get_trainable_variables(self):
        return self.trainable_variables

# cannot inherit tf.keras.Model from TF_Grid due to TF
class TF_Grid_Model_v1(tf.keras.Model, TF_Grid_Model):
    param_path = "models.tf_grid_model"
    param_name = "TF_Grid_Model_v1"

    def __init__(self):
        super(TF_Grid_Model_v1, self).__init__()

        self.global_params = []

        self.params = [
            "model_hidden_dim",
            "warmup_time",
            "effect_dotp_dim",
            "effect_dotp_MLP_hidden_sizes",
            "effect_dim",
            "effect_MLP_hidden_sizes",
            "apply_dotp_dim",
            "apply_dotp_MLP_hidden_sizes",
            "apply_MLP_hidden_sizes",
            "activation",
        ]

        self.shared_params = [
            "obs_dim",
            "neighbor_rule",
        ]

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.A = self.neighbor_rule.A
        self.state_dim = self.obs_dim + self.model_hidden_dim
        self.n_cells = self.A.shape[0]
        
        self.get_effect_inds()

        self.build_MLPs()
        self.init_hidden_state = tf.Variable(
            tf.zeros([self.n_cells, self.model_hidden_dim]),
            dtype=tf.float32, trainable=True, name="init_hidden")

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
        warmup_obs = inputs["warmup_obs"] # [batch_size, warmup_time, n_cells, obs_dim]
        # TODO: add valid mask
        pred_time_horizon = inputs["pred_time_horizon"]
        batch_size = tf.shape(warmup_obs)[0]

        # during warmup period, overwrite preds with gt (masked for valid) and update hidden state
        # handle first timestep & init_hidden_state
        current_obs = warmup_obs[:, 0, :, :]
        current_hidden_state = tf.tile(
            tf.expand_dims(self.init_hidden_state, 0),
            [batch_size, 1, 1])
        batch_cells = tf.concat([current_obs, current_hidden_state], -1)
        batch_cells = self.forward_step(batch_cells)
        # handle remaining warmup timesteps
        for t in tf.range(1, self.warmup_time):
            current_obs = warmup_obs[:, t, :, :]
            current_hidden_state = batch_cells[:, :, self.obs_dim:]
            batch_cells = tf.concat([current_obs, current_hidden_state], -1)
            batch_cells = self.forward_step(batch_cells)

        # compute and store predictions
        preds = tf.TensorArray(tf.float32, pred_time_horizon, 
            dynamic_size=False, tensor_array_name="preds", infer_shape=True)
        for t in tf.range(pred_time_horizon):
            batch_cells = self.forward_step(batch_cells)
            preds = preds.write(t, batch_cells[:, :, :self.obs_dim])

        preds = preds.stack()
        preds = tf.transpose(preds, [1, 0, 2, 3]) # [batch_size, pred_time_horizon, n_cells, obs_dim]

        return preds

    # cannot be inherited due to TF
    def predict(self, inputs):
        return self.__call__(inputs)

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

    def forward_step(self, batch_cells):
        batch_size = tf.shape(batch_cells)[0]
        cells = tf.reshape(batch_cells, [batch_size * self.n_cells, self.state_dim])

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

        return batch_cells