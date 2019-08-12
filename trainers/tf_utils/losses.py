from parameter import Parameter

import tensorflow as tf
import numpy as np

class TF_Loss(Parameter):
    param_path = "trainers.tf_utils.losses"
    param_name = "TF_Loss"

    def __init__(self):
        self.global_params = set([])

        self.params = set([])

        self.shared_params = set([])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        pass

    def get_loss(self, data_batch, model_pred):
        raise NotImplementedError

class TF_MSE_Loss(TF_Loss):
    param_path = "trainers.tf_utils.losses"
    param_name = "TF_MSE_Loss"

    def get_loss(self, data_batch, model_pred):
        loss = tf.reduce_mean(tf.square(data_batch["label"] - model_pred))
        return loss

class TF_Sparse_CEnt_Loss(TF_Loss):
    param_path = "trainers.tf_utils.losses"
    param_name = "TF_Sparse_CEnt_Loss"

    def get_loss(self, data_batch, model_pred):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.squeeze(data_batch["label"]),
            logits=model_pred,
        )
        return loss

class TF_Grid_MSE_Loss(TF_Loss):
    param_path = "trainers.tf_utils.losses"
    param_name = "TF_Grid_MSE_Loss"

    def __init__(self):
        self.global_params = [
        ]

        self.params = [
            "discount",
        ]

        self.shared_params = [
            "train_time_horizon",
        ]

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.weight_tensor = tf.constant(self.discount ** np.arange(self.train_time_horizon), dtype=tf.float32)
        self.weight_tensor /= tf.reduce_sum(self.weight_tensor)
        self.weight_tensor = tf.reshape(self.weight_tensor, [1, self.train_time_horizon, 1, 1])

    def get_loss(self, data_batch, model_pred):
        squared_errors = tf.square(data_batch["pred_gt"] - model_pred)
        loss = tf.reduce_mean(
            tf.reduce_sum(self.weight_tensor * squared_errors, 1)
        )
        return loss

class TF_Grid_L1_Loss(TF_Loss):
    param_path = "trainers.tf_utils.losses"
    param_name = "TF_Grid_L1_Loss"

    def __init__(self):
        self.global_params = [
        ]

        self.params = [
            "discount",
        ]

        self.shared_params = [
            "train_time_horizon",
        ]

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.weight_tensor = tf.constant(self.discount ** np.arange(self.train_time_horizon), dtype=tf.float32)
        self.weight_tensor /= tf.reduce_sum(self.weight_tensor)
        self.weight_tensor = tf.reshape(self.weight_tensor, [1, self.train_time_horizon, 1, 1])

    def get_loss(self, data_batch, model_pred):
        l1_errors = tf.abs(data_batch["pred_gt"] - model_pred)
        loss = tf.reduce_mean(
            tf.reduce_sum(self.weight_tensor * l1_errors, 1)
        )
        return loss
