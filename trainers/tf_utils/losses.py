from parameter import Parameter

import tensorflow as tf

class TF_Loss(Parameter):
    param_path = "trainers.tf_utils.losses"
    param_name = "TF_Loss"

    def update_parameters(self):
        self.params = {
            "param_path": TF_Loss.param_path,
            "param_name": TF_Loss.param_name,
        }

    def get_loss(self, data_batch, model_pred):
        raise NotImplementedError

class TF_MSE_Loss(TF_Loss):
    param_path = "trainers.tf_utils.losses"
    param_name = "TF_MSE_Loss"

    def update_parameters(self):
        self.params = {
            "param_path": TF_MSE_Loss.param_path,
            "param_name": TF_MSE_Loss.param_name,
        }

    def get_loss(self, data_batch, model_pred):
        loss = tf.reduce_mean(tf.square(data_batch["label"] - model_pred))
        return loss

class TF_Sparse_CEnt_Loss(TF_Loss):
    param_path = "trainers.tf_utils.losses"
    param_name = "TF_Sparse_CEnt_Loss"

    def update_parameters(self):
        self.params = {
            "param_path": TF_Sparse_CEnt_Loss.param_path,
            "param_name": TF_Sparse_CEnt_Loss.param_name,
        }

    def get_loss(self, data_batch, model_pred):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.squeeze(data_batch["label"]),
            logits=model_pred,
        )
        return loss
