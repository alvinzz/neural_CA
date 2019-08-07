from parameter import Parameter

import tensorflow as tf

class TF_Optimizer(Parameter):
    param_path = "trainers.tf_utils.optimizers"
    param_name = "TF_Optimizer"

    def update_parameters(self):
        self.params = {
            "param_path": TF_Optimizer.param_path,
            "param_name": TF_Optimizer.param_name,
        }

    # assigns a TF.keras.optimizers.Optimizer object to self.tf_optimizer
    def build_tf_optimizer(self):
        raise NotImplementedError

class TF_Adam_Optimizer(TF_Optimizer):
    param_path = "trainers.tf_utils.optimizers"
    param_name = "TF_Adam_Optimizer"

    def __init__(self):
        self.learning_rate = 0.001
        self.epsilon = 1e-7 # for imagenet, 0.1 or 1.0 is better

    def update_parameters(self):
        self.params = {
            "param_path": TF_Adam_Optimizer.param_path,
            "param_name": TF_Adam_Optimizer.param_name,

            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
        }

    def build_tf_optimizer(self):
        self.tf_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
        )
