from parameter import Parameter

import tensorflow as tf

class TF_Adam_Optimizer(Parameter):
    param_path = "trainers.tf_utils.optimizers"
    param_name = "TF_Adam_Optimizer"

    def __init__(self):
        self.global_params = set([])

        self.params = set([
            "learning_rate",
            "epsilon",
        ])

        self.shared_params = set([])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.tf_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
        )
