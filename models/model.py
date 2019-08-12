from parameter import Parameter

class Model(Parameter):
    param_path = "models.model"
    param_name = "Model"

    def __init__(self):
        self.global_params = set([])

        self.params = set([])

        self.shared_params = set([])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def predict(self, inputs):
        raise NotImplementedError