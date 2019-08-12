from parameter import Parameter

class Trainer(Parameter):
    param_path = "trainers.trainer"
    param_name = "Trainer"

    def __init__(self):
        self.global_params = set([])

        self.params = set([])

        self.shared_params = set([
            "exp_name",
            "dataset_manager",
        ])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def train(self, model):
        raise NotImplementedError
