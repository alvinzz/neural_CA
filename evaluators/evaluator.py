from parameter import Parameter

class Evaluator(Parameter):
    param_path = "evaluators.evaluator"
    param_name = "Evaluator"

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

    def val(self, model):
        raise NotImplementedError

    def test(self, model):
        raise NotImplementedError
