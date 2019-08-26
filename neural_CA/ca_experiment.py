from experiment import Experiment

class CA_Experiment(Experiment):
    param_path = "neural_CA.ca_experiment"
    param_name = "CA_Experiment"

    def __init__(self):
        self.global_params = [
            "exp_name",

            "neighbor_rule",

            "obs_dim",

            "random_seed",

            "model",

            "grid_visualizer",

            "dataset_manager",
        ]

        self.params = [
            "trainer",

            "evaluator",
        ]

        self.shared_params = []

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def save(self):
        super().save()
