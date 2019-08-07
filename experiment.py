import os
import datetime

from parameter import Parameter

class Experiment(Parameter):
    param_path = "experiment"
    param_name = "Experiment"

    def __init__(self):
        self.model = None

        self.trainer = None
        self.evaluator = None

    def update_parameters(self):
        self.params = {
            "param_path": Experiment.param_path,
            "param_name": Experiment.param_name,

            "model": self.model,

            "trainer": self.trainer,
            "evaluator": self.evaluator,
        }

    def set_exp_name(self, exp_name):
        currentDT = datetime.datetime.now()
        self.exp_name = exp_name + "_" + currentDT.strftime("%Y_%m_%d_%H_%M_%S")
        os.mkdir(self.exp_name)

        self.trainer.exp_name = self.exp_name
        self.evaluator.exp_name = self.exp_name

    def save(self):
        super().save(self.exp_name)

    def train(self):
        self.trainer.train(self.model)

    def val(self):
        self.evaluator.val(self.model)

    def test(self):
        self.evaluator.test(self.model)
