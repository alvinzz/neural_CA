import datetime
import os
import re

from parameter import Parameter

class Experiment(Parameter):
    param_path = "experiment"
    param_name = "Experiment"

    def __init__(self):
        self.global_params = [
            "exp_name",

            "model",
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

    def param_build(self):
        if re.match("_[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}", self.exp_name[-20:]):
           self.exp_name = self.exp_name[:-20]
        currentDT = datetime.datetime.now()
        self.exp_name = self.exp_name + "_" + currentDT.strftime("%Y_%m_%d_%H_%M_%S")

        self.set_global_params(overwrite=True)

        for p in self.global_params:
            v = getattr(self, p)
            if isinstance(v, Parameter):
                v.param_build()

        for p in self.params:
            v = getattr(self, p)
            if isinstance(v, Parameter):
                v.param_build()

        self._build()

    def _build(self):
        try:
            os.mkdir(self.exp_name)
        except:
            pass

    def save(self):
        super().save(self.exp_name)
