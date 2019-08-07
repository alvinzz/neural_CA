from parameter import Parameter

class Model(Parameter):
    param_path = "models.model"
    param_name = "Model"

    def __init__(self):
        pass

    def update_parameters(self):
        self.params = {
            "param_path": Model.param_path,
            "param_name": Model.param_name,
        }

    def predict(self, input):
        print("TODO: define predict method for Model")
        raise NotImplementedError