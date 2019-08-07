from parameter import Parameter

class Trainer(Parameter):
    param_path = "trainers.trainer"
    param_name = "Trainer"

    def __init__(self):
        self.data_loc = None

    def update_parameters(self):
        self.params = {
            "param_path": Trainer.param_path,
            "param_name": Trainer.param_name,

            "data_loc": self.data_loc,
        }

    def load_data(self):
        if not hasattr(self, data):
            print("TODO: implement load_data method for Trainer")
            raise NotImplementedError

    def train(self, model):
        self.load_data()
        print("TODO: implement train method for Trainer")
        raise NotImplementedError
