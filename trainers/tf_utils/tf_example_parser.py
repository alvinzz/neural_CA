from parameter import Parameter

class TF_Example_Parser(Parameter):
    param_path = "trainers.tf_utils.tf_example_parser"
    param_name = "TF_Example_Parser"

    def update_parameters(self):
        self.params = {
            "param_path": TF_Example_Parser.param_path,
            "param_name": TF_Example_Parser.param_name,
        }

    def parse_example(self, example):
        raise NotImplementedError
