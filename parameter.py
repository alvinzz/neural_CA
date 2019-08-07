import json
import glob
import importlib

class Parameter(object):
    def __init__(self):
        pass

    def update_parameters(self):
        print("TODO: update self.params dict {param: value}")
        raise NotImplementedError

    def save(self, log_dir):
        params = self.save_dict()
        # save in log_dir
        path = log_dir + "/params"
        json.dump(params, open(path, "w"), sort_keys=False, indent=2)
        # save copy in current dir
        path = "last_params"
        json.dump(params, open(path, "w"), sort_keys=False, indent=2)

    def save_dict(self):
        self.update_parameters()
        d = {}
        for (param, value) in self.params.items():
            if not isinstance(value, Parameter):
                d[param] = value
            else:
                d[param] = value.save_dict()
        return d

    def load(self, param_prefix):
        prefix_match_files = glob.glob(param_prefix + "*")
        if not prefix_match_files:
            print("Could not find file with prefix {}".format(param_prefix))
            raise ValueError
        param_file = sorted(prefix_match_files)[-1]
        print("Loading param file {}...".format(param_file))
        params = json.load(open(param_file, "r"))
        self.load_dict(params)

    def load_dict(self, d):
        for (param, value) in d.items():
            if type(value) != dict:
                setattr(self, param, value)
            else:
                sub_param_path = value["param_path"]
                sub_param_name = value["param_name"]
                sub_param_module = importlib.import_module(sub_param_path)
                sub_param = getattr(sub_param_module, sub_param_name)()
                setattr(self, param, sub_param)
                assert isinstance(sub_param, Parameter), "sub-parameter {} of {} has dict of values but is not Parameter".format(sub_param, self)
                sub_param.load_dict(value)

    def print_params(self):
        print(json.dumps(self.params, sort_keys=False, indent=2))
