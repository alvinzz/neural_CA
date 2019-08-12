import json
import glob
import importlib

class Parameter(object):
    param_path = "parameter"
    param_name = "Parameter"

    def __init__(self):
        self.global_params = set([])
        self.params = set([])
        self.shared_params = set([])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def param_build(self):
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
        pass

    def save(self, log_dir):
        params = self.save_dict()
        # save in log_dir
        path = log_dir + "/params"
        json.dump(params, open(path, "w"), sort_keys=False, indent=2)
        # save copy in current dir
        path = "last_params"
        json.dump(params, open(path, "w"), sort_keys=False, indent=2)

    def save_dict(self):
        d = {}

        for (p, v) in self.get_params_dict().items():
            if not isinstance(v, Parameter):
                d[p] = v
            else:
                d[p] = v.save_dict()

        for (p, v) in self.get_global_params_dict().items():
            if not isinstance(v, Parameter):
                d[p] = v
            else:
                d[p] = v.save_dict()

        d["param_path"] = self.param_path
        d["param_name"] = self.param_name

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
        self.param_build()

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

    def set_global_params(self, overwrite=True):
        global_params_dict = self.get_global_params_dict()
        self.set_shared_params(global_params_dict, overwrite)

    def set_shared_params(self, global_params_dict, overwrite=True):
        for (global_p, global_v) in global_params_dict.items():
            if global_p in self.shared_params:
                if overwrite or not getattr(self, global_p):
                    setattr(self, global_p, global_v)

        for (p, v) in self.get_global_params_dict().items():
            if isinstance(v, Parameter):
                v.set_shared_params(global_params_dict, overwrite)

        for (p, v) in self.get_params_dict().items():
            if isinstance(v, Parameter):
                v.set_shared_params(global_params_dict, overwrite)

    def get_params_dict(self):
        return {p: getattr(self, p) for p in self.params}

    def get_global_params_dict(self):
        return {p: getattr(self, p) for p in self.global_params}

    def print_params(self):
        print(json.dumps(self.params, sort_keys=False, indent=2))
