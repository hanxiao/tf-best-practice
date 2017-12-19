from typing import Dict


class BaseConfig:
    from lazy import lazy

    def __init__(self, work_dir,
                 data_dir,
                 log_dir,
                 script_dir,
                 parameter_yaml,
                 parameter_profile):
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.script_dir = script_dir
        self.parameter_yaml = parameter_yaml
        self.parameter_profile = parameter_profile

    def get_data_dir(self):
        return self.work_dir + self.data_dir

    def get_log_dir(self):
        return self.work_dir + self.log_dir

    def get_script_dir(self):
        return self.work_dir + self.script_dir

    @lazy
    def model_parameter(self):
        from .baseparameter import BaseParameter
        return BaseParameter.load_config(self.parameter_yaml)[self.parameter_profile]

    @staticmethod
    def load_config(fn: str) -> Dict[str, 'BaseConfig']:
        from ruamel.yaml import YAML
        yaml = YAML(typ='unsafe')
        yaml.register_class(BaseConfig)

        with open(fn) as fp:
            return yaml.load(fp.read())  # type:
