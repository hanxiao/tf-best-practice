from logging import Logger
from typing import Dict


class BaseConfig:
    from lazy import lazy

    def __init__(self, work_dir,
                 data_dir,
                 log_dir,
                 script_dir,
                 parameter_file,
                 parameter_profile,
                 log_format):
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.script_dir = script_dir
        self.parameter_file = parameter_file
        self.parameter_profile = parameter_profile
        self.log_format = log_format

    def get_data_dir(self):
        return self.work_dir + self.data_dir

    def get_log_dir(self):
        return self.work_dir + self.log_dir

    def get_script_dir(self):
        return self.work_dir + self.script_dir

    @lazy
    def log_path(self):
        import os
        from datetime import datetime
        return self.get_log_dir() + os.getenv('APPNAME', 'app') + datetime.now().strftime("%m%d-%H%M") + '.log'

    @lazy
    def model_parameter(self):
        from .baseparameter import BaseParameter
        return BaseParameter.load_config(self.parameter_file)[self.parameter_profile]

    @lazy
    def logger(self) -> Logger:
        from utils.logger import get_logger
        return get_logger(__name__, self.log_path, self.log_format)

    @staticmethod
    def load_config(fn: str) -> Dict[str, 'BaseConfig']:
        from ruamel.yaml import YAML
        yaml = YAML(typ='unsafe')
        yaml.register_class(BaseConfig)

        with open(fn) as fp:
            return yaml.load(fp.read())
