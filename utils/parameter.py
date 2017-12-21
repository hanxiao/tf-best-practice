from lazy import lazy
from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams


class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__()
        with open(yaml_fn) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.add_hparam(k, v)


class ModelParameter(YParams):
    pass


class AppConfig(YParams):
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
        return ModelParameter(self.parameter_file, self.parameter_profile)

    @lazy
    def logger(self):
        from utils.logger import get_logger
        return get_logger(__name__, self.log_path, self.log_format)
