import os
from datetime import datetime

from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams

from utils.logger import get_logger


class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__()
        with open(yaml_fn) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.add_hparam(k, v)


class ModelParams(YParams):
    pass


class AppConfig(YParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__(yaml_fn, config_name)
        self.data_dir = self.work_dir + self.data_dir
        self.log_dir = self.work_dir + self.log_dir
        self.script_dir = self.work_dir + self.script_dir
        self.log_path = self.log_dir + os.getenv('APPNAME', 'app') + datetime.now().strftime("%m%d-%H%M") + '.log'
        self.logger = get_logger(__name__, self.log_path, self.log_format)
