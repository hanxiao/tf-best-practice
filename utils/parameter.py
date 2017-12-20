from typing import Dict


class ModelParameter:
    def __init__(self,
                 batch_size,
                 split_ratio,
                 train_embedding,
                 cell,
                 init_state_type,
                 dilation,
                 num_hidden,
                 metric,
                 num_epoch,
                 optimizer,
                 learning_rate,
                 decay_rate,
                 test_interval,
                 loss,
                 input_data):
        self.input_data = input_data
        self.loss = loss
        self.test_interval = test_interval
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.metric = metric
        self.num_hidden = num_hidden
        self.dilation = dilation
        self.init_state_type = init_state_type
        self.cell = cell
        self.train_embedding = train_embedding
        self.split_ratio = split_ratio
        self.batch_size = batch_size

    @staticmethod
    def load_config(fn: str) -> Dict[str, 'ModelParameter']:
        from ruamel.yaml import YAML
        yaml = YAML(typ='unsafe')
        yaml.register_class(ModelParameter)

        with open(fn) as fp:
            return yaml.load(fp.read())


class AppConfig:
    from lazy import lazy

    def __init__(self, work_dir,
                 data_dir,
                 log_dir,
                 script_dir,
                 parameter_file,
                 parameter_profile,
                 log_format,
                 data_file):
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.script_dir = script_dir
        self.parameter_file = parameter_file
        self.parameter_profile = parameter_profile
        self.log_format = log_format
        self.data_file = data_file

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
        return ModelParameter.load_config(self.parameter_file)[self.parameter_profile]

    @lazy
    def logger(self):
        from utils.logger import get_logger
        return get_logger(__name__, self.log_path, self.log_format)

    @staticmethod
    def load_config(fn: str) -> Dict[str, 'AppConfig']:
        from ruamel.yaml import YAML
        yaml = YAML(typ='unsafe')
        yaml.register_class(AppConfig)

        with open(fn) as fp:
            return yaml.load(fp.read())
