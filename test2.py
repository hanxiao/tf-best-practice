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
    def __init__(self, yaml_fn, config_name):
        super().__init__(yaml_fn, config_name)
        self.model_parameter = ModelParameter(self.parameter_file, self.parameter_profile)


if __name__ == "__main__":
    config = AppConfig('settings/app.yaml', 'local')
    print(config.work_dir)
    print(config.model_parameter.num_hidden)
