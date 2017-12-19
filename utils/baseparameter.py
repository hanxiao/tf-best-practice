from typing import Dict


class BaseParameter:
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
    def load_config(fn: str) -> Dict[str, 'BaseParameter']:
        from ruamel.yaml import YAML
        yaml = YAML(typ='unsafe')
        yaml.register_class(BaseParameter)

        with open(fn) as fp:
            return yaml.load(fp.read())
