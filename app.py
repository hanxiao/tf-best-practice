from logging import Logger

from utils.baseconfig import BaseConfig
from utils.helper import JobContext

CONFIG = BaseConfig.load_config('config/app.yaml')['default']
LOGGER = CONFIG.logger  # type: Logger

print(CONFIG)

with JobContext('testing'):
    a = 1
