from logging import Logger

from utils.baseconfig import BaseConfig

CONFIG = BaseConfig.load_config('config/app.yaml')['default']
LOGGER = CONFIG.logger  # type: Logger

LOGGER.info('start')
print(CONFIG)
LOGGER.info('end')
