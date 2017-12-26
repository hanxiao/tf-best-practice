from logging import Logger

from utils.parameter import AppConfig, ModelParameter

APP_CONFIG = AppConfig('settings/app.yaml', 'aws')
MODEL_PARAM = APP_CONFIG.model_parameter  # type: ModelParameter
LOGGER = APP_CONFIG.logger  # type: Logger
