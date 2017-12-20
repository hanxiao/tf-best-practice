import logging

from utils.parameter import AppConfig, ModelParameter

APP_CONFIG = AppConfig.load_config('settings/app.yaml')['default']
MODEL_CONFIG = APP_CONFIG.model_parameter  # type: ModelParameter
LOGGER = APP_CONFIG.logger  # type: logging.Logger
