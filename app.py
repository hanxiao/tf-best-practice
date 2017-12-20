from logging import Logger

from utils.config import AppConfig, ModelParameter

APP_CONFIG = AppConfig.load_config('config/app.yaml')['default']
MODEL_CONFIG = APP_CONFIG.model_parameter  # type: ModelParameter
LOGGER = APP_CONFIG.logger  # type: Logger

if __name__ == "__main__":
    pass
