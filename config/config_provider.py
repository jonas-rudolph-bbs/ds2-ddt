# SPDX-FileCopyrightText: 2025 - 2025 Software GmbH, Darmstadt, Germany and/or its subsidiaries and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from .config_manager import ConfigManager
from .validation_config_schema import normalize_validation_config


class ConfigProvider:
    def __init__(self):
        self.reload()

    def reload(self):
        config_manager = ConfigManager()

        mqtt_config = config_manager.load("generated_mqtt_config.json")
        validation_configs = config_manager.load("validations")

        normalized_validation_configs = {
            config_id: normalize_validation_config(config_id, config)
            for config_id, config in validation_configs.items()
        }

        self._cache = {
            "mqtt": mqtt_config,
            "validation": normalized_validation_configs,
        }

    def mqtt(self):
        return self._cache["mqtt"]

    def validation(self):
        return self._cache["validation"]
