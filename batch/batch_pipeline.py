# SPDX-FileCopyrightText: 2025 - 2025 Software GmbH, Darmstadt, Germany and/or its subsidiaries and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# batch_pipeline.py
from typing import Callable

from data_correction import DataCorrection, CorrectionEngine
from .data_queue import DataQueue
from .batch_validator import BatchValidator
from mqtt import AlarmPublisher, ResultPublisher
from config import ConfigProvider
import pandas as pd

class BatchPipeline:
    """Glue DataQueue → BatchValidator → ResultHandler."""

    _default_publish: Callable[[str, dict], None] | None = None   # class‑level

    @classmethod
    def set_default_publisher(cls, fn: Callable[[str, dict], None]) -> None:
        """Register a process‑wide publish‑callable (e.g. MqttPublisher.publish)."""
        cls._default_publish = fn


    def __init__(self, topic:str,  config_name: str, batch_size: int):
        self.validator = BatchValidator(config_name)
        self.queue = DataQueue(batch_size, self._process)
        self.correction_engine = CorrectionEngine(topic, config_name, DataCorrection())
        publish = BatchPipeline._default_publish
        cfg_provider = ConfigProvider()
        self._alarms  = None
        self._results = None
        if publish:
            mqtt_topic_cfg = cfg_provider.mqtt()['topics'].get(topic)
            if mqtt_topic_cfg:
                self._alarms  = AlarmPublisher(mqtt_topic_cfg, publish)
                self._results = ResultPublisher(mqtt_topic_cfg, publish)


    

 
    def add(self, row: dict) -> None:
        self.queue.add(row)
        print(f"Added row to queue for topic '{self.validator.config_name}': {row}")

    # internal callback
    def _process(self, df: pd.DataFrame) -> None:
        '""Process a DataFrame asynchronously. After a DataQueue batch is ready."""'
        validation_results = self.validator(df)
        cleaned_df, alarm_events = self.correction_engine.run(validation_results, df)

        # --- alarms first ------------------------------------------------- #
        if self._alarms:
            for alarm in alarm_events:
                self._alarms.emit(cleaned_df, alarm)

        # --- publish cleaned rows ---------------------------------------- #
        if self._results:
            self._results.emit(cleaned_df, df)


    def process_sync(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a DataFrame synchronously. When a HTTP request comes in, we can use this to process the data immediately."""
        validation_results = self.validator(df)
        cleaned_df, alarm_events = self.correction_engine.run(validation_results, df)
        # Note: alarms and publishing are skipped here for sync processing
        return cleaned_df

