# config/validation_config_schema.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_wrapped_validation_config(config: Dict[str, Any]) -> bool:
    return isinstance(config, dict) and "metadata" in config and "topics" in config


def validation_topics(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns only the topic/rule part of a validation config.

    Supports both:
    - new wrapped format: { metadata, topics }
    - legacy format: { topic: { attribute: [...] } }
    """
    if is_wrapped_validation_config(config):
        return config.get("topics", {})
    return config


def normalize_validation_config(
    config_id: str,
    config: Dict[str, Any],
    existing_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Converts legacy validation configs into the new wrapped format and
    fills metadata defaults.

    Existing created_at is preserved when updating an existing config.
    """
    now = utc_now_iso()

    existing_metadata = {}
    if existing_config and is_wrapped_validation_config(existing_config):
        existing_metadata = existing_config.get("metadata", {})

    if is_wrapped_validation_config(config):
        metadata = config.get("metadata", {})
        topics = config.get("topics", {})
    else:
        metadata = {}
        topics = config

    created_at = (
        metadata.get("created_at")
        or existing_metadata.get("created_at")
        or now
    )

    return {
        "metadata": {
            "id": metadata.get("id") or existing_metadata.get("id") or config_id,
            "name": metadata.get("name") or existing_metadata.get("name") or config_id,
            "created_at": created_at,
            "updated_at": now,
            "version": metadata.get("version") or existing_metadata.get("version") or SCHEMA_VERSION,
        },
        "topics": topics,
    }