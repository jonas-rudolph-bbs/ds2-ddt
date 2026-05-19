# config_store.py
from __future__ import annotations

import json, os, tempfile, fnmatch
from pathlib import Path
from typing import Any, Dict, Optional, Union, Iterable
from .validation_config_schema import normalize_validation_config

class ConfigManager:
    """
    Unified loader/saver/deleter for JSON configs.

    Layout (kept compatible with your codebase):
      - base/
        - generated_mqtt_config.json
        - validations/
            <config_id>.json
    """

    def __init__(self, base_path: Union[str, Path] = "config") -> None:
        self.base_path = Path(base_path)

    # ---------- path helpers ----------
    def _mqtt_path(self) -> Path:
        return self.base_path / "generated_mqtt_config.json"

    def _validation_dir(self) -> Path:
        return self.base_path / "validations"

    def _validation_path(self, config_id: str) -> Path:
        if not config_id:
            raise ValueError("validation cfg_type requires a non-empty config_id")
        return self._validation_dir() / f"{config_id}.json"

    def _resolve_target(self, cfg_type: str, config_id: Optional[str]) -> Path:
        if cfg_type == "mqtt":
            return self._mqtt_path()
        if cfg_type == "validation":
            return self._validation_path(config_id or "")
        raise ValueError("cfg_type must be 'mqtt' or 'validation'")

    # ---------- read / load ----------
    def read_current(self, cfg_type: str, config_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return parsed JSON or None if missing."""
        p = self._resolve_target(cfg_type, config_id)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_file(self, path: Union[str, Path]) -> Any:
        """Load an arbitrary JSON file."""
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"JSON file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_dir(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load all *.json files in a directory into a dict keyed by file stem."""
        d = Path(path)
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")
        out: Dict[str, Any] = {}
        for fp in sorted(d.glob("*.json")):
            with fp.open("r", encoding="utf-8") as f:
                out[fp.stem] = json.load(f)
        if not out:
            raise FileNotFoundError(f"No JSON files in directory: {d}")
        return out

    def load(self, path: Optional[Union[str, Path]] = None) -> Union[Dict[str, Any], Any]:
        """
        If path is:
          - None: load all JSON files in base (not recursive).
          - file: return that JSON.
          - dir : return dict of all JSON files in that dir.
        """
        target = self.base_path if path is None else (self.base_path / path if not Path(path).is_absolute() else Path(path))
        if target.is_file():
            return self.load_file(target)
        if target.is_dir():
            return self.load_dir(target)
        raise FileNotFoundError(f"Config path '{target}' does not exist")

    def list_validation_ids(self) -> list[str]:
        """Return all validation config_ids (file stems)."""
        d = self._validation_dir()
        if not d.exists():
            return []
        return sorted(fp.stem for fp in d.glob("*.json"))

    # ---------- write (atomic) ----------
    def write_atomic(self, cfg_type: str, config_id: Optional[str], content: Dict[str, Any]) -> Path:
        """
        Atomically write JSON for either:
        - mqtt: config/generated_mqtt_config.json
        - validation: config/validations/{config_id}.json
        """
        target = self._resolve_target(cfg_type, config_id)
        target.parent.mkdir(parents=True, exist_ok=True)

        if cfg_type == "validation":
            if not config_id:
                raise ValueError("validation cfg_type requires a non-empty config_id")

            existing = self.read_current("validation", config_id)
            content = normalize_validation_config(
                config_id=config_id,
                config=content,
                existing_config=existing,
            )

        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=target.stem + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                json.dump(content, tmp, ensure_ascii=False, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, target)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        return target

    # ---------- delete ----------
    def delete(self, cfg_type: str, config_id: Optional[str] = None, *,
               pattern: Optional[str] = None, missing_ok: bool = True, pipelines: Dict[str, Any]) -> list[Path]:
        """
        Delete config files and return list of removed Paths.

        For cfg_type='mqtt':
          - ignores config_id/pattern and deletes generated_mqtt_config.json (if present).

        For cfg_type='validation':
          - if config_id is provided: delete exactly that file.
          - else if pattern is provided: delete all validation files whose stem matches pattern (fnmatch).
          - else: delete nothing (return []). You can pass pattern='*' to delete all validations.

        missing_ok controls whether missing files raise (False) or are skipped (True).
        """
        removed: list[Path] = []
        if cfg_type == "mqtt":
            p = self._mqtt_path()
            if p.exists():
                p.unlink()
                removed.append(p)
            elif not missing_ok:
                raise FileNotFoundError(f"MQTT config not found: {p}")
            return removed

        if cfg_type != "validation":
            raise ValueError("cfg_type must be 'mqtt' or 'validation'")

        d = self._validation_dir()
        if not d.exists():
            if missing_ok:
                return removed
            raise FileNotFoundError(f"Validation directory not found: {d}")

        if config_id:
            p = self._validation_path(config_id)
            in_use = self._check_validation_in_use(config_id, pipelines)
            if p.exists():
                if not in_use:     
                    p.unlink()
                    removed.append(p)
                else:
                    raise RuntimeError(f"Cannot delete validation config '{config_id}' as it is currently in use by a mqtt pipeline.")
            elif not missing_ok:
                raise FileNotFoundError(f"Validation config not found: {p}")
            return removed

        # pattern-based bulk delete
        if pattern is None:
            return removed  # explicit no-op unless asked
        for fp in d.glob("*.json"):
            if fnmatch.fnmatch(fp.stem, pattern):
                fp.unlink()
                removed.append(fp)
        if not removed and not missing_ok:
            raise FileNotFoundError(f"No validation files matched pattern '{pattern}' in {d}")
        return removed

    def _check_validation_in_use(self, config_id: str, pipelines: Dict[str, Any]) -> bool:
        """
        Check if the validation config is currently in use by any pipeline.
        """
        for pipeline in pipelines.values():
            if pipeline.validator.config_name.startswith(config_id):
                return True
        return False