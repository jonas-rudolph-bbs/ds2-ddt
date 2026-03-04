
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from .utils import suggest_regex 

import numpy as np
import pandas as pd

# Optional heavy deps. The profiler still works without them (it will skip the relevant steps).
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover
    IsolationForest = None
    RobustScaler = None
    DBSCAN = None

try:
    import dateparser  # type: ignore
except Exception:  # pragma: no cover
    dateparser = None



MISSING_STRINGS = {"", "null", "none", "nan", "NaN", "N/A", "na", "NA", "-", "—"}


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() in MISSING_STRINGS:
        return True
    return False


def try_float(x: Any) -> float:
    if is_missing(x):
        return float("nan")
    if isinstance(x, (int, np.integer)):
        return float(x)
    if isinstance(x, float):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return float("nan")


def robust_range(series: pd.Series, lo_q: float = 0.01, hi_q: float = 0.99) -> Optional[Tuple[float, float]]:
    """Range suggestion based on quantiles (ignore extreme tails)."""
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) < 20:
        return None
    lo = float(s.quantile(lo_q))
    hi = float(s.quantile(hi_q))
    return lo, hi


def robust_range_quantile(
    s: Union[pd.Series, Iterable[Any]],
    lo_q: Optional[float] = None,
    hi_q: Optional[float] = None,
    pad_frac: float = 0.01,
) -> Optional[Tuple[float, float]]:
    """Robust range suggestion based on adaptive quantiles + small padding."""
    s = pd.to_numeric(pd.Series(list(s)) if not isinstance(s, pd.Series) else s, errors="coerce")
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 50:
        return None

    # adaptive quantiles
    if lo_q is None or hi_q is None:
        if n < 200:
            lo_q, hi_q = 0.02, 0.98
        elif n < 2000:
            lo_q, hi_q = 0.01, 0.99
        else:
            lo_q, hi_q = 0.005, 0.995

    lo = float(s.quantile(lo_q))
    hi = float(s.quantile(hi_q))

    # padding
    span = hi - lo
    pad = pad_frac * span if span > 0 else 0.0
    return lo - pad, hi + pad


def robust_range_iqr(s: Union[pd.Series, Iterable[Any]], k: float = 1.5) -> Optional[Tuple[float, float]]:
    s = pd.to_numeric(pd.Series(list(s)) if not isinstance(s, pd.Series) else s, errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) < 50:
        return None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return float(q1), float(q3)
    return float(q1 - k * iqr), float(q3 + k * iqr)


def robust_range_mad(s: Union[pd.Series, Iterable[Any]], k: float = 6.0) -> Optional[Tuple[float, float]]:
    s = pd.to_numeric(pd.Series(list(s)) if not isinstance(s, pd.Series) else s, errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) < 50:
        return None
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    if mad == 0:
        return med, med
    sigma = 1.4826 * mad  # MAD -> std equiv
    return med - k * sigma, med + k * sigma


def count_decimal_places(x: Any) -> float:
    """Approximate decimal places for numeric-like strings/numbers."""
    if is_missing(x):
        return float("nan")
    if isinstance(x, (int, np.integer)):
        return 0.0
    if isinstance(x, float):
        s = f"{x}"
    else:
        s = str(x).strip().replace(",", ".")
    if "." in s:
        return float(len(s.split(".")[-1]))
    return 0.0


def _safe_parse_dt(x: Any):
    if dateparser is None:
        return None
    if isinstance(x, str):
        return dateparser.parse(x)
    return None


def is_timestamp_col(series: pd.Series, threshold: float = 0.9) -> bool:
    """Heuristic: treat a column as timestamp-like if dateparser can parse most of a small sample."""
    sample = series.dropna().head(100)
    if len(sample) < 5:
        return False
    if dateparser is None:
        # fallback: try pandas parsing on sample
        parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean()) >= threshold

    parsed = sample.apply(_safe_parse_dt)
    success_rate = pd.Series(parsed).notna().mean()
    return float(success_rate) >= threshold


def json_safe(obj: Any) -> Any:
    """
    Convert common pandas/numpy objects to JSON-serializable Python types.
    """
    # Scalars
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj

    # numpy scalars / dtypes
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.dtype,)):
        return str(obj)

    # pandas scalars
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None

    # containers
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]

    # pandas containers
    if isinstance(obj, pd.Series):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))

    # fallback to string (keeps API stable)
    return str(obj)


@dataclass
class ProfilerConfig:
    id_cols: Sequence[str] = ()
    ts_col: Optional[str] = None
    drop_cols: Sequence[str] = ()
    contamination: float = 0.1
    random_state: int = 42
    iso_estimators: int = 300
    dbscan_eps: float = 2.0
    dbscan_min_samples: int = 3
    max_examples: int = 3


class DataProfilerUpdated:
    """
    Drop-in replacement for the notebook "data_pipeline" functionality.

    Typical API usage:
        profiler = DataProfiler.from_records(payload, config=ProfilerConfig(id_cols=["id"], ts_col="dateCaptured"))
        result = profiler.run()   # JSON-safe dict
    """

    def __init__(self, df: pd.DataFrame, config: Optional[ProfilerConfig] = None):
        self.df = df.copy()
        self.config = config or ProfilerConfig()

        # computed artifacts
        self.attr_cols: List[str] = []
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.anomalies: Optional[pd.DataFrame] = None

    # ---------- constructors for API handlers ----------
    @classmethod
    def from_records(
        cls,
        records: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame],
        *,
        config: Optional[ProfilerConfig] = None,
    ) -> "DataProfiler":
        """
        Accepts:
          - list[dict] (common JSON payload)
          - dict (either a single record OR {"records": [...]})
          - pandas DataFrame
        """
        if isinstance(records, pd.DataFrame):
            df = records
        elif isinstance(records, list):
            df = pd.DataFrame(records)
        elif isinstance(records, dict):
            if "records" in records and isinstance(records["records"], list):
                df = pd.DataFrame(records["records"])
            else:
                df = pd.DataFrame([records])
        else:
            raise TypeError(f"Unsupported records type: {type(records).__name__}")
        return cls(df=df, config=config)

    # ---------- pipeline steps ----------
    def _prepare(self) -> None:
        # drop cols
        for c in self.config.drop_cols:
            if c in self.df.columns:
                self.df = self.df.drop(columns=[c])

        # timestamp coercion (if provided)
        if self.config.ts_col and self.config.ts_col in self.df.columns:
            self.df[self.config.ts_col] = pd.to_datetime(self.df[self.config.ts_col], errors="coerce")

        # attributes = all except id/timestamp
        id_set = set(self.config.id_cols or ())
        ts_set = {self.config.ts_col} if self.config.ts_col else set()
        self.attr_cols = [c for c in self.df.columns if c not in id_set and c not in ts_set]

    def build_column_profiles(self) -> Dict[str, Dict[str, Any]]:
        self._prepare()
        profiles: Dict[str, Dict[str, Any]] = {}

        for col in self.attr_cols:
            col_vals = self.df[col].tolist()

            miss_rate = float(np.mean([is_missing(v) for v in col_vals])) if len(col_vals) else 0.0

            type_counts = Counter(type(v).__name__ for v in col_vals if not is_missing(v))
            total_nonmiss = sum(type_counts.values()) or 1
            type_freq = {k: float(v / total_nonmiss) for k, v in type_counts.items()}

            numeric_parsed = np.array([try_float(v) for v in col_vals], dtype=float)
            numeric_rate = float(np.mean(np.isfinite(numeric_parsed))) if len(col_vals) else 0.0

            quantile_range = None
            adaptive_quantile_range = None
            iqr_range = None
            mad_range = None
            if numeric_rate >= 0.8:
                quantile_range = robust_range(pd.Series(numeric_parsed))
                adaptive_quantile_range = robust_range_quantile(pd.Series(numeric_parsed))
                iqr_range = robust_range_iqr(pd.Series(numeric_parsed))
                mad_range = robust_range_mad(pd.Series(numeric_parsed))

            top_values = None
            regex_suggestion = None
            if numeric_rate < 0.8:
                vals = [str(v) for v in col_vals if not is_missing(v)]
                c = Counter(vals)
                top_values = c.most_common(10)
                regex_suggestion = suggest_regex(
                            pd.Series([v for v in col_vals if not is_missing(v)]),
                            min_samples=50,
                            min_coverage=0.9,
                        )

            dec = np.array([count_decimal_places(v) for v in col_vals], dtype=float)
            dec = dec[np.isfinite(dec)]
            dec_mode = int(pd.Series(dec).mode().iloc[0]) if len(dec) else None

            is_dt = is_timestamp_col(pd.Series(col_vals))
            ts_parsed = pd.to_datetime(col_vals, errors="coerce") if is_dt else None
            ts_rate = float(ts_parsed.notna().mean()) if is_dt and ts_parsed is not None else 0.0

            profiles[col] = {
                "missing_rate": miss_rate,
                "type_freq": type_freq,
                "numeric_rate": numeric_rate,
                "suggested_quantile_range": quantile_range,
                "suggested_adaptive_quantile_range": adaptive_quantile_range,
                "suggested_irq_range": iqr_range,
                "suggested_mad_range": mad_range,
                "top_values": top_values,
                "timestamp_rate": ts_rate,
                "regex_suggestion": regex_suggestion,
                "decimal_mode": dec_mode,
            }

        self.profiles = profiles
        return profiles

    def detect_anomalies(self) -> pd.DataFrame:
        """
        IsolationForest over:
          - numeric parsed attributes
          - missingness flags
          - delta_sec between consecutive timestamps (if ts_col provided)
        """
        if IsolationForest is None or RobustScaler is None:
            # No sklearn available: return empty anomalies frame
            self.df["is_anomaly"] = False
            self.df["anomaly_score"] = np.nan
            self.anomalies = self.df.iloc[0:0].copy()
            return self.anomalies

        self._prepare()

        if not self.attr_cols:
            self.df["is_anomaly"] = False
            self.df["anomaly_score"] = np.nan
            self.anomalies = self.df.iloc[0:0].copy()
            return self.anomalies

        # numeric matrix + missing flags
        X_num = pd.DataFrame({c: [try_float(v) for v in self.df[c].tolist()] for c in self.attr_cols})
        X_miss = X_num.isna().astype(int).add_prefix("miss__")

        # time delta per key (in seconds)
        if self.config.ts_col and self.config.ts_col in self.df.columns:
            df_sorted = self.df.sort_values(self.config.ts_col)
            delta_sec = pd.Series(np.nan, index=df_sorted.index)
            d = df_sorted[self.config.ts_col].diff().dt.total_seconds()
            delta_sec.loc[df_sorted.index] = d
            X_time = pd.DataFrame({"delta_sec": delta_sec})
            X_time["delta_sec"] = X_time["delta_sec"].fillna(X_time["delta_sec"].median())
        else:
            X_time = pd.DataFrame({"delta_sec": np.zeros(len(self.df), dtype=float)})

        X = pd.concat([X_num, X_miss, X_time], axis=1)

        # Fill numeric NaNs with column medians (keep missing flags)
        for c in X_num.columns:
            med = X_num[c].median(skipna=True)
            X[c] = X[c].fillna(med if np.isfinite(med) else 0.0)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(
            n_estimators=int(self.config.iso_estimators),
            contamination=float(self.config.contamination),
            random_state=int(self.config.random_state),
        )
        pred = iso.fit_predict(X_scaled)  # -1 = anomaly
        self.df["is_anomaly"] = (pred == -1)
        self.df["anomaly_score"] = -iso.score_samples(X_scaled)  # higher => more anomalous

        anoms = self.df[self.df["is_anomaly"]].copy()

        # cluster anomalies (optional)
        if DBSCAN is not None and len(anoms) >= 5:
            Xa = X_scaled[self.df["is_anomaly"].values]
            labels = DBSCAN(eps=float(self.config.dbscan_eps), min_samples=int(self.config.dbscan_min_samples)).fit_predict(Xa)
            anoms["cluster"] = labels
        else:
            anoms["cluster"] = -1

        self.anomalies = anoms
        return anoms

    def suggest_rules(self) -> List[Dict[str, Any]]:
        """
        Generates rule candidates from:
          - per-column profiles (type drift, ranges, missingness, precision, timestamp, regex)
          - anomaly clusters (when available)
        """
        if not self.profiles:
            self.build_column_profiles()
        if self.anomalies is None:
            self.detect_anomalies()

        rules: List[Dict[str, Any]] = []

        # A) per-attribute static candidates from self-profile
        for col, p in self.profiles.items():
            # unexpected type drift within batch
            if p.get("type_freq"):
                dominant_type, dom_freq = max(p["type_freq"].items(), key=lambda kv: kv[1])
                if dom_freq >= 0.95 and len(p["type_freq"]) > 1:
                    rules.append(
                        {
                            "anomaly_class": "unexpected_type",
                            "attribute": col,
                            "evidence": {
                                "dominant_type": dominant_type,
                                "dominant_freq": dom_freq,
                                "type_freq": p["type_freq"],
                            },
                            "suggested_rule": f"Enforce type={dominant_type} for '{col}' (cast/parse or set to null otherwise).",
                        }
                    )

            # range constraints (if numeric)
            if p.get("suggested_quantile_range") is not None:
                lo, hi = p["suggested_quantile_range"]
                lo1, hi1 = p.get("suggested_irq_range") or (None, None)
                lo2, hi2 = p.get("suggested_mad_range") or (None, None)
                rules.append(
                    {
                        "anomaly_class": "constraint_violation/range",
                        "attribute": col,
                        "evidence": {
                            "suggested_quantile_range": [lo, hi],
                            "suggested_irq_range": [lo1, hi1],
                            "suggested_mad_range": [lo2, hi2],
                            "numeric_rate": p.get("numeric_rate", 0.0),
                        },
                        "suggested_rule": f"Flag '{col}' outside [{lo:.3g}, {hi:.3g}] (quantile-based).",
                    }
                )

            # missingness
            if float(p.get("missing_rate", 0.0)) >= 0.05:
                mr = float(p["missing_rate"])
                rules.append(
                    {
                        "anomaly_class": "missingness",
                        "attribute": col,
                        "evidence": {"missing_rate": mr},
                        "suggested_rule": f"Track missing rate for '{col}' (currently ~{mr:.1%}); alert if it spikes.",
                    }
                )

            # precision/scale drift
            if p.get("decimal_mode") is not None and float(p.get("numeric_rate", 0.0)) >= 0.8:
                rules.append(
                    {
                        "anomaly_class": "precision/scale_drift",
                        "attribute": col,
                        "evidence": {"decimal_mode": p["decimal_mode"]},
                        "suggested_rule": f"Enforce ~{p['decimal_mode']} decimal places for '{col}' (flag sudden changes).",
                    }
                )

            # timestamp consistency
            if float(p.get("timestamp_rate", 0.0)) > 0.9:
                rules.append(
                    {
                        "anomaly_class": "schema_refinement/timestamp",
                        "attribute": col,
                        "evidence": {"timestamp_rate": p["timestamp_rate"]},
                        "suggested_rule": f"Column '{col}' looks like a timestamp. Enforce ISO-8601 format and check for logical continuity (no future dates).",
                    }
                )

            # regex constraints (non-timestamp)
            if p.get("regex_suggestion") is not None and float(p.get("timestamp_rate", 0.0)) < 0.9:
                rs = p["regex_suggestion"]
                try:
                    pattern = rs.get("pattern")
                    coverage = rs.get("coverage")
                    examples = rs.get("examples")
                except Exception:
                    pattern, coverage, examples = None, None, None

                rules.append(
                    {
                        "anomaly_class": "regex_constraints",
                        "attribute": col,
                        "evidence": {"regex_coverage": coverage, "regex_examples": examples, "pattern": pattern},
                        "suggested_rule": f"Column '{col}' looks like it matches a stable pattern. Enforce it and flag violations.",
                    }
                )

        # B) cluster-driven suggestions from anomalies
        anoms = self.anomalies if self.anomalies is not None else self.df.iloc[0:0]
        if "cluster" in anoms.columns and len(anoms) > 0:
            for cl, g in anoms.groupby("cluster"):
                if int(cl) == -1:
                    continue
                cluster_idx = g.index

                miss_rates = {}
                for col in self.attr_cols:
                    miss_rates[col] = float(np.mean([is_missing(v) for v in self.df.loc[cluster_idx, col].tolist()]))

                top_miss = sorted(miss_rates.items(), key=lambda kv: kv[1], reverse=True)[:5]
                example_cols = list(self.config.id_cols) + ([self.config.ts_col] if self.config.ts_col else [])
                example_cols = [c for c in example_cols if c in self.df.columns]
                examples = g[example_cols].head(int(self.config.max_examples)).to_dict(orient="records") if example_cols else []

                rules.append(
                    {
                        "anomaly_class": "clustered_anomaly_pattern",
                        "attribute": None,
                        "evidence": {"cluster": int(cl), "top_missing_in_cluster": top_miss, "examples": examples},
                        "suggested_rule": "If this pattern repeats, add a targeted rule (e.g., drop/repair records when these fields are missing together).",
                    }
                )

        return rules

    def run(self) -> Dict[str, Any]:
        """
        Execute the whole pipeline and return a JSON-safe dict for your API response.
        """
        profiles = self.build_column_profiles()
        anoms = self.detect_anomalies()
        rules = self.suggest_rules()

        # A compact anomalies preview for the API (avoid returning the full DF by default)
        preview_cols = list(self.config.id_cols) + ([self.config.ts_col] if self.config.ts_col else [])
        preview_cols = [c for c in preview_cols if c in self.df.columns]
        if "anomaly_score" in self.df.columns:
            preview_cols += ["anomaly_score"]
        preview_cols += [c for c in self.attr_cols[:25] if c in self.df.columns]  # keep payload bounded

        anom_preview = anoms.sort_values("anomaly_score", ascending=False).head(50)[preview_cols] if len(anoms) else anoms

        records = anom_preview.to_dict(orient="records")
        wrapped = []
        for r in records:
            score = r.pop("anomaly_score", None)
            wrapped.append({"anomaly_score": score, "data": r,})


        out = {
            "row_count": int(len(self.df)),
            "column_count": int(self.df.shape[1]),
            "attr_cols": list(self.attr_cols),
            "profiles": profiles,
            "anomaly_summary": {
                "anomaly_count": int(len(anoms)),
                "anomaly_rate": float(len(anoms) / max(len(self.df), 1)),
            },
            "anomalies_preview": wrapped,
            "rule_candidates": rules[:200],
        }
        return json_safe(out)
