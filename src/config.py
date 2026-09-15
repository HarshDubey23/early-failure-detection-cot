"""
WHAT: Typed configuration loader with validation for the whole repository.

WHY:  Ground rule 5 requires that any run be exactly reproducible from the
      config file alone. A single validated config object - instead of magic
      numbers scattered through scripts - is what makes that enforceable. Every
      script takes `--config` and nothing else controls its numerics.

HOW:  YAML is parsed into nested dataclasses with explicit types. Every key is
      REQUIRED: `_require` raises a named error when one is missing, so a typo
      in a config file can never fall back to a hidden default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_TARGETS = ("answer_incorrect", "has_error")


@dataclass
class DataConfig:
    raw_dir: str
    processed_dir: str
    bigbench_mistake_dir: str
    synthetic: bool
    synthetic_n_questions: int
    synthetic_steps_range: list


@dataclass
class PreprocessingConfig:
    min_steps: int
    error_free_code: int


@dataclass
class SplitConfig:
    test_size: float
    group_field: str


@dataclass
class DetectorConfig:
    target: str               # primary label y (paper: answer incorrectness)
    secondary_target: str     # reported alongside (process error exists)
    stepwise_representation: str  # detector that drives lead time + intervention
    min_train_traces_per_step: int


@dataclass
class ProbesConfig:
    model: str
    generation_temperature: float
    top_p: float
    samples_per_question: int
    default_layer: int
    layer_sweep: list
    depth_bins: list
    n_problems: int
    max_new_tokens: int
    logistic_C: float
    logistic_max_iter: int
    mlp_hidden: list
    mlp_max_iter: int
    gru_hidden: int
    gru_epochs: int
    gru_lr: float


@dataclass
class BaselinesConfig:
    tfidf_max_features: int
    tfidf_ngram_range: list
    tfidf_min_df: int
    logistic_C: float


@dataclass
class MetricsConfig:
    bootstrap_resamples: int
    ece_bins: int


@dataclass
class AssumedInterventionConfig:
    p_verify_repair: float
    p_verify_corrupt: float
    p_regen_repair: float
    p_regen_corrupt: float
    p_revert_catches_corruption: float
    verify_tokens: int
    chars_per_token: float


@dataclass
class InterventionConfig:
    thresholds: list
    escalation_threshold: float
    token_budget: int
    self_consistency_N: list
    assumed: AssumedInterventionConfig


@dataclass
class OutputsConfig:
    results_dir: str
    tables_dir: str
    figures_dir: str
    logs_dir: str
    manifest: str
    report: str


@dataclass
class Config:
    """Root config object. Mirrors `configs/*.yaml` one-to-one."""

    seed: int
    data: DataConfig
    preprocessing: PreprocessingConfig
    split: SplitConfig
    prefix_grid: list
    detector: DetectorConfig
    probes: ProbesConfig
    baselines: BaselinesConfig
    metrics: MetricsConfig
    intervention: InterventionConfig
    outputs: OutputsConfig


def _require(mapping: dict, key: str, where: str):
    """Fetch `key` from `mapping`, raising a named error when absent.

    Args:
        mapping: The YAML section being read.
        key: Required key name.
        where: Section name used in the error message.

    Returns:
        The value stored under `key`.
    """
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(
            f"Config error: missing required key '{where}.{key}'. "
            f"No silent defaults are allowed (ground rule 5)."
        )
    return mapping[key]


def _section(raw: dict, name: str, cls, casts: dict):
    """Build dataclass `cls` from YAML section `name`, requiring every field."""
    section = _require(raw, name, "<root>")
    values = {key: cast(_require(section, key, name)) for key, cast in casts.items()}
    return cls(**values)


def _probability(value) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Config error: probability {value} outside [0, 1].")
    return value


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file.

    Args:
        path: Path to a `configs/*.yaml` file.

    Returns:
        A validated `Config` object.

    Example:
        >>> cfg = load_config("configs/quick.yaml")
        >>> cfg.seed
        42
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    data = _section(raw, "data", DataConfig, {
        "raw_dir": str, "processed_dir": str, "bigbench_mistake_dir": str,
        "synthetic": bool, "synthetic_n_questions": int, "synthetic_steps_range": list,
    })
    preprocessing = _section(raw, "preprocessing", PreprocessingConfig,
                             {"min_steps": int, "error_free_code": int})
    split = _section(raw, "split", SplitConfig, {"test_size": float, "group_field": str})
    if not 0.0 < split.test_size < 1.0:
        raise ValueError("Config error: split.test_size must be in (0, 1).")
    if split.group_field != "question_id":
        raise ValueError("Config error: split.group_field must be 'question_id' (rule P6).")

    prefix_grid = sorted(float(x) for x in _require(raw, "prefix_grid", "<root>"))
    if not all(0.0 < f <= 1.0 for f in prefix_grid) or prefix_grid[-1] != 1.0:
        raise ValueError("Config error: prefix_grid entries must be in (0, 1] and include 1.0.")

    detector = _section(raw, "detector", DetectorConfig, {
        "target": str, "secondary_target": str, "stepwise_representation": str,
        "min_train_traces_per_step": int,
    })
    for label in (detector.target, detector.secondary_target):
        if label not in VALID_TARGETS:
            raise ValueError(f"Config error: detector target '{label}' not in {VALID_TARGETS}.")
    if detector.stepwise_representation not in ("surface", "lexical"):
        raise ValueError("Config error: detector.stepwise_representation must be surface|lexical.")

    probes = _section(raw, "probes", ProbesConfig, {
        "model": str, "generation_temperature": float, "top_p": float,
        "samples_per_question": int, "default_layer": int, "layer_sweep": list,
        "depth_bins": list, "n_problems": int, "max_new_tokens": int,
        "logistic_C": float, "logistic_max_iter": int, "mlp_hidden": list,
        "mlp_max_iter": int, "gru_hidden": int, "gru_epochs": int, "gru_lr": float,
    })
    baselines = _section(raw, "baselines", BaselinesConfig, {
        "tfidf_max_features": int, "tfidf_ngram_range": list, "tfidf_min_df": int,
        "logistic_C": float,
    })
    metrics = _section(raw, "metrics", MetricsConfig,
                       {"bootstrap_resamples": int, "ece_bins": int})

    intervention_raw = _require(raw, "intervention", "<root>")
    assumed = _section(intervention_raw, "assumed", AssumedInterventionConfig, {
        "p_verify_repair": _probability, "p_verify_corrupt": _probability,
        "p_regen_repair": _probability, "p_regen_corrupt": _probability,
        "p_revert_catches_corruption": _probability, "verify_tokens": int,
        "chars_per_token": float,
    })
    intervention = InterventionConfig(
        thresholds=[float(t) for t in _require(intervention_raw, "thresholds", "intervention")],
        escalation_threshold=float(_require(intervention_raw, "escalation_threshold",
                                            "intervention")),
        token_budget=int(_require(intervention_raw, "token_budget", "intervention")),
        self_consistency_N=[int(n) for n in _require(intervention_raw, "self_consistency_N",
                                                     "intervention")],
        assumed=assumed,
    )
    outputs = _section(raw, "outputs", OutputsConfig, {
        "results_dir": str, "tables_dir": str, "figures_dir": str, "logs_dir": str,
        "manifest": str, "report": str,
    })
    return Config(seed=int(_require(raw, "seed", "<root>")), data=data,
                  preprocessing=preprocessing, split=split, prefix_grid=prefix_grid,
                  detector=detector, probes=probes, baselines=baselines, metrics=metrics,
                  intervention=intervention, outputs=outputs)
