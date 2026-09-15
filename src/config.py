"""
WHAT: Typed configuration loader with validation for the whole repository.

WHY:  Ground rule 5 requires that any run be exactly reproducible from the
      config file alone. A single validated config object - instead of magic
      numbers scattered through scripts - is what makes that enforceable. Every
      script takes `--config` and nothing else controls its numerics.

HOW:  YAML is parsed into nested dataclasses with explicit types. `load_config`
      validates presence and ranges of the fields the pipeline depends on and
      fails loudly (no silent defaults) when a required field is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    synthetic: bool = False
    synthetic_n_questions: int = 40
    synthetic_steps_range: list = field(default_factory=lambda: [3, 8])


@dataclass
class PreprocessingConfig:
    min_steps: int = 2
    error_free_code: int = -1


@dataclass
class SplitConfig:
    test_size: float = 0.25
    group_field: str = "question_id"


@dataclass
class ProbesConfig:
    model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    generation_temperature: float = 0.7
    samples_per_question: int = 4
    default_layer: int = 18
    layer_sweep: list = field(default_factory=lambda: [8, 12, 18, 24])
    depth_bins: list = field(
        default_factory=lambda: [[0.0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.01]]
    )
    n_problems: int = 120
    max_new_tokens: int = 400
    logistic_C: float = 0.1
    logistic_max_iter: int = 3000


@dataclass
class BaselinesConfig:
    tfidf_max_features: int = 20000
    tfidf_ngram_range: list = field(default_factory=lambda: [1, 2])
    tfidf_min_df: int = 2


@dataclass
class MetricsConfig:
    bootstrap_resamples: int = 1000
    ece_bins: int = 10


@dataclass
class InterventionConfig:
    thresholds: list = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    escalation_threshold: float = 0.8
    token_budget: int = 256
    self_consistency_N: list = field(default_factory=lambda: [2, 4, 8])


@dataclass
class OutputsConfig:
    results_dir: str = "results"
    tables_dir: str = "results/tables"
    figures_dir: str = "results/figures"
    logs_dir: str = "results/logs"
    manifest: str = "results/manifest.json"


@dataclass
class Config:
    """Root config object. Mirrors `configs/*.yaml` one-to-one."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    prefix_grid: list = field(
        default_factory=lambda: [0.10, 0.25, 0.40, 0.50, 0.75, 1.00]
    )
    probes: ProbesConfig = field(default_factory=ProbesConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


def _require(mapping: dict, key: str, where: str):
    """Fetch `key` from `mapping`, raising a named error when absent.

    Args:
        mapping: The YAML section being read.
        key: Required key name.
        where: Section name used in the error message.

    Returns:
        The value stored under `key`.
    """
    if key not in mapping:
        raise ValueError(
            f"Config error: missing required key '{where}.{key}'. "
            f"No silent defaults are allowed (ground rule 5)."
        )
    return mapping[key]


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

    cfg = Config()
    cfg.seed = int(raw.get("seed", 42))

    if "data" in raw:
        d = raw["data"]
        cfg.data = DataConfig(
            raw_dir=d.get("raw_dir", "data/raw"),
            processed_dir=d.get("processed_dir", "data/processed"),
            synthetic=bool(d.get("synthetic", False)),
            synthetic_n_questions=int(d.get("synthetic_n_questions", 40)),
            synthetic_steps_range=list(d.get("synthetic_steps_range", [3, 8])),
        )
    if "preprocessing" in raw:
        p = raw["preprocessing"]
        cfg.preprocessing = PreprocessingConfig(
            min_steps=int(p.get("min_steps", 2)),
            error_free_code=int(p.get("error_free_code", -1)),
        )
    if "split" in raw:
        s = raw["split"]
        cfg.split = SplitConfig(
            test_size=float(s.get("test_size", 0.25)),
            group_field=str(s.get("group_field", "question_id")),
        )
        if not 0.0 < cfg.split.test_size < 1.0:
            raise ValueError("Config error: split.test_size must be in (0, 1).")
    if "prefix_grid" in raw:
        grid = [float(x) for x in raw["prefix_grid"]]
        if not all(0.0 < f <= 1.0 for f in grid):
            raise ValueError("Config error: every prefix_grid entry must be in (0, 1].")
        cfg.prefix_grid = sorted(grid)
    if "probes" in raw:
        pr = raw["probes"]
        cfg.probes = ProbesConfig(
            model=str(pr.get("model", ProbesConfig.model)),
            generation_temperature=float(pr.get("generation_temperature", 0.7)),
            samples_per_question=int(pr.get("samples_per_question", 4)),
            default_layer=int(pr.get("default_layer", 18)),
            layer_sweep=list(pr.get("layer_sweep", [8, 12, 18, 24])),
            depth_bins=list(pr.get("depth_bins",
                              [[0.0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.01]])),
            n_problems=int(pr.get("n_problems", 120)),
            max_new_tokens=int(pr.get("max_new_tokens", 400)),
            logistic_C=float(pr.get("logistic_C", 0.1)),
            logistic_max_iter=int(pr.get("logistic_max_iter", 3000)),
        )
    if "baselines" in raw:
        b = raw["baselines"]
        cfg.baselines = BaselinesConfig(
            tfidf_max_features=int(b.get("tfidf_max_features", 20000)),
            tfidf_ngram_range=list(b.get("tfidf_ngram_range", [1, 2])),
            tfidf_min_df=int(b.get("tfidf_min_df", 2)),
        )
    if "metrics" in raw:
        m = raw["metrics"]
        cfg.metrics = MetricsConfig(
            bootstrap_resamples=int(m.get("bootstrap_resamples", 1000)),
            ece_bins=int(m.get("ece_bins", 10)),
        )
    if "intervention" in raw:
        iv = raw["intervention"]
        cfg.intervention = InterventionConfig(
            thresholds=list(iv.get("thresholds", [0.3, 0.5, 0.7, 0.9])),
            escalation_threshold=float(iv.get("escalation_threshold", 0.8)),
            token_budget=int(iv.get("token_budget", 256)),
            self_consistency_N=list(iv.get("self_consistency_N", [2, 4, 8])),
        )
    if "outputs" in raw:
        o = raw["outputs"]
        cfg.outputs = OutputsConfig(
            results_dir=o.get("results_dir", "results"),
            tables_dir=o.get("tables_dir", "results/tables"),
            figures_dir=o.get("figures_dir", "results/figures"),
            logs_dir=o.get("logs_dir", "results/logs"),
            manifest=o.get("manifest", "results/manifest.json"),
        )
    return cfg
