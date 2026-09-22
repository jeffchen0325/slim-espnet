#!/usr/bin/env python3
"""Generic runner template for System-based experiments."""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Sequence
from dataclasses import dataclass, field 
from omegaconf import OmegaConf
from pathlib import Path

from espnet.utils.config_utils import (
    load_and_merge_config,
)
from espnet.utils.logging_utils import configure_logging
from espnet.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
    validate_experiment_context,
)
from espnet.utils.stages_utils import (
    resolve_stages,
    run_stages,
)

# Default stage list (can be extended/overridden by callers)
STAGE_CONFIGS = {
    "create_dataset": "training_config",
    "train_tokenizer": "training_config",
    "collect_stats": "training_config",
    "train": "training_config",
    "infer": "inference_config",
    "measure": "metrics_config",
    "pack_model": ("training_config", "publication_config"),
    "upload_model": "publication_config",
}
DEFAULT_STAGES: List[str] = list(STAGE_CONFIGS.keys())


@dataclass
class DefaultConfigs:
    #stages: List[str] = field(default_factory=lambda: ["all"])
    stages: List[str] = field(default_factory=lambda: [
        "create_dataset", "train_tokenizer", "collect_stats", "train", "infer", "measure"])
    training_config: Path = "conf/training_asr_transducer.yaml"
    inference_config: Path = "conf/inference.yaml"
    metrics_config: Path = "conf/metrics.yaml"
    publication_config: Path = "conf/publication.yaml"
    dry_run: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)

def main(
    sys_args: Sequence[str],
    system_cls,
    stages: Sequence[str] = DEFAULT_STAGES,
) -> None:
    logger = configure_logging()

    core_conf = OmegaConf.structured(DefaultConfigs)
    cli_args = OmegaConf.from_cli(sys_args)
    args = OmegaConf.merge(core_conf, cli_args)
    core_keys = {"stages", "training_config", "inference_config", "metrics_config", "extra_args"}
    free_params = {k: v for k, v in args.items() if k not in core_keys}
    args.extra_args.update(free_params)

    cli_stages = OmegaConf.select(args, "stages", default=stages)
    cli_stages = OmegaConf.to_container(cli_stages, resolve=True)
    stages_to_run = resolve_stages(cli_stages, stages)

    # -----------------------------------------
    # Load configs
    # -----------------------------------------
    # Keep default_package explicit so the recipe declares which package
    # provides the default configs, instead of relying on path-based
    # inference from the user-supplied config location.
    training_config = load_and_merge_config(
        OmegaConf.select(args, "training_config"),
        config_name="training.yaml",
        default_package=__package__,
        resolve=False,
    )
    inference_config = load_and_merge_config(
        OmegaConf.select(args, "inference_config"),
        config_name="inference.yaml",
        default_package=__package__,
        resolve=False,
    )
    metrics_config = load_and_merge_config(
        OmegaConf.select(args, "metrics_config"),
        config_name="metrics.yaml",
        default_package=__package__,
        resolve=False,
    )
    publication_config = load_and_merge_config(
        OmegaConf.select(args, "publication_config"),
        config_name="publication.yaml",
        default_package=__package__,
        resolve=False,
    )
    
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        publication_config=publication_config,
        log=logger,
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(
        training_config,
        inference_config,
        metrics_config,
        publication_config,
    )

    # Guardrail: ensure required configs exist for requested stages
    config_mapping = {
        "training_config": training_config,
        "inference_config": inference_config,
        "metrics_config": metrics_config,
        "publication_config": publication_config,
    }

    missing = [
        s
        for s in stages_to_run
        if s in STAGE_CONFIGS
        and (
            any(config_mapping.get(cfg) is None for cfg in STAGE_CONFIGS[s])
            if isinstance(STAGE_CONFIGS[s], tuple)
            else config_mapping.get(STAGE_CONFIGS[s]) is None
        )
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Config not provided for stage(s): {missing_str}. "
            "Use --training_config/--inference_config/--metrics_config/"
            "--publication_config."
        )
    
    # -----------------------------------------
    # Instantiate system
    # -----------------------------------------
    system = system_cls(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        publication_config=publication_config,
    )

    # -----------------------------------------
    # Run stages
    # -----------------------------------------
    logger.info("System: %s", system_cls.__name__)
    logger.info("Requested stages: %s", OmegaConf.select(args, "stages"))
    logger.info("Resolved stages: %s", stages_to_run)

    run_stages(
        system=system,
        stages_to_run=stages_to_run,
        args=args,
        log=logger,
    )


if __name__ == "__main__":
    # Here you should replace `YourSystemClass` with the actual system class
    # you want to use for your experiment.
    import sys
    from espnet.systems.asr.system import ASRSystem

    main(
        sys_args=sys.argv[1:],
        system_cls=ASRSystem,
    )
