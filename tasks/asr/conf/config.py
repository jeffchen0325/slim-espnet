from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# common: 跨阶段共享的基础路径、并行、数据集等配置
# ============================================================

@dataclass
class ParallelConfig:
    env: str = "local"
    n_workers: int = 1


@dataclass
class CreateDatasetConfig:
    func: str = "src.creating_dataset.create_dataset"
    dataset_dir: str = "/path/to/your/dataset"


@dataclass
class DatasetEntry:
    name: Optional[str] = None
    data_src: Optional[str] = None
    data_src_args: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[Dict[str, Any]] = None


@dataclass
class DatasetPreprocessorConfig:
    _target_: str = "espnet3.components.data.data_organizer.DataOrganizer"


@dataclass
class DatasetConfig:
    _target_: str = "espnet3.components.data.data_organizer.DataOrganizer"
    _recursive_: bool = False
    recipe_dir: str = "."
    train: Optional[List[DatasetEntry]] = None
    valid: Optional[List[DatasetEntry]] = None
    test: Optional[List[DatasetEntry]] = None
    preprocessor: DatasetPreprocessorConfig = field(default_factory=DatasetPreprocessorConfig)


@dataclass
class CommonConfig:
    """所有阶段共享的基础配置"""
    num_device: int = 1
    num_nodes: int = 1
    # [FIX] seed 移入 common，确保所有阶段都能统一控制随机性
    seed: Optional[int] = None
    
    recipe_dir: str = "."
    data_dir: str = "${recipe_dir}/data"
    exp_tag: Optional[str] = None
    exp_dir: str = "${recipe_dir}/exp/${exp_tag}"
    stats_dir: str = "${recipe_dir}/exp/stats"
    dataset_dir: str = "/path/to/your/dataset"
    inference_dir: str = "${exp_dir}/${self_name:}"
    create_dataset: CreateDatasetConfig = field(default_factory=CreateDatasetConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


# ============================================================
# training_config: 训练专属配置
# ============================================================

@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.Adam"
    lr: float = 0.002
    weight_decay: float = 0.000001


@dataclass
class SchedulerConfig:
    _target_: str = "espnet2.schedulers.warmup_lr.WarmupLR"
    warmup_steps: int = 15000


@dataclass
class CollateFnConfig:
    _target_: str = "espnet2.train.collate_fn.CommonCollateFn"
    int_pad_value: int = -1


@dataclass
class BatchesConfig:
    type: str = "unsorted"
    shape_files: List[str] = field(default_factory=lambda: ["${stats_dir}/train/feats_shape"])
    batch_size: int = 4
    batch_bins: int = 4000000


@dataclass
class IterFactoryConfig:
    _target_: str = "espnet2.iterators.sequence_iter_factory.SequenceIterFactory"
    shuffle: bool = True
    collate_fn: CollateFnConfig = field(default_factory=CollateFnConfig)
    batches: BatchesConfig = field(default_factory=BatchesConfig)


@dataclass
class DataLoaderSplitConfig:
    total_shards: int = 1
    dist_world_size: int = 1
    iter_factory: IterFactoryConfig = field(default_factory=IterFactoryConfig)


@dataclass
class DataLoaderConfig:
    collate_fn: CollateFnConfig = field(default_factory=CollateFnConfig)
    train: DataLoaderSplitConfig = field(default_factory=DataLoaderSplitConfig)
    valid: DataLoaderSplitConfig = field(default_factory=lambda: DataLoaderSplitConfig(
        iter_factory=IterFactoryConfig(shuffle=False)
    ))


@dataclass
class LoggerConfig:
    _target_: str = "lightning.pytorch.loggers.TensorBoardLogger"
    save_dir: str = "${exp_dir}/tensorboard"
    name: str = "tb_logger"


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    # [FIX] 使用绝对路径插值，兼容 ESPnetFullConfig 嵌套结构
    devices: Any = "${common.num_device}"
    num_nodes: Any = "${common.num_nodes}"
    accumulate_grad_batches: int = 1
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 100
    max_epochs: int = 100
    logger: List[LoggerConfig] = field(default_factory=lambda: [LoggerConfig()])
    strategy: str = "auto"


@dataclass
class BestModelCriterionEntry:
    monitor: str = "valid/loss"
    top_k: int = 3
    mode: str = "min"


@dataclass
class TrainingConfig:
    """训练阶段专属配置"""
    parallel: ParallelConfig = field(default_factory=lambda: ParallelConfig(
        env="local",
        n_workers=8  # [建议] 训练默认给较高 worker 数
    ))
    task: Optional[str] = None
    tokenizer: Optional[Dict[str, Any]] = None
    model: Optional[Dict[str, Any]] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    scheduler_interval: str = "step"
    scheduler_monitor: Optional[str] = None
    best_model_criterion: List[BestModelCriterionEntry] = field(
        default_factory=lambda: [BestModelCriterionEntry()]
    )
    init: Optional[str] = None
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    fit: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# inference_config: 推理专属配置
# ============================================================

@dataclass
class InferenceProviderConfig:
    _target_: str = "espnet3.systems.base.inference_provider.InferenceProvider"


@dataclass
class InferenceRunnerConfig:
    _target_: str = "espnet3.systems.base.inference_runner.InferenceRunner"


@dataclass
class InferenceConfig:
    """推理阶段专属配置"""
    parallel: ParallelConfig = field(default_factory=lambda: ParallelConfig(
        env="local",
        n_workers=1  # 推理默认保守
    ))
    input_key: str = "speech"
    output_keys: Optional[List[str]] = None
    idx_key: str = "utt_id"
    batch_size: Optional[int] = None
    output_fn: Optional[Dict[str, Any]] = None
    output_artifacts: Optional[Dict[str, Any]] = None
    provider: InferenceProviderConfig = field(default_factory=InferenceProviderConfig)
    runner: InferenceRunnerConfig = field(default_factory=InferenceRunnerConfig)


# ============================================================
# metrics_config: 评估指标专属配置
# ============================================================

@dataclass
class MetricDef:
    _target_: str = ""
    ref_key: str = "ref"
    hyp_key: str = "hyp"
    clean_types: Optional[List[str]] = None
    bpemodel: Optional[str] = None


@dataclass
class MetricEntry:
    metric: MetricDef = field(default_factory=MetricDef)


@dataclass
class MetricsConfig:
    """评估指标专属配置"""
    metrics: List[MetricEntry] = field(default_factory=lambda: [
        MetricEntry(metric=MetricDef(_target_="espnet3.systems.asr.metrics.wer.WER")),
        MetricEntry(metric=MetricDef(_target_="espnet3.systems.asr.metrics.cer.CER")),
    ])


# ============================================================
# publication_config: 模型打包、上传、Demo 专属配置
# ============================================================

@dataclass
class PackModelConfig:
    out_dir: str = "${exp_dir}/model_pack"
    allow_overwrite: bool = False
    inference_dir: str = "${inference_dir}"
    include_model_detail: bool = False
    readme: str = "${config_path:../src/hf_model_readme.md}"
    include: List[str] = field(default_factory=lambda: ["src", "dataset"])
    exclude: List[str] = field(default_factory=lambda: [
        "inference", "inference_*", "last.ckpt",
        "**/step*.ckpt", "**/*.log",
        "**/tensorboard/**", "**/wandb/**",
    ])


@dataclass
class UploadModelConfig:
    hf_repo: str = "espnet/${set_corpus_and_system:}_${exp_tag}"
    private: bool = False
    update: bool = False
    delete_patterns: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class DemoModelConfig:
    dir_or_tag: str = "exp/${exp_tag}/model_pack"
    trust_user_code: bool = False
    call_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UIComponent:
    key: str = ""
    type: str = ""
    label: str = ""


@dataclass
class UIConfig:
    app_script: str = "src/app.py"
    title: str = "${set_corpus_and_system:} demo"
    description: str = "README.md"
    inputs: List[UIComponent] = field(default_factory=lambda: [
        UIComponent(key="speech", type="audio", label="Input Audio")
    ])
    outputs: List[UIComponent] = field(default_factory=lambda: [
        UIComponent(key="hyp", type="text", label="Transcription")
    ])


@dataclass
class ReadmeContext:
    title: str = "${ui.title}"
    emoji: str = "\U0001f399\ufe0f"
    color_from: str = "blue"
    color_to: str = "green"
    sdk: str = "gradio"
    app_file: str = "app.py"
    pinned: str = "false"
    license: str = "apache-2.0"
    tags: str = "- espnet\n- espnet3\n- gradio\n- speech-recognition\n- automatic-speech-recognition"
    hf_repo: str = "${upload_demo.hf_repo}"
    model_ref: str = "${model.dir_or_tag}"


@dataclass
class PackDemoConfig:
    out_dir: str = "demo"
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=lambda: [
        "git+https://github.com/espnet/espnet.git"
    ])
    readme: str = "${config_path:../src/hf_demo_readme.md}"
    readme_context: ReadmeContext = field(default_factory=ReadmeContext)


@dataclass
class UploadDemoConfig:
    hf_repo: str = "espnet/${set_corpus_and_system:}_${exp_tag}"
    update: bool = False
    delete_patterns: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class PublicationConfig:
    """模型发布、Demo 打包与上传专属配置"""
    pack_model: PackModelConfig = field(default_factory=PackModelConfig)
    upload_model: UploadModelConfig = field(default_factory=UploadModelConfig)
    demo_model: DemoModelConfig = field(default_factory=DemoModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    pack_demo: PackDemoConfig = field(default_factory=PackDemoConfig)
    upload_demo: UploadDemoConfig = field(default_factory=UploadDemoConfig)


# ============================================================
# 顶层聚合配置
# ============================================================

@dataclass
class ESPnetFullConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    publication: PublicationConfig = field(default_factory=PublicationConfig)