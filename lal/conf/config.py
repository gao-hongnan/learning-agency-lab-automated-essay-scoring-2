# pylint: disable=all
from __future__ import annotations

import os
from enum import Enum
from typing import Any, List, Literal, Type

import modal
import modal.gpu
import torch
from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
from modal import App, Image, Volume
from peft import TaskType
from pydantic import BaseModel, Field, field_validator

ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "false").lower() == "true"
IN_MODAL = os.environ.get("IN_MODAL", "false").lower() == "true"
ALLOWED_DTYPES = ["float16", "float32", "float64", "bfloat16"]


class Constants(str, Enum):
    """Enum class for constants (simple data types) used in the app."""

    MODAL_VERSION = "0.62.181"
    APP_NAME = "lal-kaggle-chris"
    CACHE_DIR = "/root/.cache/huggingface" if IN_MODAL else "./.cache/huggingface"
    SOURCE_ARTIFACTS_DIR = "artifacts-volume"
    TARGET_ARTIFACTS_DIR = "/artifacts"
    TIMEOUT = "86400"
    CONTAINER_IDLE_TIMEOUT = "600"
    PRETRAINED_MODEL_NAME_OR_PATH = "mistralai/Mistral-7B-Instruct-v0.2"  # "openai-community/gpt2"

    def __str__(self) -> str:
        """Return the string representation of the constant.

        .. code-block:: python
            print(Constants.APP_NAME) # "golden-gate-bridge-repeng"
        """
        return str.__str__(self)


def download_model_weights() -> None:
    """Download model weights from huggingface hub and cache it to `CACHE_DIR`."""
    snapshot_download(
        repo_id=Constants.PRETRAINED_MODEL_NAME_OR_PATH,
        cache_dir=Constants.CACHE_DIR,
    )


IMAGE = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "wandb==0.16.3",
        "rich==13.7.1",
        "pandas",
        "scikit-learn",
        "accelerate==0.31.0",
        "datasets==2.20.0",
        "sentencepiece",
        "transformers==4.41.2",
        "torch~=2.2.0",
        "torchvision~=0.16",
        "triton~=2.2.0",
        "peft==0.11.1",
        "omniverse==0.0.37",
        "tabulate",
    )
    .run_function(
        download_model_weights,
        secrets=[
            modal.Secret.from_name("huggingface"),
        ],
    )
)
app = App(
    name=Constants.APP_NAME,
    image=IMAGE,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
        *([modal.Secret.from_name("wandb")] if ALLOW_WANDB else []),
    ],
)

VOLUME = Volume.from_name(label=Constants.SOURCE_ARTIFACTS_DIR, create_if_missing=True)
H100_80_GPU = modal.gpu.H100(count=1)
A10_24_GPU = modal.gpu.A10G(count=1)
A100_40_GPU = modal.gpu.A100(count=1, size="40GB")
A100_80_GPU = modal.gpu.A100(count=1, size="80GB")


class WandbConfig(BaseModel):
    """Base class for wandb configuration."""


class DatasetConfig(BaseModel):
    """Base class for dataset configuration."""


class TokenizerConfig(BaseModel):
    """Base class for tokenizer configuration."""


class LowRankConfig(BaseModel):
    """Base class for LoRA configuration. See `LoraConfig` from the peft library."""


class LanguageModelConfig(BaseModel):
    """Base class for model configuration."""


class Shared(BaseModel):
    """Shared configurations."""

    # fmt: off
    # TODO: misc mutable stats - these actually belong to states - fix this
    # by propagating these to State(...)

    # some misc stats
    task: Literal["CAUSAL_LM", "SINGLE_LABEL_CLASSIFICATION", "MULTI_LABEL_CLASSIFICATION", "REGRESSION"] = "SINGLE_LABEL_CLASSIFICATION" # NOTE: this is similar to task_type in lora but difference is task_type in lora dont have regression for instance

    # wandb config
    project: str = "learning-agency-lab-automated-essay-scoring-2"
    entity: str | None = "hongnangao"
    job_type: Literal["train", "pretrain", "fullfit", "train_with_external", "debug"] = "train" # NOTE: interpolate in other places like loading data
    tags: List[str] | None = ["CAUSAL_LM"]
    group: str | None = None
    notes: str | None = None
    mode: str | None = None
    name: str | None = None

    # dataset config
    train_filepath: str | None = None
    valid_filepath: str | None = None
    pretraining_data_filepath: str | os.PathLike[str] | None = None
    external_data_filepath: str | os.PathLike[str] | None = None
    predicted_prompt_filepath: str | os.PathLike[str] | None = None
    train_topic_filepath: str | os.PathLike[str] | None = None
    topics_map_filepath: str | os.PathLike[str] | None = None

    ## prompting stuff
    system_prompt: str = "Please read the following essay and assign a score of 1,2,3,4,5,6 where 6 is the best. Output only a single number with no explanation.\n\n"

    ## dataset columns
    essay_id: str = "essay_id"
    full_text: str = "full_text"
    score: str = "score"
    label: str = "label"
    description: str = "description"

    ## dataset split
    resample_strategy: Literal["StratifiedKFold", "KFold", "GroupKFold", "StratifiedGroupKFold"] = "StratifiedKFold"
    resample_params: dict[str, Any] = {
        "n_splits": 7,
        "shuffle": True,
        "random_state": 1992,
    }
    group_by: str | None = None
    stratify_by: str | None = "score"
    fold_column: Literal["fold"] = "fold"
    target_column: str = "label"
    fold: int = 0
    debug_samples: int = 128

    # tokenizer config
    ## from_pretrained(...)
    padding_side: Literal["left", "right"] = "left"

    ## tokenizer(...)
    max_length: int = 1024  # alias context_window/context_length
    truncation: (
        Literal["do_not_truncate", "longest_first", "only_first", "only_second"] | bool
    ) = True
    return_tensors: Literal["pt", "tf", "np"] | None = None
    add_special_tokens: bool = False
    padding: Literal["longest", "max_length", "do_not_pad"] | bool = "longest"

    ## preprocess(...) and misc
    return_tokenized_text: bool = True

    # low rank config: able to unpack **
    task_type: TaskType = TaskType.CAUSAL_LM
    inference_mode: bool = False
    r: int = 32  # attention heads
    lora_alpha: int = 16  # regularization
    lora_dropout: float = 0.1  # dropout
    target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
    ]
    bias: Literal["none", "all", "lora_only"] = "none"
    modules_to_save: List[str] | None = None

    # language model config: able to unpack **
    force_download: bool = False
    device_map: str | dict[str, int] | None = "auto"
    torch_dtype: torch.dtype = Field(default=None)
    load_in_4bit: bool = False
    output_hidden_states: bool = Field(
        default=False,
        description="Read https://huggingface.co/docs/transformers/en/main_classes/output",
    )
    output_attentions: bool = Field(
        default=False,
        description="Read https://huggingface.co/docs/transformers/en/main_classes/output",
    )

    ## pooler
    pooler_type: Literal["context", "mean", "attention", "gem"] | None = None
    pooler_config: dict[str, Any] = {}

    cls_type: str = "vanilla"

    # criterion
    criterion: Literal["mse", "cross_entropy", "bce","smooth_l1_with_mse", "reg_cls_loss", "huber", "ordinal_loss", "ordinal_reg_loss"] | None = None # ordinal-log-loss, cross-entropy, mse
    criterion_config: dict[str, Any] = {}

    # optimizer
    # optimizer_type

    # init config
    init_config: dict[str, Any] = {}

    # ???
    pretrained_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2"
    load_backbone_only: bool = False
    cache_dir: str = "/root/.cache/huggingface"
    target_artifacts_dir: str = "/artifacts"
    verbose: bool = False
    show_model_summary: bool = False
    dry_run: bool = False
    default: bool = True # default model from hf, no custom model

    load_from: str | None = None

    # model misc
    inference: bool = False
    use_lora: bool = True
    num_layers_to_remove: int | None = None
    reinitialize_n_layers_of_backbone: int = 0
    enable_gradient_checkpointing: bool = False

    # freeze
    freeze_these_layers_indices: List[int] | None = None
    freeze_embeddings: bool = False

    # classification/regression
    num_labels: int = 6

    # seeding
    seed: int = 42
    seed_torch: bool = True
    set_torch_deterministic: bool = False

    # training args -> take from training arg in hf or custom
    ## below is inside training args
    output_dir: str | None = None
    overwrite_output_dir: bool = True
    ddp_find_unused_parameters: bool | None = None # False if DDP
    do_train: bool = Field(
        default=True,
        description="Whether to run training or not. This argument is not directly used by `Trainer`, "
                    "it's intended to be used by your training/evaluation scripts instead. "
                    "See the example scripts for more details."
    )
    do_eval: bool = Field(
        default=True,
        description="Whether to run evaluation on the validation set or not. Will be set to `True` if `eval_strategy` "
                    "is different from 'no'. This argument is not directly used by `Trainer`, "
                    "it's intended to be used by your training/evaluation scripts instead. "
                    "See the example scripts for more details."
    )
    do_predict: bool = Field(
        default=False,
        description="Whether to run predictions on the test set or not. This argument is not directly used by `Trainer`, "
                    "it's intended to be used by your training/evaluation scripts instead. "
                    "See the example scripts for more details."
    )
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    prediction_loss_only: bool = False
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1 # derived later from effective batch size
    learning_rate: float | None = None # derived later from base learning rate
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 2
    max_steps: int | None = None # overrides num_train_epochs
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: dict[str, Any] = {}
    warmup_ratio: float = 0.05
    warmup_steps: int = 0
    logging_dir: str | None = None
    logging_strategy: Literal["no", "steps", "epoch"] = "steps"
    logging_first_step: bool = True
    logging_steps: int | None = None # derived in code
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_steps: int | None = None # derived in code
    save_total_limit: int = 3
    save_safetensors: bool = True
    save_on_each_node: bool = False
    save_only_model: bool = False
    restore_callback_states_from_checkpoint: bool = False
    data_seed: int | None = None
    dataloader_num_workers: int = 0
    disable_tqdm: bool = False
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    label_smoothing_factor: float = 0.0
    resume_from_checkpoint: str | None = None
    eval_steps: int | None = None
    report_to: str | List[str] = "none"
    fp16: bool = False
    bf16: bool = False
    optim: str = "adamw_torch"
    full_determinism: bool = False
    gradient_checkpointing: bool = False
    include_tokens_per_second: bool = False

    ## not inside the training args
    base_learning_rate: float = 6.25e-06 # this works ok for lm 5e-5 # we derive the real lr later.
    desired_effective_batch_size: int = 16 # this means no matter how many gpus, world size or what not, you want a 16 global batch size.
    enable_mixed_precision: bool = True
    scheduler_specific_kwargs: dict[str, Any] = {}
    very_custom_optimizer_group: bool = False
    layer_wise_learning_rate_decay: float | None = Field(default=None)
    # fmt: on

    class Config:
        """Pydantic configuration."""

        protected_namespaces = ()
        arbitrary_types_allowed = True
        use_enum_values = True

    @field_validator("torch_dtype", mode="before")
    @classmethod
    def convert_torch_dtype(cls: Type[Shared], v: torch.dtype | str) -> torch.dtype:
        if isinstance(v, torch.dtype):
            return v

        if v is None:
            return torch.float32  # type: ignore[unreachable]

        if isinstance(v, str):
            assert v in ALLOWED_DTYPES, f"Invalid torch dtype: {v}"
            try:
                return getattr(torch, v)  # type: ignore[no-any-return]
            except AttributeError as exc:
                raise ValueError(f"Invalid torch dtype: {v}") from exc


class Composer(BaseModel):
    """Configurations composed."""

    shared: Shared = Field(default_factory=Shared)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        use_enum_values = True
