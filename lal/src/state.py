# pylint: disable=all

from datetime import datetime
from typing import Any

from omnivault.utils.reproducibility.git_utils import get_git_commit_hash
from pydantic import BaseModel, Field


class Statistics(BaseModel):
    class_statistics: Any = None

    len_train_dataloader: int

    total_train_samples: int
    total_valid_samples: int | None = None
    total_test_samples: int | None = None

    effective_train_batch_size: int

    total_train_steps_per_epoch: int
    total_valid_steps_per_epoch: int | None = None
    total_train_steps: int
    total_valid_steps: int | None = None

    base_model_total_params: float
    base_model_total_trainable_params: float
    base_model_with_adapter_total_params: float | None = None
    base_model_with_adapter_total_trainable_params: float | None = None


class State(BaseModel):
    git_commit_hash: str = Field(
        default_factory=lambda: get_git_commit_hash(working_dir=".", are_there_untracked_or_uncommitted=False)
    )
    # git_commit_hash: str = "111"
    timestamp: str = Field(default_factory=lambda: str(datetime.now().strftime("%Y%m%d%H%M%S")))

    base_model_config: Any = None  # this is the model config from hf
    hf_training_args: Any = None  # this is the training args from hf

    hf_tokenizer_kwargs: Any = None  # this is the tokenizer from hf
    statistics: Statistics = Field(default_factory=dict)
