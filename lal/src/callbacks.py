from __future__ import annotations

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .logger import get_logger

logger = get_logger(__name__)


class SaveLoraHeadCallback(TrainerCallback):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        fname = f"{args.output_dir}/checkpoint-{state.global_step}/score.original_module.pt"
        # torch.save(model.model.score.original_module.state_dict(), fname)
        torch.save(kwargs["model"].score.original_module.state_dict(), fname)
