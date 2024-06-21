# pylint: disable=all

from __future__ import annotations

import logging
import types
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed
from omegaconf import OmegaConf as om
from omnivault.distributed.core import get_world_size
from omnivault.utils.config_management.omegaconf import load_yaml_config, merge_configs
from omnivault.utils.reproducibility.seed import seed_all
from omnivault.utils.torch_utils.hf_utils import maybe_resize_token_embeddings
from omnivault.utils.torch_utils.model_utils import get_named_modules, total_parameters, total_trainable_parameters
from omnivault.utils.train_utils.resampling import create_folds
from peft import LoraConfig, get_peft_model
from rich.pretty import pprint
from sklearn.metrics import cohen_kappa_score
from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import ModelOutput
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

import wandb
from .conf.config import A10_24_GPU, ALLOW_WANDB, IMAGE, VOLUME, Composer, Constants, Shared, app, IN_MODAL, A100_40_GPU
from .src.callbacks import SaveLoraHeadCallback
from .src.dataset import load_data
from .src.logger import get_logger
from .src.metrics import compute_metrics_for_classification, compute_metrics_for_regression
from .src.models import DebertaV2WithAttentionPooler
from .src.patches import deberta_v2_seq_cls_forward
from .src.preprocessing import add_prompt_name_group, create_dataset, preprocess, process_labels
from .src.state import State, Statistics
from .src.utils import load_model

logger = get_logger(__name__, level=logging.DEBUG)


def setup_wandb(**kwargs: Any) -> Run | RunDisabled | None:
    run = wandb.init(**kwargs)
    return run


class ImmutableProxy:
    """Immutable proxy object for wrapping mutable objects.

    Example
    -------
    >>> from omnivault.utils.general import ImmutableProxy
    >>> obj = ImmutableProxy([1, 2, 3])
    >>> obj.append(4)
    Traceback (most recent call last):
        ...
        AttributeError: Attempting to set attribute append with <built-in method append of list object at xxx>. list object is immutable.
    """

    def __init__(self, obj: object) -> None:
        object.__setattr__(self, "_obj", obj)

    def __getattr__(self, item: str) -> Any:
        attr = getattr(self._obj, item)
        if callable(attr):

            def immutable_method(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
                raise AttributeError(
                    f"Attempting to modify object with method `{item}`. "
                    f"`{self._obj.__class__.__name__}` object is immutable."
                )

            return immutable_method
        return attr

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(
            f"Attempting to set attribute `{key}` with `{value}`. "
            f"`{self._obj.__class__.__name__}` object is immutable."
        ) from None


@app.function(
    image=IMAGE,
    gpu=A100_40_GPU,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    volumes={Constants.TARGET_ARTIFACTS_DIR: VOLUME},
    _allow_background_volume_commits=True,  # docs say is best to set to True if don't use volume.commit(), see https://modal.com/docs/guide/volumes#huggingface-transformers
)
def main(composer: Composer, state: State) -> None:
    IS_DEBUG = composer.shared.job_type == "debug"  # redundant call but needed for modal
    # NOTE: seed all
    seed = seed_all(
        composer.shared.seed,
        composer.shared.seed_torch,
        composer.shared.set_torch_deterministic,
    )
    logger.info("Seed set to %s", seed)

    # NOTE: update composer - mutating
    if composer.shared.name is None:
        composer.shared.name = (
            composer.shared.job_type
            + "-"
            + composer.shared.task
            + "-"
            + composer.shared.project
            + "-"
            + "v"
            + str(state.timestamp)
            + "-"
            + "git-"
            + state.git_commit_hash
        )

    if composer.shared.tags is None:
        composer.shared.tags = [composer.shared.task]

    if ALLOW_WANDB and not IS_DEBUG:
        logger.info("Setting up wandb.")
        run = setup_wandb(
            project=composer.shared.project,
            entity=composer.shared.entity,
            name=composer.shared.name,
            job_type=composer.shared.job_type,
            tags=composer.shared.tags,
            group=composer.shared.group,
            notes=composer.shared.notes,
            mode=composer.shared.mode,
        )

    # NOTE: loading data stuff
    df = pd.read_csv(composer.shared.train_filepath)
    logger.debug("DataFrame columns: %s, Shape: %s", df.columns.tolist(), df.shape)

    df = process_labels(df, task=composer.shared.task)

    if composer.shared.group_by:
        df = add_prompt_name_group(
            df,
            pd.read_csv("learning_agency_lab_automated_essay_scoring_2/data/predicted_prompt.csv"),
        )
    if composer.shared.external_data_filepath:
        external_df = pd.read_csv(composer.shared.external_data_filepath)
        external_df = process_labels(external_df, task=composer.shared.task, target_column="holistic_essay_score")
        logger.debug(
            "External DataFrame columns: %s, Shape: %s",
            external_df.columns.tolist(),
            external_df.shape,
        )
    # logger.debug("DataFrame Head:\n%s", df.head().to_string(index=False))

    df = create_folds(
        df,
        resample_strategy=composer.shared.resample_strategy,
        resample_params=composer.shared.resample_params,
        group_by=composer.shared.group_by,
        stratify_by=composer.shared.stratify_by,
        fold_column=composer.shared.fold_column,
    )
    pprint(df.groupby(["fold", "label"]).size())

    # load data based on job type, whether to use external data, or pretrain etc
    train_df, valid_df = load_data(
        df_or_filepath=df,
        job_type=composer.shared.job_type,
        fold=composer.shared.fold,
        debug_samples=composer.shared.debug_samples,
        external_data_df_or_filepath=external_df,
    )
    logger.info(
        "Train DataFrame shape: %s, Valid DataFrame shape: %s",
        train_df.shape,
        valid_df.shape,
    )
    pprint(train_df.head())
    pprint(valid_df.head())

    # NOTE: tokenizer shenanigans
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=composer.shared.pretrained_model_name_or_path,
        cache_dir=composer.shared.cache_dir,
        padding_side=composer.shared.padding_side,
    )
    logger.info("Base Tokenizer vocab size: %s", len(tokenizer))
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a `pad_token`. Setting it to `eos_token`!?")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # pprint(tokenizer.init_kwargs)

    if composer.shared.task in ["CLASSIFICATION", "REGRESSION"]:
        tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])

        # test tokenizer on some text
        sample_text = "This is a test.\n\nThis is a new paragraph.  This sentence has double spaces."
        sample_text_tokenized = tokenizer.tokenize(sample_text)
        assert "\n" in sample_text_tokenized
        assert " " * 2 in sample_text_tokenized
        pprint(sample_text_tokenized)

    tokenized_train_dataset = create_dataset(
        df=train_df,
        tokenizer=tokenizer,
        composer=composer,
    )
    tokenized_valid_dataset = create_dataset(
        df=valid_df,
        tokenizer=tokenizer,
        composer=composer,
    )

    # collator for trainer
    # TODO: DataCollatorWithPadding which collator should I use?
    if composer.shared.task in ["CLASSIFICATION", "REGRESSION"]:
        data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding="longest")

    base_model_config: PretrainedConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=composer.shared.pretrained_model_name_or_path,
        force_download=composer.shared.force_download,
        cache_dir=composer.shared.cache_dir,
        torch_dtype=composer.shared.torch_dtype,
        device_map=composer.shared.device_map,
        load_in_4bit=composer.shared.load_in_4bit,
        output_hidden_states=composer.shared.output_hidden_states,
        output_attentions=composer.shared.output_attentions,
    )

    if composer.shared.task == "CLASSIFICATION":
        base_model_config.problem_type = "single_label_classification"
        base_model_config.num_labels = composer.shared.num_labels
    elif composer.shared.task == "REGRESSION":
        base_model_config.problem_type = "regression"
        base_model_config.num_labels = composer.shared.num_labels
        base_model_config.attention_dropout = (
            0.0  # see https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260729
        )
        base_model_config.attention_probs_dropout_prob = 0.0
        base_model_config.hidden_dropout_prob = 0.0
        base_model_config.pooler_dropout = 0.0
    elif composer.shared.task == "CAUSAL_LM":
        logger.warning("I AM EMPTY!")
    else:
        raise ValueError(f"Unsupported task type: {composer.shared.task}")

    if base_model_config.pad_token_id is None or base_model_config.pad_token_id != tokenizer.pad_token_id:
        logger.warning("Setting the `base_model_config`'s `pad_token_id` to `tokenizer.pad_token_id`.")
        base_model_config.pad_token_id = (
            tokenizer.pad_token_id
        )  # see https://stackoverflow.com/questions/68084302/assertionerror-cannot-handle-batch-sizes-1-if-no-padding-token-is-defined

    pprint(base_model_config)
    # update this at the end?
    state.base_model_config = base_model_config

    # load base model
    # NOTE: we are using `AutoModel` here instead of `AutoModelForCausalLM`
    # the difference is that `AutoModel` does not have the `lm_head` attribute
    # which is what we want as we are going to replace it with a custom head.
    # So you can think of `AutoModel` to be a "backbone" without the head.
    # see https://discuss.huggingface.co/t/difference-between-automodel-and-automodelforlm/5967
    # base_model = load_model(
    #     pretrained_model_name_or_path=composer.shared.pretrained_model_name_or_path,
    #     load_backbone_only=composer.shared.load_backbone_only,
    #     task=composer.shared.task,
    #     cache_dir=composer.shared.cache_dir,
    #     config=base_model_config,
    # )
    base_model = DebertaV2WithAttentionPooler(config=base_model_config)

    if maybe_resize_token_embeddings(base_model, tokenizer):
        logger.info("Embedding Size Mismatch. Resizing token embeddings.")
        base_model.resize_token_embeddings(len(tokenizer))
        base_model_config.vocab_size = len(tokenizer)  # update

    # base_model.forward = types.MethodType(deberta_v2_seq_cls_forward, base_model)
    # base_model.pooler = AttentionPooling(
    #     num_hidden_layers=base_model.config.num_hidden_layers,
    #     hidden_size=base_model.config.hidden_size,
    #     pooler_hidden_dim_fc=base_model.config.hidden_size,
    #     pooler_dropout=base_model.config.pooler_dropout,
    # )

    pprint(base_model)
    base_model_named_modules = get_named_modules(base_model)
    # pprint(base_model_named_modules)
    base_model_last_module_name = list(base_model_named_modules[-1].keys())[0]
    if "dropout" in base_model_last_module_name:
        base_model_last_module_name = list(base_model_named_modules[-2].keys())[0]
    potential_target_modules = get_named_modules(base_model)

    if base_model.config.to_dict() != base_model_config.to_dict():
        logger.warning(
            "Model configuration mismatch. Base model config:\n%s\nModel config:\n%s",
            base_model.config.to_dict(),
            base_model_config.to_dict(),
        )

    # NOTE: base model params
    base_model_total_params = total_parameters(base_model)
    base_model_total_trainable_params = total_trainable_parameters(base_model)
    logger.info(
        "Base model %s\ntotal trainable parameters: %.2fM\ntotal parameters: %.2fM",
        base_model.__class__.__name__,
        base_model_total_trainable_params / 1_000_000,
        base_model_total_params / 1_000_000,
    )

    # DO DRY RUN
    # pprint(base_model.score.weight[0].detach().cpu().numpy())
    # pprint(base_model.classifier.weight[0][0].detach().cpu().numpy())
    sample_batch = [tokenized_train_dataset[i] for i in range(2)]
    collated_sample_batch = data_collator(sample_batch)  # 2 samples
    pprint(collated_sample_batch["input_ids"].shape)
    pprint(collated_sample_batch["labels"])
    outputs: ModelOutput = base_model(**collated_sample_batch)
    pprint(outputs.keys())
    pprint(outputs["logits"].shape)
    pprint(outputs["logits"][0][0].detach().cpu().numpy())

    # FIXME: not working
    # if torch.cuda.is_available() and not IS_DEBUG:
    #     logger.info("Showing model summary.")
    #     logger.warning(
    #         "Be careful as `torchinfo` might MUTATE model init weights, so if you run without `torchinfo` your results from model may differ!"
    #     )
    #     torchinfo.summary(
    #         base_model,
    #         verbose=1,
    #         input_data=collated_sample_batch,
    #         dtypes=list[torch.LongTensor],  # type: ignore[arg-type]
    #         device=base_model.device,
    #     )

    if composer.shared.use_lora:
        logger.info(
            "Our base model's last module is named `%s` consider adding it to `modules_to_save`?",
            base_model_last_module_name,
        )

        adatper_config = LoraConfig(
            task_type=composer.shared.task,
            inference_mode=composer.shared.inference_mode,
            r=composer.shared.r,
            lora_alpha=composer.shared.lora_alpha,
            lora_dropout=composer.shared.lora_dropout,
            target_modules=composer.shared.target_modules,
            bias=composer.shared.bias,
            modules_to_save=composer.shared.modules_to_save,
        )

        try:
            base_model_with_adapter = get_peft_model(model=base_model, peft_config=adatper_config)
            (
                base_model_with_adapter_total_trainable_params,
                base_model_with_adapter_total_params,
            ) = base_model_with_adapter.get_nb_trainable_parameters()
            base_model_with_adapter.print_trainable_parameters()
        except ValueError as exc:
            logger.exception(msg="Error creating LoraConfig, check `target_modules` and `modules_to_save`.")
            logger.info("Base model target modules: %s", potential_target_modules)
            raise exc from None

    # TOOD: DO PROFILE?
    total_train_samples = len(tokenized_train_dataset)
    total_valid_samples = len(tokenized_valid_dataset) if tokenized_valid_dataset else 0
    local_world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    world_size = get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1

    # need to set the gradient accumulation steps here before the attribute is used in `effective_train_batch_size`
    composer.shared.gradient_accumulation_steps = (
        composer.shared.desired_effective_batch_size // composer.shared.per_device_train_batch_size
    )

    effective_train_batch_size = (
        composer.shared.per_device_train_batch_size
        * composer.shared.gradient_accumulation_steps
        * local_world_size
        * world_size
    )
    total_train_steps_per_epoch = total_train_samples // effective_train_batch_size
    total_train_steps = total_train_steps_per_epoch * composer.shared.num_train_epochs

    if IS_DEBUG:
        composer.shared.logging_steps = total_train_steps // 2
        composer.shared.eval_steps = total_train_steps // 2
        composer.shared.save_steps = composer.shared.eval_steps * 2
        composer.shared.per_device_train_batch_size = 2
        composer.shared.per_device_eval_batch_size = 2
        composer.shared.num_train_epochs = 1
    else:
        composer.shared.logging_steps = composer.shared.logging_steps or (total_train_steps // 8)
        composer.shared.eval_steps = composer.shared.eval_steps or (total_train_steps // 32)
        composer.shared.save_steps = composer.shared.save_steps or composer.shared.eval_steps

    composer.shared.output_dir = composer.shared.output_dir or str(
        Path(composer.shared.target_artifacts_dir) / f"output_v{state.timestamp}"
    )
    composer.shared.logging_dir = composer.shared.logging_dir or str(
        Path(composer.shared.target_artifacts_dir) / composer.shared.name / f"logs_v{state.timestamp}"
    )

    if composer.shared.enable_mixed_precision:
        composer.shared.fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
        composer.shared.bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    composer.shared.learning_rate = (
        composer.shared.learning_rate
        or (  # NOTE: if lr is not set explicitly then use some loose heuristic
            composer.shared.base_learning_rate
            * composer.shared.gradient_accumulation_steps
            * composer.shared.per_device_train_batch_size
        )
    )

    statistics = Statistics(
        total_train_samples=total_train_samples,
        total_valid_samples=total_valid_samples,
        effective_train_batch_size=effective_train_batch_size,
        total_train_steps_per_epoch=total_train_steps_per_epoch,
        total_train_steps=total_train_steps,
        base_model_total_params=base_model_total_params,
        base_model_total_trainable_params=base_model_total_trainable_params,
        base_model_with_adapter_total_params=None
        if not composer.shared.use_lora
        else base_model_with_adapter_total_params,
        base_model_with_adapter_total_trainable_params=None
        if not composer.shared.use_lora
        else base_model_with_adapter_total_trainable_params,
    )
    pprint(statistics)

    training_args = TrainingArguments(
        adam_beta1=composer.shared.adam_beta1,
        adam_beta2=composer.shared.adam_beta2,
        adam_epsilon=composer.shared.adam_epsilon,
        bf16=composer.shared.bf16,
        data_seed=composer.shared.data_seed,
        disable_tqdm=composer.shared.disable_tqdm,
        do_eval=composer.shared.do_eval,
        do_predict=composer.shared.do_predict,
        do_train=composer.shared.do_train,
        eval_steps=composer.shared.eval_steps,
        eval_strategy=composer.shared.eval_strategy,
        fp16=composer.shared.fp16,
        full_determinism=composer.shared.full_determinism,
        gradient_accumulation_steps=composer.shared.gradient_accumulation_steps,
        gradient_checkpointing=composer.shared.gradient_checkpointing,
        greater_is_better=composer.shared.greater_is_better,
        include_tokens_per_second=composer.shared.include_tokens_per_second,
        learning_rate=composer.shared.learning_rate,
        load_best_model_at_end=composer.shared.load_best_model_at_end,
        logging_dir=composer.shared.logging_dir,
        logging_steps=composer.shared.logging_steps,
        lr_scheduler_type=composer.shared.lr_scheduler_type,
        max_grad_norm=composer.shared.max_grad_norm,
        metric_for_best_model=composer.shared.metric_for_best_model,
        num_train_epochs=composer.shared.num_train_epochs,
        optim=composer.shared.optim,
        output_dir=composer.shared.output_dir,
        overwrite_output_dir=composer.shared.overwrite_output_dir,
        per_device_eval_batch_size=composer.shared.per_device_eval_batch_size,
        per_device_train_batch_size=composer.shared.per_device_train_batch_size,
        report_to=composer.shared.report_to,
        run_name=composer.shared.name,
        save_steps=composer.shared.save_steps,
        save_strategy=composer.shared.save_strategy,
        save_total_limit=composer.shared.save_total_limit,
        seed=composer.shared.seed,
        warmup_ratio=composer.shared.warmup_ratio,
        weight_decay=composer.shared.weight_decay,
    )
    pprint(training_args)

    if composer.shared.task == "CLASSIFICATION":
        compute_metrics = compute_metrics_for_classification
    elif composer.shared.task == "REGRESSION":
        compute_metrics = compute_metrics_for_regression
    else:
        compute_metrics = None

    model = base_model_with_adapter if composer.shared.use_lora else base_model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        compute_metrics=compute_metrics,
    )

    if composer.shared.do_train:
        if (
            composer.shared.use_lora
            and "Mistral" in composer.shared.pretrained_model_name_or_path
            and composer.shared.task in ["CLASSIFICATION", "REGRESSION"]
        ):
            trainer.add_callback(SaveLoraHeadCallback(model))
        if composer.shared.resume_from_checkpoint:
            logger.warning(
                "Resuming training from checkpoint. Ensure your `num_train_epochs` is greater than your resumed checkpoint's `num_train_epochs`."
            )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)  # if None is no ops
        torch.cuda.empty_cache()
        trainer.save_model(output_dir=composer.shared.output_dir)
        trainer.save_state()
        tokenizer.save_pretrained(composer.shared.output_dir)
        model.save_pretrained(composer.shared.output_dir)
        # if hasattr(trainer.model, "base_model"):
        #     logger.info("Saving base model.")
        #     pprint(trainer.model.base_model.model)
        #     trainer.model.base_model.model.save_pretrained(
        #         f"{str(Constants.TARGET_ARTIFACTS_DIR)}/output_v{VER}/base_model"
        #     )

    if ALLOW_WANDB and not IS_DEBUG:
        run.config.update(composer.model_dump())
        run.finish()

    if composer.shared.do_eval:
        y_true = valid_df["score"].values

        if composer.shared.task == "REGRESSION":
            logits = trainer.predict(tokenized_valid_dataset).predictions
            predictions = logits.round(0) + 1
            valid_df["logits"] = logits + 1
            qwk = cohen_kappa_score(
                valid_df.score.values,
                valid_df.logits.values.clip(1, 6).round(0),
                weights="quadratic",
            )
        elif composer.shared.task == "CLASSIFICATION":
            logits = trainer.predict(tokenized_valid_dataset).predictions
            predictions = logits.argmax(axis=1) + 1
            columns = [f"p{x}" for x in range(composer.shared.num_labels)]
            valid_df[columns] = logits
            qwk = cohen_kappa_score(
                valid_df.score.values,
                valid_df.iloc[:, -6:].values.argmax(axis=1) + 1,
                weights="quadratic",
            )

        elif composer.shared.task == "CAUSAL_LM":
            preds = []
            # NOTE: inference so change composer config, dont do this in prod
            composer.shared.inference = True
            composer.shared.return_tensors = "pt"
            model.eval()
            for i, row in valid_df.iterrows():
                if i % 100 == 0:
                    print(row)
                    print(i, ", ", end="")

                tokenized_sample = preprocess(
                    row,
                    tokenizer=tokenizer,
                    task=composer.shared.task,
                    inference=composer.shared.inference,
                    system_prompt=composer.shared.system_prompt,
                    return_tokenized_text=composer.shared.return_tokenized_text,
                    max_length=composer.shared.max_length,
                    truncation=composer.shared.truncation,
                    padding=composer.shared.padding,
                    return_tensors=composer.shared.return_tensors,
                    add_special_tokens=composer.shared.add_special_tokens,
                )

                tokenized_sample = {k: v.to(model.device) for k, v in tokenized_sample.items()}
                generated_ids = model.generate(
                    **tokenized_sample,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                try:
                    answer = decoded[0].rsplit("The score is: ", 1)[1]
                    preds.append(int(answer))
                except:
                    print(f"Error at index {i}")
                    preds.append(3)

                if i == 7:
                    print(f"preds[:8]={preds}, ", end="")

            qwk = cohen_kappa_score(valid_df.score.values, preds, weights="quadratic")
            print(f"Validation QWK Score = {qwk}")

    valid_df.to_csv(
        f"{str(composer.shared.output_dir)}/valid_df_fold_{composer.shared.fold}.csv",
        index=False,
    )
    print(f"Validation QWK Score = {qwk}")

    pprint(composer)
    import json

    with open(f"{str(composer.shared.output_dir)}/composer.json", "w") as f:
        f.write(json.dumps(composer.model_dump_json(exclude="shared.torch_dtype"), indent=4))


@app.local_entrypoint()
def entrypoint(yaml_path: str) -> None:
    yaml_cfg = load_yaml_config(yaml_path)
    cfg = merge_configs(yaml_cfg, [])
    # cfg = merge_configs(yaml_cfg, args_list)
    om.resolve(cfg)

    composer = Composer(shared=Shared(**cfg.shared))
    state = State()
    pprint(composer)
    pprint(state)
    base_composer = ImmutableProxy(composer.model_copy(update=None, deep=True))
    base_state = ImmutableProxy(state.model_copy(update=None, deep=True))

    IS_DEBUG = composer.shared.job_type == "debug"

    if IS_DEBUG:
        composer.shared.set_torch_deterministic = True
        composer.shared.max_length = 64
        composer.shared.cache_dir = "./.cache/huggingface"
        composer.shared.target_artifacts_dir = "./artifacts"


    # main(composer, state)
    main.remote(composer, state)

# if not IN_MODAL:
#     entrypoint("lal/conf/deberta_reg.yaml")

# if __name__ == "__main__":
#     yaml_path = sys.argv[1]
#     args_list = sys.argv[2:]

#     yaml_cfg = load_yaml_config(yaml_path)
#     cfg = merge_configs(yaml_cfg, args_list)
#     om.resolve(cfg)  # inplace ops

#     composer = Composer(shared=Shared(**cfg.shared))
#     state = State()
#     pprint(composer)
#     pprint(state)
#     # NOTE: base composer is basically an immutable copy of composer where the
#     # base configurations provided by user are stored. Why this? Cause in ml,
#     # it is often the case of modifying the configurations midway and we need to keep
#     # track of the original configurations.
#     base_composer = ImmutableProxy(composer.model_copy(update=None, deep=True))
#     base_state = ImmutableProxy(state.model_copy(update=None, deep=True))

#     IS_DEBUG = composer.shared.job_type == "debug"

#     if IS_DEBUG:
#         composer.shared.set_torch_deterministic = True
#         composer.shared.max_length = 64

#     main(composer, state)

"""
# DEBERTA DEBUG

python -m learning_agency_lab_automated_essay_scoring_2.entrypoint_w_hf_trainer \
learning_agency_lab_automated_essay_scoring_2/config.yaml \
task=CLASSIFICATION \
shared.job_type=debug \
shared.train_filepath=learning_agency_lab_automated_essay_scoring_2/data/train.csv \
shared.external_data_filepath=learning_agency_lab_automated_essay_scoring_2/data/persuade_2.0_human_scores_demo_id_github.csv \
shared.use_lora=False \
shared.pretrained_model_name_or_path=microsoft/deberta-v3-small \
shared.task_type='SEQ_CLS' \
shared.target_modules='["query_proj", "key_proj", "value_proj"]' \
shared.modules_to_save='["classifier"]' \
shared.target_artifacts_dir=learning_agency_lab_automated_essay_scoring_2/artifacts \
shared.metric_for_best_model='eval_qwk' \
shared.greater_is_better=True \
shared.lr_scheduler_type='cosine'

 {'eval_qwk': -0.15267175572519087, 'eval_loss': 1.4653687477111816}
{'train_loss': 1.60413059592247}
Validation QWK Score = -0.08688309251266646

# MISTRAL DEBUG - CAUSAL_LM

python -m learning_agency_lab_automated_essay_scoring_2.entrypoint_w_hf_trainer \
learning_agency_lab_automated_essay_scoring_2/config.yaml \
task=CAUSAL_LM \
shared.job_type=debug \
shared.train_filepath=learning_agency_lab_automated_essay_scoring_2/data/train.csv \
shared.external_data_filepath=learning_agency_lab_automated_essay_scoring_2/data/persuade_2.0_human_scores_demo_id_github.csv \
shared.use_lora=True \
shared.pretrained_model_name_or_path=mistralai/Mistral-7B-Instruct-v0.2 \
shared.task_type='CAUSAL_LM' \
shared.modules_to_save='["score"]' \
shared.padding_side='right' \
shared.max_length=1536 \
shared.torch_dtype='float32' \
shared.cache_dir="/root/.cache/huggingface" \
shared.base_learning_rate=5e-5 \
shared.metric_for_best_model='eval_loss' \
shared.greater_is_better=False \
shared.lr_scheduler_type='cosine' \
shared.warmup_ratio=0.05 \
shared.num_train_epochs=2 \
shared.per_device_train_batch_size=2 \
shared.per_device_eval_batch_size=8

export ALLOW_WANDB=true && \
modal run --detach \
learning_agency_lab_automated_essay_scoring_2.entrypoint_w_hf_trainer \
--yaml-path=./learning_agency_lab_automated_essay_scoring_2/mistral_causal.yaml

export ALLOW_WANDB=true && \
modal run --detach \
lal.entrypoint \
--yaml-path=lal/conf/deberta_reg.yaml

export ALLOW_WANDB=true && \
modal run --detach \
learning_agency_lab_automated_essay_scoring_2.entrypoint \
--yaml-path=./learning_agency_lab_automated_essay_scoring_2/deberta_cls.yaml
"""
# export ALLOW_WANDB=true && modal run --detach learning_agency_lab_automated_essay_scoring_2.train --train-filepath=./learning_agency_lab_automated_essay_scoring_2/data/train.csv
# modal shell learning_agency_lab_automated_essay_scoring_2.chris
# modal volume ls artifacts-volume
# modal volume get artifacts-volume output_v20240616054048/checkpoint-4050 .

"""
array(0.49022794, dtype=float32)

array(-0.12705599, dtype=float32)
Validation QWK Score = -0.06416748545083517
"""
