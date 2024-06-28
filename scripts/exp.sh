#!/bin/bash

# Define the path to save the logs
LOG_DIR="./artifacts"

mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
FOLD=0

nohup sh -c "export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=0 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=train \
    shared.num_labels=1 \
    shared.resample_strategy=StratifiedKFold \
    shared.resample_params.n_splits=5 \
    shared.resample_params.shuffle=true \
    shared.resample_params.random_state=42 \
    shared.fold=$FOLD \
    shared.padding_side=right \
    shared.max_length=1536 \
    shared.add_special_tokens=True \
    shared.padding=False \
    shared.truncation=True \
    shared.output_hidden_states=False \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-small \
    shared.target_artifacts_dir=/mnt/data/jundazhu/artifacts/exp-$TIMESTAMP  \
    shared.verbose=False \
    shared.adam_epsilon=1e-8 \
    shared.data_seed=null \
    shared.eval_strategy=epoch \
    shared.greater_is_better=True \
    shared.learning_rate=3e-6 \
    shared.load_best_model_at_end=True \
    shared.logging_first_step=True \
    shared.lr_scheduler_type=linear \
    shared.max_grad_norm=1.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=2 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.save_strategy=epoch \
    shared.save_total_limit=2 \
    shared.warmup_ratio=0 \
    shared.weight_decay=0.01 \
    shared.desired_effective_batch_size=2 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.reinitialize_n_layers_of_backbone=0 \
    shared.pooler_type=null" > /mnt/data/jundazhu/artifacts/exp-$TIMESTAMP/nohup_chris_$fold.log 2>&1 &