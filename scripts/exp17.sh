#!/bin/bash


TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
GIT_COMMIT_HASH=$(git rev-parse HEAD)
LOG_DIR="./artifacts/$TIMESTAMP-$GIT_COMMIT_HASH"
ARTIFACTS_DIR="/mnt/data/jundazhu/artifacts/$TIMESTAMP-$GIT_COMMIT_HASH"
PID_FILE="$LOG_DIR/pids.txt"

mkdir -p $LOG_DIR


for FOLD in {0..4}
do
    DEVICE=$(echo "${FOLD}+0" | bc)

    nohup sh -c "export ALLOW_WANDB=true && \
    export CUDA_VISIBLE_DEVICES=$DEVICE && \
    python -m lal.entrypoint_local \
        lal/conf/deberta_reg.yaml \
        shared.task=REGRESSION \
        shared.job_type=train_with_external \
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
        shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-large \
        shared.target_artifacts_dir=$ARTIFACTS_DIR  \
        shared.verbose=False \
        shared.seed=20230310 \
        shared.adam_epsilon=1e-6 \
        shared.data_seed=null \
        shared.eval_strategy=epoch \
        shared.greater_is_better=True \
        shared.learning_rate=1e-5 \
        shared.load_best_model_at_end=True \
        shared.logging_first_step=True \
        shared.lr_scheduler_type=cosine \
        shared.max_grad_norm=1.0 \
        shared.metric_for_best_model=eval_qwk \
        shared.num_train_epochs=5 \
        shared.optim=adamw_torch \
        shared.per_device_train_batch_size=2 \
        shared.per_device_eval_batch_size=16 \
        shared.report_to=wandb \
        shared.save_strategy=epoch \
        shared.save_total_limit=1 \
        shared.warmup_ratio=0.1 \
        shared.weight_decay=0.01 \
        shared.scheduler_specific_kwargs.num_cycles=0.5 \
        shared.desired_effective_batch_size=8 \
        shared.enable_mixed_precision=True \
        shared.model_type=SubclassedDebertaV2ForSequenceClassification \
        shared.criterion=mse \
        shared.reinitialize_n_layers_of_backbone=1 \
        shared.pooler_type=null \
        shared.freeze_embeddings=False \
        shared.very_custom_optimizer_group=False" > $LOG_DIR/exp-fold-$FOLD.log 2>&1 &
    echo $! >> $PID_FILE
done

# waitingggggg for all processes to complete
while IFS= read -r pid; do
    wait $pid
done < "$PID_FILE"


TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
GIT_COMMIT_HASH=$(git rev-parse HEAD)
LOG_DIR="./artifacts/$TIMESTAMP-$GIT_COMMIT_HASH"
ARTIFACTS_DIR="/mnt/data/jundazhu/artifacts/$TIMESTAMP-$GIT_COMMIT_HASH"
PID_FILE="$LOG_DIR/pids.txt"

mkdir -p $LOG_DIR


for FOLD in {0..4}
do
    DEVICE=$(echo "${FOLD}+0" | bc)

    nohup sh -c "export ALLOW_WANDB=true && \
    export CUDA_VISIBLE_DEVICES=$DEVICE && \
    python -m lal.entrypoint_local \
        lal/conf/deberta_reg.yaml \
        shared.task=REGRESSION \
        shared.job_type=train_with_external \
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
        shared.output_hidden_states=True \
        shared.output_attentions=False \
        shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-large \
        shared.target_artifacts_dir=$ARTIFACTS_DIR  \
        shared.verbose=False \
        shared.seed=20230310 \
        shared.adam_epsilon=1e-6 \
        shared.data_seed=null \
        shared.eval_strategy=epoch \
        shared.greater_is_better=True \
        shared.learning_rate=3e-6 \
        shared.load_best_model_at_end=True \
        shared.logging_first_step=True \
        shared.lr_scheduler_type=cosine \
        shared.max_grad_norm=1.0 \
        shared.metric_for_best_model=eval_qwk \
        shared.num_train_epochs=5 \
        shared.optim=adamw_torch \
        shared.per_device_train_batch_size=2 \
        shared.per_device_eval_batch_size=16 \
        shared.report_to=wandb \
        shared.save_strategy=epoch \
        shared.save_total_limit=1 \
        shared.warmup_ratio=0.1 \
        shared.weight_decay=0.01 \
        shared.scheduler_specific_kwargs.num_cycles=0.5 \
        shared.desired_effective_batch_size=4 \
        shared.enable_mixed_precision=True \
        shared.model_type=SubclassedDebertaV2ForSequenceClassification \
        shared.criterion=mse \
        shared.reinitialize_n_layers_of_backbone=1 \
        shared.pooler_type=attention \
        shared.freeze_embeddings=True \
        shared.freeze_these_layers_indices='[0,1]' \
        shared.very_custom_optimizer_group=True \
        shared.layerwise_learning_rate_decay_mulitplier=0.95" > $LOG_DIR/exp-fold-$FOLD.log 2>&1 &
    echo $! >> $PID_FILE
done

# waitingggggg for all processes to complete
while IFS= read -r pid; do
    wait $pid
done < "$PID_FILE"

echo "All training processes have completed."

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
GIT_COMMIT_HASH=$(git rev-parse HEAD)
LOG_DIR="./artifacts/$TIMESTAMP-$GIT_COMMIT_HASH"
ARTIFACTS_DIR="/mnt/data/jundazhu/artifacts/$TIMESTAMP-$GIT_COMMIT_HASH"
PID_FILE="$LOG_DIR/pids.txt"

mkdir -p $LOG_DIR


for FOLD in {0..4}
do
    DEVICE=$(echo "${FOLD}+0" | bc)

    nohup sh -c "export ALLOW_WANDB=true && \
    export CUDA_VISIBLE_DEVICES=$DEVICE && \
    python -m lal.entrypoint_local \
        lal/conf/deberta_reg.yaml \
        shared.task=REGRESSION \
        shared.job_type=train_with_external \
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
        shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-large \
        shared.target_artifacts_dir=$ARTIFACTS_DIR  \
        shared.verbose=False \
        shared.seed=20230310 \
        shared.adam_epsilon=1e-6 \
        shared.data_seed=null \
        shared.eval_strategy=epoch \
        shared.greater_is_better=True \
        shared.learning_rate=3e-6 \
        shared.load_best_model_at_end=True \
        shared.logging_first_step=True \
        shared.lr_scheduler_type=cosine \
        shared.max_grad_norm=10 \
        shared.metric_for_best_model=eval_qwk \
        shared.num_train_epochs=5 \
        shared.optim=adamw_torch \
        shared.per_device_train_batch_size=2 \
        shared.per_device_eval_batch_size=16 \
        shared.report_to=wandb \
        shared.save_strategy=epoch \
        shared.save_total_limit=1 \
        shared.warmup_ratio=0.1 \
        shared.weight_decay=0.01 \
        shared.scheduler_specific_kwargs.num_cycles=0.5 \
        shared.desired_effective_batch_size=4 \
        shared.enable_mixed_precision=True \
        shared.model_type=SubclassedDebertaV2ForSequenceClassification \
        shared.criterion=smooth_l1_with_mse \
        shared.reinitialize_n_layers_of_backbone=1 \
        shared.pooler_type=mean \
        shared.freeze_embeddings=True \
        shared.freeze_these_layers_indices='[0,1]' \
        shared.very_custom_optimizer_group=True \
        shared.layerwise_learning_rate_decay_mulitplier=0.9" > $LOG_DIR/exp-fold-$FOLD.log 2>&1 &
    echo $! >> $PID_FILE
done