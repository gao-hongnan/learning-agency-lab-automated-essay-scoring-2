python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=debug \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-large \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=attention



# HUBER
export CUDA_VISIBLE_DEVICES=5,6 && \
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.ddp_find_unused_parameters=False \
    shared.task=REGRESSION \
    shared.job_type=train \
    shared.resample_strategy=StratifiedGroupKFold \
    shared.resample_params.n_splits=7 \
    shared.resample_params.shuffle=true \
    shared.resample_params.random_state=20230310 \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-large \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=4 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=huber \
    shared.pooler_type=attention


# cls-f2_output_v20240624191419-reg_cls_loss-sgkf
export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=7 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=SINGLE_LABEL_CLASSIFICATION \
    shared.job_type=train \
    shared.resample_strategy=StratifiedGroupKFold \
    shared.resample_params.n_splits=7 \
    shared.resample_params.shuffle=true \
    shared.resample_params.random_state=20230310 \
    shared.num_labels=6 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-base \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=reg_cls_loss \
    shared.criterion_config.alpha=0.8 \
    shared.pooler_type=attention

# output_v20240624203109
# reg-skf-context=1024-model=small-lr=1e05-cosine-warmup=0-grad_norm=10-epochs=4-bs=8-optim=adamw-criterion=mse-pooler=mean
nohup sh -c 'export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=7 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=train \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-small \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=mean' > ./artifacts/nohup1.log 2>&1 &

# output_v20240624203640
# reg-skf-context=2048-model=small-lr=1e05-cosine-warmup=0-grad_norm=10-epochs=4-bs=8-optim=adamw-criterion=mse-pooler=mean
nohup sh -c 'export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=5 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=train \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=2048 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-small \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=mean' > ./artifacts/nohup.log 2>&1 &


# reg-skf-context=1024-model=small-lr=1e05-cosine-warmup=0-grad_norm=10-epochs=4-bs=8-optim=adamw-criterion=mse-pooler=mean
nohup sh -c 'export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=5 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=debug \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=2048 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=microsoft/deberta-v3-small \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=gem \
    shared.pooler_config.gem_p=3 \
    shared.pooler_config.gem_eps=1e-6' > ./artifacts/nohup.log 2>&1 &
