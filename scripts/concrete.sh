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

# f2_output_v20240624204630
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

# f2_output_v20240624205858
# reg-skf-context=2048-model=small-lr=1e05-cosine-warmup=0-grad_norm=10-epochs=4-bs=8-optim=adamw-criterion=mse-pooler=mean
nohup sh -c 'export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=4 && \
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
    shared.target_artifacts_dir=/mnt/data/jundazhu/artifacts \
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
    shared.pooler_type=mean' > ./artifacts/nohup2048.log 2>&1 &


# reg-skf-context=1024-model=small-lr=1e05-cosine-warmup=0-grad_norm=10-epochs=4-bs=8-optim=adamw-criterion=mse-pooler=gem
nohup sh -c 'export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=3 && \
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
    shared.target_artifacts_dir=/mnt/data/jundazhu/artifacts  \
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
    shared.pooler_config.gem_eps=1e-6' > ./artifacts/nohupgem.log 2>&1 &


# v20240624231703
nohup sh -c 'export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=3 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=train \
    shared.num_labels=1 \
    shared.resample_strategy=StratifiedKFold \
    shared.resample_params.n_splits=5 \
    shared.resample_params.shuffle=true \
    shared.resample_params.random_state=42 \
    shared.fold=3 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=False \
    shared.truncation=True \
    shared.output_hidden_states=False \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-small \
    shared.target_artifacts_dir=/mnt/data/jundazhu/artifacts  \
    shared.verbose=False \
    shared.adam_epsilon=1e-8 \
    shared.data_seed=null \
    shared.eval_strategy=epoch \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.load_best_model_at_end=True \
    shared.logging_first_step=True \
    shared.lr_scheduler_type=linear \
    shared.max_grad_norm=1.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.save_strategy=epoch \
    shared.save_total_limit=1 \
    shared.warmup_ratio=0 \
    shared.weight_decay=0.01 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=null' > ./artifacts/nohup_chris_4.log 2>&1 &

# v20240625063058 + fullfit
nohup sh -c 'export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=3 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=fullfit \
    shared.num_labels=1 \
    shared.resample_strategy=StratifiedKFold \
    shared.resample_params.n_splits=5 \
    shared.resample_params.shuffle=true \
    shared.resample_params.random_state=42 \
    shared.fold=3 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=False \
    shared.truncation=True \
    shared.output_hidden_states=False \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-small \
    shared.target_artifacts_dir=/mnt/data/jundazhu/artifacts  \
    shared.verbose=False \
    shared.adam_epsilon=1e-8 \
    shared.data_seed=null \
    shared.eval_strategy=epoch \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.load_best_model_at_end=True \
    shared.logging_first_step=True \
    shared.lr_scheduler_type=linear \
    shared.max_grad_norm=1.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.save_strategy=epoch \
    shared.save_total_limit=1 \
    shared.warmup_ratio=0 \
    shared.weight_decay=0.01 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=null' > ./artifacts/nohup_chris_4.log 2>&1 &


# attn
nohup sh -c 'export ALLOW_WANDB=true && \
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
    shared.fold=0 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=False \
    shared.truncation=True \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-small \
    shared.target_artifacts_dir=/mnt/data/jundazhu/artifacts/1  \
    shared.verbose=False \
    shared.adam_epsilon=1e-8 \
    shared.data_seed=null \
    shared.eval_strategy=epoch \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.load_best_model_at_end=True \
    shared.logging_first_step=True \
    shared.lr_scheduler_type=linear \
    shared.max_grad_norm=1.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.save_strategy=epoch \
    shared.save_total_limit=1 \
    shared.warmup_ratio=0 \
    shared.weight_decay=0.01 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=attention' > ./artifacts/nohup_chris_4.log 2>&1 &

export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=6 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=train \
    shared.num_labels=1 \
    shared.resample_strategy=StratifiedKFold \
    shared.resample_params.n_splits=5 \
    shared.resample_params.shuffle=true \
    shared.resample_params.random_state=42 \
    shared.fold=1 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=False \
    shared.truncation=True \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-small \
    shared.target_artifacts_dir=/mnt/data/jundazhu/artifacts/exp10-ordinal \
    shared.verbose=False \
    shared.adam_epsilon=1e-8 \
    shared.data_seed=null \
    shared.eval_strategy=epoch \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.load_best_model_at_end=True \
    shared.logging_first_step=True \
    shared.lr_scheduler_type=linear \
    shared.max_grad_norm=1.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=8 \
    shared.report_to=wandb \
    shared.save_strategy=epoch \
    shared.save_total_limit=1 \
    shared.warmup_ratio=0 \
    shared.weight_decay=0.01 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.reinitialize_n_layers_of_backbone=0 \
    shared.default=False \
    shared.criterion=ordinal_reg_loss \
    shared.criterion_config.num_classes=6 \
    shared.criterion_config.init_cutpoints=ordered \
    shared.pooler_type=null


export ALLOW_WANDB=true && \
export CUDA_VISIBLE_DEVICES=7 && \
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=REGRESSION \
    shared.job_type=train \
    shared.num_labels=1 \
    shared.resample_strategy=StratifiedKFold \
    shared.resample_params.n_splits=5 \
    shared.resample_params.shuffle=true \
    shared.resample_params.random_state=42 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=False \
    shared.truncation=True \
    shared.output_hidden_states=False \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/home/jundazhu/models/deberta-v3-small \
    shared.target_artifacts_dir=/mnt/data/jundazhu/artifacts/exp-multi-head  \
    shared.verbose=False \
    shared.adam_epsilon=1e-6 \
    shared.data_seed=null \
    shared.eval_strategy=epoch \
    shared.greater_is_better=True \
    shared.learning_rate=1e-5 \
    shared.load_best_model_at_end=True \
    shared.logging_first_step=True \
    shared.lr_scheduler_type=linear \
    shared.max_grad_norm=1.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=8 \
    shared.per_device_eval_batch_size=16 \
    shared.report_to=wandb \
    shared.save_strategy=epoch \
    shared.save_total_limit=2 \
    shared.warmup_ratio=0 \
    shared.weight_decay=0.01 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.reinitialize_n_layers_of_backbone=0 \
    shared.pooler_type=null \
    shared.very_custom_optimizer_group=False \
    shared.layer_wise_learning_rate_decay=null \
    shared.cls_type=topic_id_cls
