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
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=attention


python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.task=SINGLE_LABEL_CLASSIFICATION \
    shared.job_type=debug \
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
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=reg_cls_loss \
    shared.pooler_type=null