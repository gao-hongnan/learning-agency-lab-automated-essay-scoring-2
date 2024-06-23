# BELOW TWO YIELDS SAME RESULTS 59347e4389ac9dfc98013131ebc5d8f7e2d5c3c2
# INFO: 2024-06-23 20:08:17,547: __main__  Sanity Check Last Layer Weights: -0.0143431295
# INFO: 2024-06-23 20:08:17,635: __main__  Collated sample batch keys: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
# INFO: 2024-06-23 20:08:17,635: __main__  Collated sample batch input_ids shape: torch.Size([2, 64])
# INFO: 2024-06-23 20:08:17,635: __main__  Collated sample batch labels shape: torch.Size([2])
# INFO: 2024-06-23 20:08:17,635: __main__  Collated sample batch attention_mask shape: torch.Size([2, 64])
# INFO: 2024-06-23 20:08:17,775: __main__  Dry run status: SUCCESS
# INFO: 2024-06-23 20:08:17,776: __main__  Dry run outputs keys: odict_keys(['loss', 'logits'])
# INFO: 2024-06-23 20:08:17,776: __main__  Dry run outputs logits shape: torch.Size([2, 1])
# INFO: 2024-06-23 20:08:17,776: __main__  Dry run outputs logits[0][0]: -0.27593312
# INFO: 2024-06-23 22:01:04,191: __main__  Dry run outputs loss: 5.179263114929199

python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.taks=REGRESSION \
    shared.job_type=debug \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=False \
    shared.output_attentions=False \
    shared.pooler_type=null \
    shared.pretrained_model_name_or_path=microsoft/deberta-v3-base \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=3e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=16 \
    shared.per_device_eval_batch_size=16 \
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=16 \
    shared.enable_mixed_precision=True \
    shared.default=True \
    shared.criterion=null \
    shared.pooler_type=null \
    shared.data_seed=null

python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.taks=REGRESSION \
    shared.job_type=debug \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=False \
    shared.output_attentions=False \
    shared.pooler_type=null \
    shared.pretrained_model_name_or_path=microsoft/deberta-v3-base \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=3e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=16 \
    shared.per_device_eval_batch_size=16 \
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=16 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=null \
    shared.data_seed=null

# SAMPLE ATTENTION POOLER
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.taks=REGRESSION \
    shared.job_type=debug \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pooler_type=null \
    shared.pretrained_model_name_or_path=microsoft/deberta-v3-base \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=3e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=16 \
    shared.per_device_eval_batch_size=16 \
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=16 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=attention

# HUBER

python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.taks=REGRESSION \
    shared.job_type=debug \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pooler_type=null \
    shared.pretrained_model_name_or_path=microsoft/deberta-v3-base \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=3e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=16 \
    shared.per_device_eval_batch_size=16 \
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=16 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=huber \
    shared.pooler_type=null

# AND StratifiedGroupKFold
python -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.taks=REGRESSION \
    shared.job_type=debug \
    shared.resample_strategy=StratifiedGroupKFold \
    shared.resample_params.n_splits=7 \
    shared.resample_params.shuffle=true \
    shared.resample_params.random_state=20230310 \
    random_state: 20230310 \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=True \
    shared.output_attentions=False \
    shared.pooler_type=null \
    shared.pretrained_model_name_or_path=microsoft/deberta-v3-base \
    shared.target_artifacts_dir=./artifacts \
    shared.verbose=False \
    shared.greater_is_better=True \
    shared.learning_rate=3e-5 \
    shared.lr_scheduler_type=cosine \
    shared.max_grad_norm=10.0 \
    shared.metric_for_best_model=eval_qwk \
    shared.num_train_epochs=4 \
    shared.optim=adamw_torch \
    shared.per_device_train_batch_size=16 \
    shared.per_device_eval_batch_size=16 \
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=16 \
    shared.enable_mixed_precision=True \
    shared.default=False \
    shared.criterion=huber \
    shared.pooler_type=null