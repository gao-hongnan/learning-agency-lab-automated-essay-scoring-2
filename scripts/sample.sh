FOLD=0
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
ARTIFACTS_DIR="./artifacts/exp1-$TIMESTAMP"
python -m lal.entrypoint_local \
        lal/conf/deberta_reg.yaml \
        shared.task=REGRESSION \
        shared.job_type=debug \
        shared.num_labels=1 \
        shared.resample_strategy=StratifiedKFold \
        shared.resample_params.n_splits=5 \
        shared.resample_params.shuffle=true \
        shared.resample_params.random_state=42 \
        shared.fold=$FOLD \
        shared.padding_side=right \
        shared.max_length=1024 \
        shared.add_special_tokens=True \
        shared.padding=False \
        shared.truncation=True \
        shared.output_hidden_states=False \
        shared.output_attentions=False \
        shared.pretrained_model_name_or_path=microsOft/deberta-v3-small \
        shared.target_artifacts_dir=$ARTIFACTS_DIR  \
        shared.verbose=False \
        shared.seed=42 \
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
        shared.num_train_epochs=5 \
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
        shared.model_type=SubclassedDebertaV2ForSequenceClassificationMultiHead \
        shared.criterion=mse \
        shared.reinitialize_n_layers_of_backbone=0 \
        shared.pooler_type=null \
        shared.very_custom_optimizer_group=False \
        shared.layerwise_learning_rate_decay_mulitplier=null

# INFO: 2024-06-25 08:36:49,293: __main__  Sanity Check Last Layer Weights: -0.004328871
# INFO: 2024-06-25 08:36:49,358: __main__  Collated sample batch keys: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
# INFO: 2024-06-25 08:36:49,358: __main__  Collated sample batch input_ids shape: torch.Size([2, 64])
# INFO: 2024-06-25 08:36:49,358: __main__  Collated sample batch labels shape: torch.Size([2])
# INFO: 2024-06-25 08:36:49,358: __main__  Collated sample batch attention_mask shape: torch.Size([2, 64])
# INFO: 2024-06-25 08:36:49,429: __main__  Dry run status: SUCCESS
# INFO: 2024-06-25 08:36:49,429: __main__  Dry run outputs keys: odict_keys(['loss', 'logits'])
# INFO: 2024-06-25 08:36:49,429: __main__  Dry run outputs logits shape: torch.Size([2, 1])
# INFO: 2024-06-25 08:36:49,429: __main__  Dry run outputs logits[0][0]: 0.79482317
# INFO: 2024-06-25 08:36:49,429: __main__  Dry run outputs loss: 1.301110029220581
# {'loss': 0.9527, 'grad_norm': 42.16598892211914, 'learning_rate': 5e-06, 'epoch': 0.5}
# {'eval_qwk': 0.06788637457871927, 'eval_loss': 1.039064884185791, 'eval_runtime': 2.4788, 'eval_samples_per_second': 51.638, 'eval_steps_per_second': 25.819, 'epoch': 0.5}
# {'loss': 1.0067, 'grad_norm': 13.904077529907227, 'learning_rate': 0.0, 'epoch': 1.0}
# {'eval_qwk': 0.09054857898215463, 'eval_loss': 1.0542982816696167, 'eval_runtime': 1.758, 'eval_samples_per_second': 72.808, 'eval_steps_per_second': 36.404, 'epoch': 1.0}
# {'train_runtime': 30.5798, 'train_samples_per_second': 4.186, 'train_steps_per_second': 2.093, 'train_loss': 0.9796938598155975, 'epoch': 1.0}
# Validation QWK Score = 0.09054857898215463


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
    shared.report_to=none \
    shared.warmup_ratio=0 \
    shared.desired_effective_batch_size=8 \
    shared.enable_mixed_precision=True \
    shared.model_type=SubclassedDebertaV2ForSequenceClassification \
    shared.criterion=mse \
    shared.pooler_type=mean \
    shared.reinitialize_n_layers_of_backbone=0 \
    shared.freeze_these_layers_indices='[]' \
    shared.freeze_embeddings=False

# Validation QWK Score = 0.1040995444950561