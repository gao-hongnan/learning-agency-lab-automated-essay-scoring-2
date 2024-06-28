# Reproducibility

Dealing with HF training is not fun, and by and large I would like my own small
customizations and to confirm that my changes are working as expected I need a
way to tell if my changes with some default settings change the score values.

Checking out to below commit and run locally on mps yields the below results.

## 55bda58ab5fbcb7f8421f900c830480ca2c4b97b

```text
INFO: 2024-06-25 08:36:49,293: __main__  Sanity Check Last Layer Weights: -0.004328871
INFO: 2024-06-25 08:36:49,358: __main__  Collated sample batch keys: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
INFO: 2024-06-25 08:36:49,358: __main__  Collated sample batch input_ids shape: torch.Size([2, 64])
INFO: 2024-06-25 08:36:49,358: __main__  Collated sample batch labels shape: torch.Size([2])
INFO: 2024-06-25 08:36:49,358: __main__  Collated sample batch attention_mask shape: torch.Size([2, 64])
INFO: 2024-06-25 08:36:49,429: __main__  Dry run status: SUCCESS
INFO: 2024-06-25 08:36:49,429: __main__  Dry run outputs keys: odict_keys(['loss', 'logits'])
INFO: 2024-06-25 08:36:49,429: __main__  Dry run outputs logits shape: torch.Size([2, 1])
INFO: 2024-06-25 08:36:49,429: __main__  Dry run outputs logits[0][0]: 0.79482317
INFO: 2024-06-25 08:36:49,429: __main__  Dry run outputs loss: 1.301110029220581
{'loss': 0.9527, 'grad_norm': 42.16598892211914, 'learning_rate': 5e-06, 'epoch': 0.5}
{'eval_qwk': 0.06788637457871927, 'eval_loss': 1.039064884185791, 'eval_runtime': 2.4788, 'eval_samples_per_second': 51.638, 'eval_steps_per_second': 25.819, 'epoch': 0.5}
{'loss': 1.0067, 'grad_norm': 13.904077529907227, 'learning_rate': 0.0, 'epoch': 1.0}
{'eval_qwk': 0.09054857898215463, 'eval_loss': 1.0542982816696167, 'eval_runtime': 1.758, 'eval_samples_per_second': 72.808, 'eval_steps_per_second': 36.404, 'epoch': 1.0}
{'train_runtime': 30.5798, 'train_samples_per_second': 4.186, 'train_steps_per_second': 2.093, 'train_loss': 0.9796938598155975, 'epoch': 1.0}
Validation QWK Score = 0.09054857898215463
```

```python
{"loss": 0.9527, "grad_norm": 42.16598892211914, "learning_rate": 5e-06, "epoch": 0.5}
{
    "eval_qwk": 0.06788637457871927,
    "eval_loss": 1.039064884185791,
    "eval_runtime": 2.4788,
    "eval_samples_per_second": 51.638,
    "eval_steps_per_second": 25.819,
    "epoch": 0.5,
}
{"loss": 1.0067, "grad_norm": 13.904077529907227, "learning_rate": 0.0, "epoch": 1.0}
{
    "eval_qwk": 0.09054857898215463,
    "eval_loss": 1.0542982816696167,
    "eval_runtime": 1.758,
    "eval_samples_per_second": 72.808,
    "eval_steps_per_second": 36.404,
    "epoch": 1.0,
}
{
    "train_runtime": 30.5798,
    "train_samples_per_second": 4.186,
    "train_steps_per_second": 2.093,
    "train_loss": 0.9796938598155975,
    "epoch": 1.0,
}
```

```text
Validation QWK Score = 0.09054857898215463
```

```python
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
    shared.default=False \
    shared.criterion=mse \
    shared.pooler_type=mean \
    shared.reinitialize_n_layers_of_backbone=0 \
    shared.freeze_these_layers_indices='[]' \
    shared.freeze_embeddings=False
```