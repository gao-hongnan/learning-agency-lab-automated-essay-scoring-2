#!/bin/sh

# filepaths all put inside `lal/data/` directory, `topics_map.json` is inside
# `lal/conf/` directory.

CUDA_VISIBLE_DEVICES="6,7,8,9" torchrun --nnodes=1 --nproc_per_node=4 -m lal.entrypoint_local \
    lal/conf/deberta_reg.yaml \
    shared.taks=REGRESSION \
    shared.job_type=train \
    shared.num_labels=1 \
    shared.fold=2 \
    shared.padding_side=right \
    shared.max_length=1024 \
    shared.add_special_tokens=True \
    shared.padding=max_length \
    shared.output_hidden_states=False \
    shared.output_attentions=False \
    shared.pretrained_model_name_or_path=/data/model/deberta-v3-small \
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
    shared.criterion=ordinal_loss \
    shared.pooler_type=null
