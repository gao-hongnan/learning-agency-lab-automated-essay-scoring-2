# Learning Agency Lab Automated Essay Scoring 2

## Entrypoint

```bash
~ $ git clone https://github.com/gao-hongnan/learning-agency-lab-automated-essay-scoring-2.git
~ $ cd learning-agency-lab-automated-essay-scoring-2
. $ python -m venv .venv
. $ source .venv/bin/activate
. (.venv) $ pip install -r requirements.txt
```

Make sure to add the training data in the `lal/data` directory.

Sample command:

```bash
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
    shared.default=True \
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
    shared.enable_mixed_precision=True
```

If you need to use weights and biases, change or override `shared.entity` and
also make sure the wandb API key is set. Then run with `ALLOW_WANDB` set to
`True`.

```bash
export ALLOW_WANDB=True
```

## Attention Pooling Patch

-   918d0f306a5c3e8c9ef0db1752c5e5116adbcfdb
    -   https://wandb.ai/gaohn-teaching-mlops/learning-agency-lab-automated-essay-scoring-2/runs/llem1kv0?nw=nwuserhongnangao

## [Training/Inference] Mixed Precision

On a related topic, if I load a model in float32 and subsequently train it using
mixed precision fp16 / bf16 mode, during inference, is it "ok" to load the model
back in float16 as in some cases, the inference server may not have the same gpu
specs as the training setup.

## [HF] Do Not Shift Labels When Preparing Data for Causal Language Modeling

In
[the HuggingFace documentation on causal modeling](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#initializing-a-new-model),
it is highlighted that when preparing the dataset, we construct the labels by
**_cloning_** the inputs instead of shifting the inputs by one position to the
right. This is because the model will shift the inputs and labels to align them
inside the model. Quoting the documentation:

> ⚠️ Shifting the inputs and labels to align them happens inside the model, so
> the data collator just copies the inputs to create the labels.

See also:

-   https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408
-   https://github.com/huggingface/transformers/blob/571dd693b5d20754ecc472030903a94f92cfa9f8/src/transformers/models/gpt2/modeling_gpt2.py#L1100
-   https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#initializing-a-new-model

## [Tokenizer] Pad Left Or Pad Right

-   https://discuss.huggingface.co/t/the-effect-of-padding-side/67188/6
-   https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
-   https://huggingface.co/docs/transformers/en/model_doc/gpt2#usage-tips
-   https://ai.stackexchange.com/questions/41485/while-fine-tuning-a-decoder-only-llm-like-llama-on-chat-dataset-what-kind-of-pa

## [Tokenizer] Set PAD Token as EOS Token

-   https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954/13
-   https://discuss.huggingface.co/t/how-does-gpt-decide-to-stop-generating-sentences-without-eos-token/41623
-   https://www.reddit.com/r/LocalLLaMA/comments/17n22vk/troubleshooting_special_tokens_in_transformer/

## [HF] Convert PyTorch Dataset to HF Dataset

-   https://github.com/huggingface/datasets/issues/4983

## [Training] Attempting to unscale FP16 gradients

Can we inference a model with fp16 but loaded and trained with fp32.

-   https://github.com/OpenAccess-AI-Collective/axolotl/issues/1031
-   https://github.com/huggingface/transformers/issues/23165
-   https://github.com/huggingface/peft/issues/341#issuecomment-1884911753

# TODO

## Stratified(Group)KFold

-   [https://www.kaggle.com/code/emiz6413/predict-the-prompts](https://www.kaggle.com/code/emiz6413/predict-the-prompts)
    -   help us group the `essay_id` to 7 unique `prompt_name`.

## Optimized Threshold For Regression

-   [https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/502279](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/502279)

## Pooling

-   [https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently](https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently)

## Loss, Combined Loss, Auxiliary Loss etc

-   https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/494873

## MISC

-   https://github.com/huggingface/transformers/blob/b7672826cad31e30319487af876e608d8af7d37b/examples/pytorch/language-modeling/run_clm.py#L264
    -   logging level
    -   see modal example also
-   modal container exec with nvitop

## Train

1. group by prompt id?
2. lr finder huggingface
3. unify label to labels, and change target col name from label to labels or
   label to score.

## References

### Transformers Representations

-   https://www.kaggle.com/code/javigallego/deberta-from-the-ground-up-2-approaches
-   https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers
-   https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently

### Profiling

-   https://github.com/yqhu/profiler-workshop/blob/c8d4a7c30a61cc7b909d89f88f5fd36b70c55769/hf_training_trainer_prof.py

### Training Repo

-   https://github.com/pytorch/torchtune/blob/main/torchtune/utils/memory.py
-   https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/train_utils.py
-   https://github.com/OpenAccess-AI-Collective/axolotl/tree/main?tab=readme-ov-file#debugging-axolotl
