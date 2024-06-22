from __future__ import annotations

import logging
from typing import Any, List, TypedDict

import pandas as pd
import psutil
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
import json

from ..conf.config import Composer

logger = logging.getLogger(__name__)


def process_labels(df: pd.DataFrame, task: str, target_column: str = "score") -> pd.DataFrame:
    """
    Process the label column in the DataFrame based on the specified task in the Composer object.

    Args:
    df (pd.DataFrame): The DataFrame containing the score and possibly full_text columns.
    composer (Composer): An object containing configuration details including the task type.

    Returns:
    pd.DataFrame: The modified DataFrame with the processed label column.

    Raises:
    ValueError: If an invalid task type is provided.
    """
    df["label"] = df[target_column].apply(lambda x: x - 1)

    if task == "CLASSIFICATION":
        df["label"] = df["label"].astype("int32")
    elif task == "REGRESSION":
        df["label"] = df["label"].astype("float32")
    elif task == "CAUSAL_LM":
        logger.warning(
            "Using %s task, setting label to the input text. If you are using with "
            "huggingface model, then you do not need to shift the label because hf has "
            "done it for you inside the model class - this is confusing and not well "
            "documented.",
            task,
        )
        df["label"] = df["full_text"]
    else:
        raise ValueError(f"Invalid task type: {task}")

    return df


def add_prompt_name_group(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(df_1, df_2, on="essay_id", how="left")
    return merged_df[df_1.columns.tolist() + ["prompt_name"]].reset_index(drop=True)

def merge_topic_info_to_df(df: pd.DataFrame, train_topic_filepath: str, topics_map_path: str) -> pd.DataFrame:
    train_topic_df = pd.read_csv(train_topic_filepath)
    train_topic_df = train_topic_df[['essay_id', 'topics']]
    merge_df_and_train_topic = pd.merge(df, train_topic_df, on="essay_id", how="left")
    with open(topics_map_path, 'r') as file:
        topics_dict = json.load(file)

    topics_df = pd.DataFrame(list(topics_dict.items()), columns=['topics', 'description'])
    topics_df['topics'] = topics_df['topics'].astype(int)
    final_merged_df = pd.merge(merge_df_and_train_topic, topics_df, on='topics', how='left')
    final_merged_df = final_merged_df.drop(columns=['topics'])
    return final_merged_df


class TokenizedBatchDict(TypedDict):
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[List[int]]


def preprocess(
    sample: Any,
    tokenizer: PreTrainedTokenizerBase | PreTrainedTokenizerFast,
    task: str,
    inference: bool,
    system_prompt: str,
    return_tokenized_text: bool,
    *,  # tokenizer kwargs
    max_length: int,
    truncation: bool,
    padding: bool,
    return_tensors: str,
    add_special_tokens: bool,
    **kwargs: Any,
) -> TokenizedBatchDict:
    # print(type(sample)) #@ <class 'datasets.formatting.formatting.LazyRow'>

    if task == "CAUSAL_LM":
        prompt = sample["full_text"]
        answer = "" if inference else str(sample["score"])

        messages = [
            {
                "role": "user",
                "content": system_prompt + prompt,
            },
            {"role": "assistant", "content": "\n\nThe score is: " + answer},
        ]
        formatted_sample = tokenizer.apply_chat_template(messages, tokenize=False)

        if inference:
            formatted_sample = formatted_sample.replace("</s>", "")

        tokenized_sample = tokenizer(
            formatted_sample,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        if return_tensors == "pt":
            tokenized_sample["labels"] = tokenized_sample["input_ids"].clone()
        else:
            tokenized_sample["labels"] = tokenized_sample["input_ids"].copy()

        if not return_tokenized_text:
            return formatted_sample
        return tokenized_sample

    elif task in ["CLASSIFICATION", "REGRESSION"]:
        tokenized_sample = tokenizer(
            sample["description"],
            sample["full_text"],
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        tokenized_sample["labels"] = sample["label"]
        return tokenized_sample
    else:
        raise ValueError(f"Unsupported task type: {task}")


def create_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase | PreTrainedTokenizerFast,
    composer: Composer,
) -> Dataset:
    # NOTE: there will be `labels` column created by hf.Dataset
    ds = Dataset.from_pandas(df)
    logger.info("Dataset columns: %s", ds.column_names)

    batched = (
        True if tokenizer.is_fast else False
    )  # https://discuss.huggingface.co/t/num-proc-is-not-working-with-map/45641

    # FIXME: because prompt returns a list of str when batched = True, it cannot
    # add to system prompt which is a single string.
    ds = ds.map(
        preprocess,
        # batched=batched,
        # batch_size=1,
        num_proc=psutil.cpu_count(logical=True),
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": composer.shared.task,
            "inference": composer.shared.inference,
            "system_prompt": composer.shared.system_prompt,
            "return_tokenized_text": composer.shared.return_tokenized_text,
            "max_length": composer.shared.max_length,
            "truncation": composer.shared.truncation,
            "padding": composer.shared.padding,
            "return_tensors": composer.shared.return_tensors,
            "add_special_tokens": composer.shared.add_special_tokens,
        },
    )
    print(ds.column_names)
    columns_to_remove = [
        composer.shared.essay_id,
        composer.shared.full_text,
        composer.shared.score,
        composer.shared.fold_column,
        composer.shared.label,
        composer.shared.group_by,
        composer.shared.description,
    ]
    existing_columns = [col for col in columns_to_remove if col in ds.column_names]
    ds = ds.remove_columns(existing_columns)
    logger.info("Tokenized dataset columns: %s", ds.column_names)

    for row in ds.take(1):
        logger.debug("Dataset sample: %s", row)

    from rich.pretty import pprint

    pprint(ds)
    return ds
