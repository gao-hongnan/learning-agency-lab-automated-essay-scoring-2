from __future__ import annotations

import logging
import os
from typing import Literal, Tuple

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)


def read_csv_file(filepath: str | os.PathLike) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError("The file %s was not found.", filepath) from None
    except pd.errors.EmptyDataError:
        raise ValueError("No data: The file %s is empty.", filepath) from None


def load_data(
    df_or_filepath: pd.DataFrame | str | os.PathLike[str],
    fold: int,
    job_type: Literal["train", "pretrain", "fullfit", "train_with_external", "debug"],
    fold_column: Literal["fold"] = "fold",
    pretraining_data_filepath: str | os.PathLike[str] | None = None,
    external_data_df_or_filepath: pd.DataFrame | str | os.PathLike[str] | None = None,
    debug_samples: int = 128,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """This func only for this project."""
    df = df_or_filepath if isinstance(df_or_filepath, pd.DataFrame) else read_csv_file(df_or_filepath)

    train_folds = df[df[fold_column] != fold]
    valid_folds = df[df[fold_column] == fold].copy()

    if job_type == "train":
        logger.info("Training fold %s", fold)
        train_folds = train_folds[train_folds[fold_column] != fold]
        valid_folds = valid_folds[valid_folds[fold_column] == fold]

    if job_type == "pretrain":
        logger.info("Pretraining fold %s", fold)
        if pretraining_data_filepath is None:
            raise ValueError("Pretraining data path is required for this job type.")
        # config.training.epochs = 1
        train_folds = pd.read_csv(pretraining_data_filepath)

    if job_type == "fullfit":
        logger.info("Fullfitting all folds")
        train_folds = pd.concat([train_folds, valid_folds])

    if job_type == "train_with_external":
        logger.info("Training with external data")
        if external_data_df_or_filepath is None:
            raise ValueError("External data path is required for this job type.")
        external_data = (
            external_data_df_or_filepath
            if isinstance(external_data_df_or_filepath, pd.DataFrame)
            else read_csv_file(external_data_df_or_filepath)
        )
        external_data = pd.DataFrame(
            external_data[["essay_id_comp", "full_text", "holistic_essay_score", "label"]].values,
            columns=["essay_id", "full_text", "score", "label"],
        )

        train_folds = pd.concat([train_folds, external_data], axis=0).drop_duplicates(subset=["full_text"])
        train_folds = train_folds[~train_folds["full_text"].isin(valid_folds["full_text"])]

    if job_type == "debug":
        logger.info("Debugging with %s samples", debug_samples)
        train_folds = train_folds.head(debug_samples)
        valid_folds = valid_folds.head(debug_samples)
    return train_folds.reset_index(drop=True), valid_folds.reset_index(drop=True)
