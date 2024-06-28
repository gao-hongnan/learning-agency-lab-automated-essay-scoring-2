from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def validate_model(
    valid_df: pd.DataFrame,
    tokenized_valid_dataset: Any,  # Define more specifically based on your dataset type
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    composer: Any,  # Define more specifically if you have a specific type for composer
    preprocess,  # Add specific callable type hints if possible
    trainer: Any,  # Define more specifically based on your trainer type
) -> float:
    """
    Performs model validation for different tasks and calculates the QWK score.

    Args:
    - valid_df (pd.DataFrame): The validation dataframe.
    - tokenized_valid_dataset (Any): Tokenized validation dataset.
    - tokenizer (PreTrainedTokenizerBase): Tokenizer used for processing text data.
    - model (PreTrainedModel): The model to be validated.
    - composer (Any): Configuration object with settings for the task.
    - preprocess (callable): Function to preprocess the data.
    - trainer (Any): Trainer object used for model prediction.

    Returns:
    - float: The calculated QWK score.
    """
    y_trues = valid_df.score.values
    qwk = 0

    if composer.shared.task == "REGRESSION":
        logits = trainer.predict(tokenized_valid_dataset).predictions
        valid_df["logits"] = logits + 1
        qwk = cohen_kappa_score(
            y_trues,
            valid_df.logits.values.clip(1, 6).round(0),
            weights="quadratic",
        )

    elif composer.shared.task == "SINGLE_LABEL_CLASSIFICATION":
        logits = trainer.predict(tokenized_valid_dataset).predictions
        columns = [f"p{x}" for x in range(composer.shared.num_labels)]
        valid_df[columns] = logits
        qwk = cohen_kappa_score(
            y_trues,
            valid_df.iloc[:, -composer.shared.num_labels :].values.argmax(axis=1) + 1,
            weights="quadratic",
        )

    elif composer.shared.task == "CAUSAL_LM":
        preds = []
        model.eval()
        for i, row in valid_df.iterrows():
            tokenized_sample = preprocess(
                row,
                tokenizer=tokenizer,
                task=composer.shared.task,
                inference=True,  # Assuming inference setup modifies behavior suitably
                system_prompt=composer.shared.system_prompt,
                return_tokenized_text=composer.shared.return_tokenized_text,
                max_length=composer.shared.max_length,
                truncation=composer.shared.truncation,
                padding=composer.shared.padding,
                return_tensors="pt",
                add_special_tokens=composer.shared.add_special_tokens,
            )

            tokenized_sample = {k: v.to(model.device) for k, v in tokenized_sample.items()}
            generated_ids = model.generate(
                **tokenized_sample,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            try:
                answer = decoded[0].rsplit("The score is: ", 1)[1]
                preds.append(int(answer))
            except:
                preds.append(3)  # Default/fallback prediction

        qwk = cohen_kappa_score(valid_df.score.values, preds, weights="quadratic")

    # Save results
    valid_df.to_csv(
        f"{str(composer.shared.output_dir)}/valid_df_fold_{composer.shared.fold}.csv",
        index=False,
    )
    print(f"Validation QWK Score = {qwk}")
    return qwk
