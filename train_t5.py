import argparse
import json
import logging

import torch
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from datasets import Dataset


def fine_tuning(model_name, file_path, representation_rank_file, test_df_path, ranks_to_finetune,
                label_column, max_input_length=512, max_target_length=10, batch_size=8, epochs=5, debug=False):
    for rank in ranks_to_finetune:

        model, tokenizer, df, device, rankings = bootstrap(model_name, file_path, representation_rank_file)
        label_repr = extract_rank_x_repr(rank, rankings)

        if debug:
            logger.debug(f"Rank: {rank}, representations {label_repr}")

        def preprocess_data(examples):
            inputs = [text for text in examples["Sentence"] if text != None]

            # Setup the tokenizer for targets
            texts_target = [label_repr[text] for text in examples[label_column] if text != None]

            model_inputs = tokenizer(inputs, max_length=max_input_length, padding='max_length',
                                     truncation=True)
            model_inputs['labels'] = tokenizer(text_target=texts_target, max_length=max_target_length,
                                               padding='max_length', truncation=True)['input_ids']
            return model_inputs

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred

            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            if debug:
                with open(f"targets_debug_rank_{'minus_' + str(abs(rank)) if rank < 0 else rank}.json", "w") as target:
                    target.write(json.dumps(decoded_labels))

                with open(f"prediction_debug_rank_{'minus_' + str(abs(rank)) if rank < 0 else rank}.json",
                          "w") as target:
                    target.write(json.dumps(decoded_preds))

            # Calculate accuracy and f-score for each class
            report = classification_report(decoded_preds, decoded_labels, output_dict=True)

            return {
                'accuracy': report['accuracy'],
                'f1_score_macro': report['macro avg']['f1-score'],
                'f1_score_weighted': report['weighted avg']['f1-score'],
            }

        model_dir = f"models/fine-tuned-model"

        model_args = Seq2SeqTrainingArguments(
            model_dir,
            evaluation_strategy="epoch",
            eval_steps=50000,
            logging_strategy="epoch",
            logging_steps=1000,
            save_strategy="epoch",
            save_steps=50000,
            learning_rate=4e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=False
        )

        test_df = pd.read_csv(test_df_path)
        test = Dataset.from_pandas(test_df)
        train = Dataset.from_pandas(df)

        # Tokenize data
        tokenized_train = train.map(preprocess_data,
                                    batched=True)

        tokenized_test = test.map(preprocess_data,
                                  batched=True)

        train_dataset = tokenized_train.remove_columns(
            [column for column in tokenized_train.features if column not in ['attention_mask', 'labels', 'input_ids']])
        test_dataset = tokenized_test.remove_columns(
            [column for column in tokenized_test.features if column not in ['attention_mask', 'labels', 'input_ids']])

        train_dataset = train_dataset.shuffle(seed=42)
        test_dataset = test_dataset.shuffle(seed=42)

        # Load data collator
        # data_collator = DataCollatorForSeq2Seq(tokenizer)

        trainer = Seq2SeqTrainer(
            model=model,
            args=model_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            # data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        # trainer.save_model(f"models/rank_{'minus_' + str(abs(rank)) if rank < 0 else rank}/fine-tuned-model")
        predictions_to_save = trainer.predict(test_dataset)
        test_pred = predictions_to_save.predictions
        test_labels = predictions_to_save.label_ids
        decoded_test_preds = tokenizer.batch_decode(test_pred, skip_special_tokens=True)
        decoded_test_labels = tokenizer.batch_decode(test_labels, skip_special_tokens=True)

        with open(f"prediction_rank_{'minus_' + str(abs(rank)) if rank < 0 else rank}.tsv", "w") as output_file:
            output_file.write("y_true\ty_pred\n")
            for pred, ground_truth in zip(decoded_test_preds, decoded_test_labels):
                output_file.write(f"{ground_truth}\t{pred}\n")


def extract_rank_x_repr(rank, ranked_label):
    return {key: (ranked_label[key][rank]) for key in ranked_label}


def bootstrap(model_name, dataset_path, representation_rank_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"{device} found!")

    df = pd.read_csv(dataset_path)
    logger.debug(f"Dataset loaded!")
    with open(representation_rank_file) as input_file:
        label_repr_ranked = json.loads(input_file.read())

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Model and tokenizer loaded from pretrained {model_name}.")
    logger.debug(f"{model}")

    model.to(device)
    logger.debug(f"Model moved to {device}")

    return model, tokenizer, df, device, label_repr_ranked


if __name__ == "__main__":

    # Uninteresting argument parser configuration stuff
    parser = argparse.ArgumentParser(description="Train T5 with representation from a ranked list of "
                                                 "label representation sorted on distance from train example")
    parser.add_argument("file_path", help='Path to the file containing the training dataset')
    parser.add_argument("label_column", type=str, help='Name of the column containing the labels to analyze')
    parser.add_argument("representation_rank_file",
                        help="Path to the file containing the ordered representation for each label")
    parser.add_argument("--debug", help='Flag to activate debug mode', action="store_true")
    parser.add_argument("--model_name", help="Hugging face string of the model and tokenizer name (T5)",
                        default="gsarti/it5-base")
    parser.add_argument("--batch_size", default=8, help="Set the model batch size.")
    parser.add_argument("--epochs", default=10, help="Set how many epochs train the model on.")
    args = parser.parse_args()

    # Uninteresting logging configuration stuff
    logger = logging.getLogger(__name__)

    console_handler = logging.StreamHandler()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    fine_tuning(model_name=args.model_name, file_path=args.file_path,
                representation_rank_file=args.representation_rank_file,
                test_df_path="data/test_1_filtered.csv",
                ranks_to_finetune=range(10),
                label_column=args.label_column, batch_size=int(args.batch_size), epochs=int(args.epochs),
                debug=args.debug)

