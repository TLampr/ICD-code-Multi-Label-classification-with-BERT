from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, trainer_utils, EarlyStoppingCallback
import transformers
from datasets import load_dataset

from pathlib import Path
import json
import argparse
import itertools
import os


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def compute_multi_label_metrics(pred, thres=.5):
    labels = pred.label_ids
    preds = (pred.predictions >= thres).astype(int)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': round(acc, 4),
        'f1': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_dir', dest='config', help="Pass path to the directory containing the config",
                        type=str, required=False, default="./config.json")
    args = parser.parse_args()

    print("\n" + f"USING {args.config} AS A CONFIG FILE")

    config_file = open(args.config)
    config = json.load(fp=config_file)

    print('\n' + json.dumps(config, indent=4, sort_keys=True))

    epochs = config['parameters']['epochs']
    batch_size = config['parameters']['batch']['size']
    grad_accum = config['parameters']['batch']['accumulation']
    lr = config['parameters']['learning_rate']
    patience = config['parameters']['patience']
    save_folder = config["save_name"]

    def preprocess_data(examples):
        text = examples["full_note"]
        encoding = tokenizer(text, max_length=512, truncation=True, padding=False)  # , padding="max_length")
        eye = np.eye(label_num)
        labels = examples["codes"]
        encoding["labels"] = [np.sum([eye[:, cat2id[cat]] for cat in sample_cats], axis=0).tolist() for sample_cats in labels]
        return encoding


    train_dataset = load_dataset('csv', data_files=config['csv_data_path'], delimiter='|', split='train[0%:80%]')
    train_dataset = train_dataset.map(lambda x: {'codes': x['codes'].split(',')})
    val_dataset = load_dataset('csv', data_files=config['csv_data_path'], delimiter='|', split='train[80%:90%]')
    val_dataset = val_dataset.map(lambda x: {'codes': x['codes'].split(',')})
    test_dataset = load_dataset('csv', data_files=config['csv_data_path'], delimiter='|', split='train[90%:100%]')
    test_dataset = test_dataset.map(lambda x: {'codes': x['codes'].split(',')})

    labels = train_dataset['codes'] + val_dataset['codes'] + test_dataset['codes']
    flattened_labels = list(itertools.chain(*labels))
    unique_categories = set(flattened_labels)
    label_num = len(unique_categories)

    cat2id = {cat: id for id, cat in enumerate(unique_categories)}
    id2cat = {id: cat for cat, id in cat2id.items()}


    def label_to_index(labels):
        return {'labels': [[cat2id[cat] for cat in sample_cats] for sample_cats in labels['codes']]}

    if "model_path" in config:
        checkpoints = [config["model_path"]]
    else:
        checkpoints = [str(x) for x in Path(config['checkpoints']['path']).glob(config['checkpoints']['flag'])]

    np.random.seed(config["random_state"])
    for checkpoint in checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            model_max_length=512,
            max_length=512,
            truncation=True,
            padding=True)

        train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)

        val_dataset = val_dataset.map(preprocess_data, batched=True, remove_columns=val_dataset.column_names)

        test_dataset = test_dataset.map(preprocess_data, batched=True, remove_columns=test_dataset.column_names)

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=label_num)

        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        multi_training_args = TrainingArguments(
            output_dir=checkpoint + save_folder + '/output/',
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=grad_accum,
            save_total_limit=1,
            learning_rate=lr,
            evaluation_strategy=trainer_utils.IntervalStrategy.EPOCH,
            save_strategy=trainer_utils.IntervalStrategy.EPOCH,
            load_best_model_at_end=True,
            lr_scheduler_type=transformers.SchedulerType.CONSTANT
        )

        trainer = MultilabelTrainer(
            model=model,
            args=multi_training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_multi_label_metrics
        )

        trainer.train()

        prediction_output = trainer.predict(test_dataset=test_dataset)
        trainer.save_model(checkpoint + '/Checkpoints-Multiclass-Classification/Best-Model')
        accuracy = f"Accuracy for the Hold out Set: {prediction_output.metrics['test_accuracy']}\n"
        precision = f"Precision for the Hold out Set: {prediction_output.metrics['test_precision']}\n"
        recall = f"Recall for the Hold out Set: {prediction_output.metrics['test_recall']}\n"
        f1 = f"F1 score for the Hold out Set: {prediction_output.metrics['test_f1']}"
        text = accuracy + precision + recall + f1
        log_folder = checkpoint + save_folder + '/test_results/'
        os.makedirs(os.path.dirname(log_folder), exist_ok=True)
        with open(log_folder + 'result-Final.txt', 'w') as f:
            print(text, file=f)
        print(text)
