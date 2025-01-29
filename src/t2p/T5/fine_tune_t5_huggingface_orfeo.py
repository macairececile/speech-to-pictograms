# -*- coding: utf-8 -*-

# ------------ Libraries ------------ #
from datasets import load_dataset
import evaluate
from transformers import TrainingArguments, Trainer
import torch
import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import numpy as np

metric = evaluate.load("sacrebleu")

# ------------ Load dataset ------------ #
train_data = load_dataset('csv', data_files=["/gpfswork/rech/czj/uef37or/Fine_tuning_t5/data_orfeo/train.csv"], delimiter='\t')
valid_data = load_dataset('csv', data_files=["/gpfswork/rech/czj/uef37or/Fine_tuning_t5/data_orfeo/valid.csv"], delimiter='\t')

train_data = train_data['train']
valid_data = valid_data['train']

def convert_string_to_dict(string_dict):
    return ast.literal_eval(string_dict)

train_data = train_data.map(lambda example: {'translation': convert_string_to_dict(example['translation'])})
valid_data = valid_data.map(lambda example: {'translation': convert_string_to_dict(example['translation'])})

# # ------------ Tokenizer ------------ #
checkpoint = "/gpfsstore/rech/czj/uef37or/pretrained_models/t5-base/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
source_lang = "fr"
target_lang = "frp"
max_input_length = 128
max_target_length = 128
prefix = ""

def preprocess_function(examples):
    inputs = [str(prefix + ex[source_lang]) for ex in examples["translation"]]
    targets = [str(ex[target_lang]) for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_valid = valid_data.map(preprocess_function, batched=True)


# ------------ Evaluation ------------ #
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# ------------ Fine-tuning ------------ #
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints_orfeo/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=40,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train() # resume_from_checkpoint=True
