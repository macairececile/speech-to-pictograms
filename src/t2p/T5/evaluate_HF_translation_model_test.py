# -*- coding: utf-8 -*-

# ------------ Libraries ------------ #
from datasets import load_dataset
import evaluate
import torch
import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter

metric = evaluate.load("sacrebleu")
source_lang = "fr"
target_lang = "frp"
max_input_length = 128
max_target_length = 128
prefix = ""


# ------------ Load dataset and process ------------ #
def convert_string_to_dict(string_dict):
    return ast.literal_eval(string_dict)


def load_dataset_eval(test_data):
    test_data = load_dataset('csv', data_files=[test_data],
                             delimiter='\t')

    test_data = test_data['train']

    test_data = test_data.map(lambda example: {'translation': convert_string_to_dict(example['translation'])})
    return test_data


# ------------ Tokenizer ------------ #
def load_model(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = model.to("cuda:0")
    return tokenizer, model


# ------------ Evaluation ------------ #
def generate(examples, tokenizer, model):
    inputs = [str(prefix + ex[source_lang]) for ex in examples["translation"]]
    targets = [str(ex[target_lang]) for ex in examples["translation"]]
    preds = []
    for i in inputs:
        inputs = tokenizer(i, return_tensors="pt").input_ids
        outputs = model.generate(inputs.to("cuda:0"), max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
        decode_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decode_out)
        preds.append(decode_out)

    return preds, targets


def generate_data_and_eval(test_data, tokenizer, model, output_file):
    preds_test, targets_test = generate(test_data, tokenizer, model)
    print("------ TEST EVAL ------")
    compute_metrics(preds_test, targets_test, tokenizer, output_file)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(preds, targets, tokenizer, output_file):
    decoded_preds, decoded_labels = postprocess_text(preds, targets)
    with open(output_file, 'w', encoding='utf-8') as file:
        for pred, label in zip(decoded_preds, decoded_labels):
            file.write(f"{pred}\t{label}\n")

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    print(result)


# ------------ Main ------------ #
def main(args):
    test_data = load_dataset_eval(args.test)
    tokenizer, model = load_model(args.model)
    generate_data_and_eval(test_data, tokenizer, model, args.output)


parser = ArgumentParser(description="Evaluation of a translation model HF with BLEU.",
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('--test', type=str, required=True,
                    help="")
parser.add_argument('--model', type=str, required=True,
                    help="")
parser.add_argument('--output', type=str, required=True,
                    help="")
parser.set_defaults(func=main)
args = parser.parse_args()
args.func(args)
