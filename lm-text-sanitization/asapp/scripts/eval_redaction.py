#!/usr/bin/env python
"""
Redacts a CSV file with PII annotations as produced by data curation hub.


Usage: eval_redaction.py [options] <model> <model-identifier> <input-csv> <output-csv>


Options:
   --p THLD         privacy threshold (greater than clean up) [default: 0.0]
   --k THLD         k for kanonymity [default: 2]
   --s THLD         similarity threshold [default: 0.0]
   --w2e-load DIR   path to pickle word2embedding dictionary
   --w2e-dump DIR   path to save pickle word2embedding dictionary
   --readable       toggles human readable outpu (tab separated by default)
   --verbosity LVL  sets the log verbosity level from 0 to 4 [default: 3]
   --help           shows this description
"""


import docopt
import logging
import json
import time

from csv import DictReader, DictWriter
from tqdm import tqdm
from asapp.pfilteringlm import ULM, FillMaskModel, GPT2Model, T5Model
from asapp.kanonymity import KAnonymizer
from asapp.commons import ABCD, Redactor, RedactorReplace, Unigram


def initialize_model(model_type, model_identifier = None, k = 2, w2e_load_filename = None):
    """
    Initialize the language model.

    Args:
    model_type(str): the type of model. Options are: ulm, fmm, gpt, t5 or kanonymizer.
    model_identifier(str): the path (for ulm models) or variation of the model (for fmm or gpt models).
    k(int): k parameter of KAnonymizer model.
    w2e_load_filename(str): path to pickle word2embedding dictionary.

    Returns:
    A language model.
    """
    if model_type == "ulm": #Unigram Language Model.
        model = ULM(model_filename = model_identifier)
    elif model_type == "fmm": #Fill Masking Model.
        model = FillMaskModel(model_name = model_identifier)
    elif model_type == "fmm-replace": #Fill Masking model.
        model = FillMaskModel(model_name = model_identifier)
    elif model_type == "gpt": #GPT model.
        model = GPT2Model(model_name = model_identifier)
    elif model_type == "t5": #T5 model.
        model = T5Model(model_name = model_identifier)
    elif model_type == "t5-replace": #T5 model.
        model = T5Model(model_name = model_identifier)
    elif model_type == "kanonymity": #k-anonymity
        model = KAnonymizer(k = k, model_filename = model_identifier)
        model.load()
    else:
        raise ValueError('Wrong model type.')
    if w2e_load_filename is not None:
        model.upload_word2embedding_file(w2e_load_filename)
    return model


def print_results(p, elapsed_time, tp, tn, fp, fn, readable):
    """
    Prints a report of the evaluation results.
    Args:
    p(float): privacy threshold used.
    elapsed_time(float): run time in seconds.
    tp(int): true positive count.
    tn(int):true negative count.
    fp(int): false positive count.
    fn(int): false negative count.
    readable(bool): whether to print a human readable format or not.
    """
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision, recall, f1 = 0, 0, 0
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    fpr = 0
    if fp > 0:
        fpr = fp / (fp + tn)
    specificty = 0
    if tn > 0:
        specificty = tn / (tn + fp)
    if readable:
        print("=" * 80)
        print(f"Redaction for p={p} finished successfully ({elapsed_time:.2f}s)")
        print(f"True positives: {tp} (unsafe caught)")
        print(f"True negatives: {tn} (safe let through)")
        print(f"False positives: {fp} (safe over-redacted)")
        print(f"False negatives: {fn} (unsafe leaked)")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Precision: {precision:.4f} (unsafe_redacted / total_redacted)")
        print(f"Recall: {recall:.4f} (unsafe_redacted / total_unsafe)")
        print(f"FPR: {fpr:4f} (safe_redacted / total_safe)")
        print(f"Specificity: {specificty:.4f} (safe_unredacted / total_safe)")
    else:
        print(f"{p}\t{precision}\t{recall}\t{fpr}\t{specificty}")


def get_sensitive_terms(text, entities):
    """
    Returns the set of sensitive terms in the text.

    Args:
    text(str): text from where to get sensitive terms.
    entities(list): tuples (tag, start, length) delimiting sensitive substrings.

    Returns:
    result(set): sensitive term tokens.
    """
    result = set()
    for entity in entities:
        tag, start, length = entity
        end = start + length
        entity_text = text[start:end]
        for token in Unigram.tokenize(entity_text):
            result.add(token.term)
    return result


def run_experiment(redactor, p, input_filename, output_filename, readable):
    """
    Runs the redaction experiment.
    Args:
    redactor(Redactor): a redactor implementation.
    p(float): the privacy threshold to use.
    input_filename(str): file from where to get utterances to redact.
    output_filename(str): file where to write redacted utterances.
    readable(bool): indicates whether to output human readable format.
    """
    start_time = time.time()
    tp, tn, fp, fn = 0, 0, 0, 0
    with open(input_filename, 'r') as input_file, \
         open(output_filename, 'w') as output_file:
        reader = DictReader(input_file)
        header = list(reader.fieldnames) + ["redactions"]
        writer = DictWriter(output_file, header)
        writer.writeheader()
        for row in tqdm(reader):
            text = row["text"]
            entities = json.loads(row["entities"])
            safe_terms, redactions = redactor.find_redactions(text)
            row["redactions"] = json.dumps(redactions)
            writer.writerow(row)
            sensitive_terms = get_sensitive_terms(text, entities)
            for token in Unigram.tokenize(text):
                term = token.term
                predicted_unsafe = any([
                    token.start >= redaction[1] and 
                    token.end <= redaction[1] + redaction[2] 
                    for redaction in redactions
                ])
                actually_unsafe = term not in ABCD.blatantly_safe and term in sensitive_terms
                if predicted_unsafe and actually_unsafe:
                    tp += 1
                elif not predicted_unsafe and not actually_unsafe:
                    tn += 1
                elif predicted_unsafe and not actually_unsafe:
                    fp += 1
                elif not predicted_unsafe and actually_unsafe:
                    fn += 1
    elapsed_time = time.time() - start_time
    print_results(p, elapsed_time, tp, tn, fp, fn, readable)


def main(arguments):
    """Entry point of the script."""
    verbosity = int(arguments['--verbosity'])
    logger = logging.getLogger()
    logger.setLevel(10 * (5 - int(verbosity)))

    ps = arguments['--p']
    ps = [float(p) for p in ps.split(',')]
    k = int(arguments['--k'])
    s = float(arguments['--s'])
    w2e_load_filename = arguments['--w2e-load']
    w2e_dump_filename = arguments['--w2e-dump']
    readable = arguments['--readable']
    model_type = arguments['<model>']
    model_identifier = arguments['<model-identifier>']
    input_filename = arguments['<input-csv>']
    output_filename = arguments['<output-csv>']

    for p in ps:
        model = initialize_model(model_type, model_identifier, k, w2e_load_filename)
        if model_type.endswith('-replace'):
            redactor = RedactorReplace(model, p, s)
        else:
            redactor = Redactor(model, p)
        p_output = output_filename.replace(".csv", f"_p{p}.csv")
        run_experiment(redactor, p, input_filename, p_output, readable)
        if w2e_dump_filename is not None:
            model.dump_word2embedding_file(w2e_dump_filename)


if __name__ == '__main__':
   main(docopt.docopt(__doc__, help=True))
