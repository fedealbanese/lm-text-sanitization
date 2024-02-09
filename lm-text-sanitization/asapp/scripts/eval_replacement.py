#!/usr/bin/env python
"""
Redacts a CSV file with PII annotations as produced by data curation hub.


Usage: eval_redaction.py [options] <model> <model-identifier> <input-csv> <output-csv>


Options:
   --s THLD         similarity threshold [default: 0.0]
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
from asapp.pfilteringlm import FillMaskModel, T5Model
from asapp.commons import ABCD, RedactorReplace, Unigram


def initialize_model(model_type, model_identifier = None):
    """
    Initialize the language model.

    Args:
    model_type(str): the type of model. Options are: fmm or t5.
    model_identifier(str): the path (for ulm models) or variation of the model (for fmm or gpt models).

    Returns:
    A language model.
    """
    if model_type == "fmm": #Fill Masking Model.
        model = FillMaskModel(model_name = model_identifier)
    elif model_type == "t5": #T5 model.
        model = T5Model(model_name = model_identifier)
    else:
        raise ValueError('Wrong model type.')
    return model


def print_results(s, elapsed_time, tp, tn, fp, fn, readable):
    """
    Prints a report of the evaluation results.
    Args:
    p(float): similarity threshold used.
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
        print(f"Redaction for s={s} finished successfully ({elapsed_time:.2f}s)")
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
        print(f"{s}\t{precision}\t{recall}\t{fpr}\t{specificty}")


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

def run_replacement(model, redactor, s, input_filename, output_filename, readable, default_replacement:str = "***"):
    """
    Runs the redaction experiment.

    Args:
    model(PFilteringModel): PFilteringModel to find replacement.
    redactor(Redactor): a redactor implementation.
    s(float): similarity threshold to use.
    input_filename(str): file from where to get utterances to redact.
    output_filename(str): file where to write redacted utterances.
    readable(bool): indicates whether to output human readable format.
    default_replacement(str): default replacement in case there are no alternative tokens with similarity bigger than s.
    """

    start_time = time.time()
    tp, tn, fp, fn = 0, 0, 0, 0
    safe_words = [
        "yes", "no", "ok", "okay",
        "he", "she", "it", "you", "we", "they", "his", "her", "their", "our", "your","yours", 
        "how", "why","who", "what", "where", "does","do", "did", "were","was","will",
        "name", "username", "usernames", "email", "emails", "mail", "mails", "phone", "cellphone", "number",
        "address", "zip", "code", "account", "id"
    ]
    with open(input_filename, 'r') as input_file, \
         open(output_filename, 'w') as output_file:
        reader = DictReader(input_file)
        header = list(reader.fieldnames) + ["replaced_text", "replacements"]
        writer = DictWriter(output_file, header)
        writer.writeheader()
        context = ""
        all_replacements = {word: word for word in safe_words} #key:redacted_token. value: replace_token. It is initialized with a list of safe words.
        for row in tqdm(reader):
            text = row["text"]
            entities = json.loads(row["entities"])
            redactions = json.loads(row["redactions"])
            #Find replacements
            replaced_text, index_t, utterance_replacements, all_replacements  = redactor.find_all_replacements(
                    redactions,
                    text,
                    context,
                    s,
                    model,
                    all_replacements,
            )
            replaced_text += text[index_t:] #add the safe text after the last redaction.
            context = redactor.redact_document_with_redactions(text, redactions) #update context
            row["replaced_text"] = json.dumps(replaced_text)
            row["replacements"] = json.dumps(utterance_replacements)    
            writer.writerow(row)
            #Evaluate
            sensitive_terms = get_sensitive_terms(text, entities)
            for token in Unigram.tokenize(text):
                term = token.term
                predicted_unsafe = any([
                    token.start >= redaction[1] and 
                    token.end <= redaction[1]+redaction[2] 
                    for redaction in redactions
                ])
                actually_unsafe = term not in ABCD.blatantly_safe and term in sensitive_terms
                token_replaced = (
                    predicted_unsafe and 
                    term in utterance_replacements and 
                    utterance_replacements[term] != default_replacement
                )
                safe_replacementent = utterance_replacements[term] not in sensitive_terms if token_replaced else True #if no replacement, defualt: True
                if predicted_unsafe and actually_unsafe: 
                    tp += 1
                elif predicted_unsafe and not actually_unsafe:
                    if not token_replaced: 
                        fp += 1
                    elif safe_replacementent: #and token_replaced
                        tn += 1
                    else: #(token_replaced and not safe_replacementent)
                        fn += 1
                elif not predicted_unsafe and not actually_unsafe: 
                    tn += 1
                elif not predicted_unsafe and actually_unsafe: 
                    fn += 1
    elapsed_time = time.time() - start_time
    print_results(s, elapsed_time, tp, tn, fp, fn, readable)


def main(arguments):
    """Entry point of the script."""
    verbosity = int(arguments['--verbosity'])
    logger = logging.getLogger()
    logger.setLevel(10 * (5 - int(verbosity)))

    ss = arguments['--s']
    ss = [float(s) for s in ss.split(',')]
    readable = arguments['--readable']
    model_type = arguments['<model>']
    model_identifier = arguments['<model-identifier>']
    input_filename = arguments['<input-csv>']
    output_filename = arguments['<output-csv>']

    if readable:
        print("Starting replacement")

    model = initialize_model(model_type, model_identifier)
    redactor = RedactorReplace(model)
    for s in ss:
        s_output = output_filename.replace(".csv", f"_s{s}.csv")
        run_replacement(model, redactor, s, input_filename, s_output, readable)


if __name__ == '__main__':
   main(docopt.docopt(__doc__, help=True))
