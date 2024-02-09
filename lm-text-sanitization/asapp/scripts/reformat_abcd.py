#!/usr/bin/env python
"""
Reformat ABCD metadata in the Data Curation Hub format for consistency.

Usage: reformat_abcd.py [options]

Options:
    --abcd FILE      ABCD dataset [default: ./data/abcd/abcd_v1.1.json]
    --output CSV     output [default: ./data/abcd/abcd_v1.1_reformatted.csv]
    --category CAT   ABCD category to reformat [default: dev]
    --verbosity LVL  sets the log verbosity level from 0 to 4 [default: 3]
    --help           shows this description
"""

import docopt
import logging
import json
import re

from csv import DictWriter

from asapp.commons import ABCD


def get_entities(text, sensitive_data):
    """
    Extracts entities substring delimiters.
    :param text: str, utterance from where to extract entities.
    :param sensitive_data: dict, ABCD sensitive metadata.
    :return: list, entities in data curation hub format.
    """
    entities = []
    whitespaces = re.compile(r'\s+')
    for tag, value in sensitive_data.items():
        tokens = [
            f"\\b{re.escape(token)}\\b"
            for token in whitespaces.split(value.lower())
        ]
        sensitive_regex = re.compile('|'.join(tokens), re.I)
        for match in sensitive_regex.finditer(text + ' '):
            start = match.start()
            end = match.end()
            length = end - start
            term = match.group()
            if tag == "username" and len(text) > end and text[end] == "@":
                continue
            if term.lower() not in ABCD.blatantly_safe:
                entities.append([tag.upper(), start, length])
    return entities


def main(arguments):
    """Entry point of the script."""
    verbosity = int(arguments['--verbosity'])
    abcd_filename = arguments['--abcd']
    output_filename = arguments['--output']
    category = arguments['--category']

    logger = logging.getLogger()
    logger.setLevel(10 * (5 - int(verbosity)))

    print("Started dataset preprocessing")

    logging.info(f"Reading ABCD dataset from {abcd_filename}")
    abcd = ABCD(abcd_filename)

    logging.info(f"Preprocessing utterances to {output_filename}")
    with open(output_filename, 'w') as output_file:
        writer = DictWriter(output_file, ["text", "entities"])
        writer.writeheader()
        for entry in abcd.abcd_data[category]:
            sensitive_data = abcd.get_sensitive_data(entry)
            for utterance in abcd.iter_entry_utterances(entry):
                entities = get_entities(utterance, sensitive_data)
                writer.writerow({
                    'text': utterance,
                    'entities': json.dumps(entities)
                })

    print("Finished preprocessing successfully")


if __name__ == '__main__':
    main(docopt.docopt(__doc__, help=True))
