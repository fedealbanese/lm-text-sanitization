#!/usr/bin/env python
"""
Preprocess the ABCD dataset generating a text file with utterances formatted to
be used for training the ULM.

Usage: preprocess_abcd.py [options]

Options:
    --abcd FILE      path to ABCD dataset [default: ./data/abcd/abcd_v1.1.json]
    --output FILE    out file [default: ./data/abcd/abcd_v1.1.1_preprocessed.txt]
    --category CAT   ABCD category [default: train]
    --verbosity LVL  sets the log verbosity level from 0 to 4 [default: 3]
    --help           shows this description
"""

import docopt
import logging

from asapp.commons import ABCD


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

    abcd.replace_unique_names()
    if output_filename.endswith(".json"):
        abcd.dump(output_filename)
    else:
        logging.info(f"Preprocessing utterances to {output_filename}")
        with open(output_filename, 'w') as output_file:
            for utterance in abcd.iter_utterances(category):
                output_file.write(utterance)
                output_file.write('\n')

    print("Finished preprocessing successfully")


if __name__ == '__main__':
    main(docopt.docopt(__doc__, help=True))
