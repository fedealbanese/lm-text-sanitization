#!/usr/bin/env python
"""
Trains an ULM or a KAnonymity model from a text file.

Usage: train_model.py [options] <model-type> <dataset> <model>

Options:
    --p THLD         threshold to clean up model [default: 0.0]
    --column C       document column if file is csv
    --verbosity LVL  sets the log verbosity level from 0 to 4 [default: 3]
    --help           shows this description
"""

import docopt
import logging
import time

from csv import DictReader

from asapp.pfilteringlm import ULM
from asapp.kanonymity import KAnonymizer


def main(arguments):
    """Entry point of the script."""
    verbosity = int(arguments['--verbosity'])
    p = float(arguments['--p'])
    csv_column = arguments['--column']
    model_type = arguments['<model-type>']
    dataset_filename = arguments['<dataset>']
    model_filename = arguments['<model>']

    logger = logging.getLogger()
    logger.setLevel(10 * (5 - int(verbosity)))

    print("Started model training")
    start_time = time.time()

    logging.info(f"Initializing model: {model_filename}")
    if model_type == "ulm":
        model = ULM(
            model_filename=model_filename,
            load_init=False
        )
    elif model_type == "kanonymity":
        model = KAnonymizer(
            k=1,
            model_filename=model_filename
        )
    else:
        raise ValueError('model-type should be "ulm" or "kanonymity"')
    
    logging.info(f"Training with dataset: {dataset_filename}")
    with open(dataset_filename) as dataset_file:
        if dataset_filename.endswith('.txt'):
            for document in dataset_file:
                model.add_doc(document)
        else:
            delimiter = (
                ',' if dataset_filename.endswith('.csv') else
                '\t' if dataset_filename.endswith('.tsv') else
                None
            )
            if delimiter is None:
                logging.error("Unknown file format, only txt and csv are supported")
                return
            if csv_column is None:
                logging.error("CSV ULM training requires column name")
                return
            csv_reader = DictReader(dataset_file, delimiter=delimiter)
            for row in csv_reader:
                model.add_doc(row[csv_column])

    if model_type == "ulm" and p > 0.0:
        logging.info(f"Cleaning up terms for p = {p}")
        model.clean_up(p)

    logging.info(f"Dumping model to file: {model_filename}")
    model.dump()

    elapsed_time = time.time() - start_time
    print(f"Finished model training successfully ({elapsed_time:.2f}s)")


if __name__ == '__main__':
    main(docopt.docopt(__doc__, help=True))
