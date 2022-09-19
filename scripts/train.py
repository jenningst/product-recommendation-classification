
import argparse
import json
import logging
import os
import random
from shutil import copy

import mlflow
import numpy as np
from numpy import genfromtxt


logging.basicConfig(
    format='%(levelname)s - %(asctime)s - %(filename)s - %(message)s',
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    # read in args
    args = read_args()
    with open(args.config_file) as f:
        config = json.load(f)
    
    # define directories for data-reading and read in data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # TODO: start mlflow run

    train_data_path = os.path.join(base_dir, config['train_data_path'])
    test_data_path = os.path.join(base_dir, config['test_data_path'])
    
    train_data = genfromtxt(fname=train_data_path)
    test_data = genfromtxt(fname=test_data_path)

    if config['model'] == 'random_forest':
        # set the model to the RandomForest class
        pass
    elif config['model'] == 'logistic_regression':
        # set thd model to the LogisticRegression class
        pass
    elif config['model'] == 'support_vector_machine':
        # set model to SVCClassifier class
        pass
    else:
        # raise error
        pass

    # do model evaluation OR model training
    if not config['evaluate']:
        # train the model
        pass

    # mlflow logging params and metrics
    # module logging