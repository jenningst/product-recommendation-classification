import argparse
import csv
import json
import os
from typing import Dict, List

OUTFILE_NAME = 'clean_review_data.json'


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Loads and processes a raw dataset.')
    parser.add_argument(
        '--raw-data-path', 
        type=str,
        help='a directory path to the raw data'
    )
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='a directory path for the processed (output) data'
    )
    return parser.parse_args()


def read_data(datapath: str) -> List[Dict]:
    '''
    Read raw data from a directory path.

    Parameters
    ----------
    datapath : str
        The path to the raw data.

    Returns
    -------
    List[Dict]
        A list of json objects for each instance in the dataset.
    '''
    with open(datapath, 'r') as f:
        reader = csv.DictReader(f, delimiter=',', fieldnames=[
            'index',
            'clothing_id',
            'age',
            'title',
            'review_text',
            'rating',
            'recommended_id',
            'positive_feedback_count',
            'division_name',
            'department_name',
            'class_name',
        ])
        next(reader) # skip the header
        return [row for row in reader]


def normalize_and_clean(datapoints: List[str]) -> List[str]:
    '''
    Performs basic data-cleansing on the raw dataset by removing
    records with empty review text or recommended status indicator.

    Parameters
    ----------
    datapoints : List[str]
        A list of comma-separated product review data.

    Returns
    -------
    List[str]
        A list of cleansed json objects from the dataset.
    '''
    cleansed_datapoints = []
    for data in datapoints:
        if data['review_text'] != '':
            if data['recommended_id'] != '':
                cleansed_datapoints.append(data)
    return cleansed_datapoints


if __name__ == '__main__':
    args = read_args()

    # read in the raw data, clean it, and output to the interim directory
    if not os.path.exists(os.path.join(args.output_dir, OUTFILE_NAME)):
        raw_datapoints = read_data(args.raw_data_path)
        clean_datapoints = normalize_and_clean(raw_datapoints)

        with open(os.path.join(args.output_dir, OUTFILE_NAME), 'w') as f:
            json.dump(clean_datapoints, f)
        print(f'New outfile created at {os.path.join(args.output_dir, OUTFILE_NAME)}')
    else:
        print('Found existing outfile; doing nothing')