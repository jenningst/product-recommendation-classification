import argparse
import csv
import json
import os
import string
import time
from typing import Dict, List, Text

from click import progressbar

import contractions
from alive_progress import alive_bar
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

INFILE_NAME = 'clean_review_data.json'
OUTFILE_NAME = 'featurized_data.json'
STOPWORDS = stopwords.words('english')
  

def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Loads and processes a raw dataset.')
    parser.add_argument(
        '--interim-data-path', 
        type=str,
        help='a directory path to the intermediate data (i.e., post-cleansing)'
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
        datapoints = json.load(f)
        return [datapoint for datapoint in datapoints]


def trim_features(datapoints: List[Dict]) -> List[Dict]:
    '''
    Drops all but the necessary columns for modeling.

    Parameters
    ----------
    datapoints : List[Dict]
        List of review instances.

    Returns
    -------
    List[Dict]
        A list of datapoints containing only select features.
    '''
    print('Trimming features...')

    trimmed_datapoints = []
    for datapoint in datapoints:
        trimmed_datapoint = {
            'review_text': datapoint['review_text'],
            'recommended': datapoint['recommended_id']
        }
        trimmed_datapoints.append(trimmed_datapoint)
    print('Done')
    return trimmed_datapoints


def featurize_clean_review_tokens(datapoints: List[Dict]) -> List[Dict]:
    '''
    Adds features for clean tokens, number of clean tokens, lemmatized tokens, 
    and lemmatized sentence for each review in a dataset.

    Parameters
    ----------
    datapoints : List[Dict]
        List of review instances.

    Returns
    -------
    List[Dict]
        List of review instances with new features.
    '''
    print('Cleansing Reviews...')
    lemmatizer = WordNetLemmatizer()

    featurized_datapoints = []
    for datapoint in datapoints:
        featurized_review = []
        lemmatized_review = []
        n_tokens = 0
        for word in datapoint['review_text'].split(' '):
            # expand contractions, this returns a list of the original token if no contraction or a list of expanded tokens
            expanded_contraction = contractions.fix(word)
            for part in expanded_contraction.split(' '):
                # strip the part of punctuation and digits, then remove stopwords
                char_string = ''.join([ char for char in part if char not in string.punctuation and not char.isdigit() ])
                lemma_part = [ lemmatizer.lemmatize(token.lower(), 'v') for token in char_string.split() ]
                clean_part = [ token.lower() for token in char_string.split() if token.lower() not in STOPWORDS ]

                # extend the featurized review with the clean tokens and update the n_token count
                if clean_part_length := len(clean_part) > 0:
                    featurized_review.extend(clean_part)
                    lemmatized_review.extend(lemma_part)
                    n_tokens += clean_part_length

            # append new data
            datapoint['n_tokens'] = n_tokens
            datapoint['clean_review_tokens'] = featurized_review 
            datapoint['lemma'] = lemmatized_review
            datapoint['lemma_sent'] = ' '.join(lemmatized_review)
        featurized_datapoints.append(datapoint)
    print('Done')
    return featurized_datapoints


def featurize_sentiment_scores(datapoints: List[Dict]) -> List[Dict]:
    '''
    Adds features for sentiment intensity scores.

    Parameters
    ----------
    datapoints : List[Dict]
        List of review instances.

    Returns
    -------
    List[Dict]
        List of review instances with new features.
    '''
    print('Calculating Sentiment Scores...')
    intensity_analyzer = SentimentIntensityAnalyzer()

    featurized_datapoints = []
    for datapoint in datapoints:
        # calculate individual sentiment scores
        sentence = TextBlob(datapoint['lemma_sent'])
        scores = intensity_analyzer.polarity_scores(datapoint['lemma_sent'])
        datapoint['polarity'] = sentence.sentiment.polarity
        datapoint['subjectivity'] = sentence.sentiment.subjectivity
        datapoint['negative'] = scores['neg']
        datapoint['neutral'] = scores['neu']
        datapoint['positive'] = scores['pos']
        featurized_datapoints.append(datapoint)
    print('Done')
    return featurized_datapoints


def featurize_data(datapoints: List[Dict]) -> List[Dict]:
    return featurize_sentiment_scores(
        featurize_clean_review_tokens(
            trim_features(datapoints)
        )
    )


if __name__ == '__main__':
    args = read_args()

    # read in interim dataset, featurize, and output to the interim directory
    if not os.path.exists(os.path.join(args.interim_data_path, OUTFILE_NAME)):
        datapoints =  read_data(args.interim_data_path)
        featurized_datapoints = featurize_data(datapoints)

        with open(os.path.join(args.output_dir, OUTFILE_NAME), 'w') as f:
            json.dump(featurized_datapoints, f)
        print(f'New outfile created at {os.path.join(args.output_dir, OUTFILE_NAME)}')
    else:
        print('Found existing outfile; doing nothing')