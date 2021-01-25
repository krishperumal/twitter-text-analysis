from argparse import ArgumentParser
import os
from gensim import corpora
import csv
import pickle
import spacy
import json


def parse_arguments():
    parser = ArgumentParser('Predict topics for all tweets in JSON format in '
                            'input directory using trained LDA model.')
    parser.add_argument('--input_dir_path', '-i', type=str,
                        help='Path to input directory containing JSON files '
                        'with tweets.')
    parser.add_argument('--mallet_path', '-m', type=str,
                        help='Path to mallet directory (download from '
                        'http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip).')
    parser.add_argument('--output_file_path', '-o', type=str, required=True,
                        help='Path to output CSV file with top topics '
                        'for each tweet in input directory.')
    parser.add_argument('--model_file_path', type=str, required=True,
                        help='Path to trained LDA model file')
    parser.add_argument('--lda_mallet_prefix', type=str, required=True,
                        help='Prefix for LDA Mallet model.')
    parser.add_argument('--num_top_topics', type=int, default=3,
                        help='Number of top topics to output for each entry.'
                        ' Defaults to 3.')
    parser.add_argument('--spacy_model', '-s', type=str,
                        default='en_core_web_sm',
                        help='Name of spacy model to use.'
                        )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    with open(args.model_file_path, 'rb') as pickle_reader:
        ldamodel = pickle.load(pickle_reader)
        ldamodel.prefix = args.lda_mallet_prefix

    # load spacy nlp module
    nlp = spacy.load(args.spacy_model)

    tweet_texts = []
    tweet_docs = []
    # iterate over each subdir in input_subdir_paths
    for input_file_name in os.listdir(args.input_dir_path):

        # only read JSON files
        if not input_file_name.endswith('.json'):
            continue

        print('Processing file: %s' % input_file_name)

        input_file_path = os.path.join(args.input_dir_path, input_file_name)

        # load data from json file
        with open(input_file_path, 'r') as file_reader:
            data = json.load(file_reader)

        # iterate over each tweet in json, and form ngram frequency dictionary
        for tweet in data['results']:
            if 'extended_tweet' in tweet:
                tweet_text = tweet['extended_tweet']['full_text']
            else:
                tweet_text = tweet['text']

            tweet_texts.append(tweet_text)

            doc = nlp(tweet_text.lower().strip())

            tweet_tokens = [token.lemma_ for token in doc
                            if not token.is_stop and
                            not token.is_punct and
                            not token.is_space]

            tweet_docs.append(tweet_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(tweet_docs)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(doc) for doc in tweet_docs]

    with open(args.output_file_path, 'w', encoding='utf-8',
              errors='ignore') as file_writer:

        csv_writer = csv.writer(file_writer)

        header_row = ['tweet']

        # write header with topic IDs and scores
        for t in range(args.num_top_topics):
            header_row.extend(['topic_' + str(t+1),
                               'topic_' + str(t+1) + '_score'])
        csv_writer.writerow(header_row)

        for i, topic_row in enumerate(ldamodel[corpus]):
            topic_row = sorted(topic_row, key=lambda x: (x[1]),
                               reverse=True)

            output_row = [tweet_texts[i]]
            for t in range(args.num_top_topics):
                topic_num, topic_score = topic_row[t]
                output_row.extend([topic_num, format(topic_score, '.2f')])

            csv_writer.writerow(output_row)
