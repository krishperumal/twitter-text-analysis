"""Get dominant LDA topic distribution for each JSON file in input directory
with tweets.
"""
import json
import os
import csv
import spacy
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser('Get dominant LDA topic distribution for each '
                            'JSON file in input directory with tweets.')
    parser.add_argument('--input_dir_path', '-i', type=str,
                        help='Path to input directory containing JSON files '
                        'with tweets.')
    parser.add_argument('--model_path', '-m', type=str,
                        help='Path to LDA trained model file.')
    parser.add_argument('--model_num_topics', '-t', type=int,
                        help='Number of topics in trained LDA model.')
    parser.add_argument('--output_file_path', '-o', type=str,
                        help='Path to output CSV file containing dominant '
                        'topic information.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.isdir(args.input_dir_path):
        print('Input path must be a directory.')

    # load spacy nlp module
    nlp = spacy.load('en_core_web_sm')

    # load LDA model
    ldamodel = LdaModel.load(args.model_path)

    with open(args.output_file_path, 'w', encoding='utf-8') as file_writer:

        csv_writer = csv.DictWriter(file_writer,
                                    fieldnames=['filename'] +
                                    ['topic_' + str(t)
                                     for t in range(args.model_num_topics)])
        csv_writer.writeheader()

        # iterate over each subdir in input_subdir_paths
        for input_file_name in os.listdir(args.input_dir_path):

            # only read JSON files
            if not input_file_name.endswith('.json'):
                continue

            print('Processing file: %s' % input_file_name)

            input_file_path = os.path.join(
                args.input_dir_path, input_file_name)

            # load data from json file
            with open(input_file_path, 'r') as file_reader:
                data = json.load(file_reader)

            tweet_docs = []
            # iterate over each tweet in json, and form ngram frequency dictionary
            for tweet in data['results']:
                if 'extended_tweet' in tweet:
                    tweet_text = tweet['extended_tweet']['full_text']
                else:
                    tweet_text = tweet['text']

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

            # Get dominant topic in each document
            dominant_topic_dist = {}
            for i, row in enumerate(ldamodel[corpus]):
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                # Get the most dominant topic
                topic_num, _ = row[0]
                if topic_num in dominant_topic_dist:
                    dominant_topic_dist[topic_num] += 1
                else:
                    dominant_topic_dist[topic_num] = 1

            topic_freq_list = []
            for t in range(args.model_num_topics):
                freq = 0
                if t in dominant_topic_dist:
                    freq = dominant_topic_dist[t]

                topic_freq_list.append(freq)

            csv_writer.writerow([input_file_name] + topic_freq_list)
