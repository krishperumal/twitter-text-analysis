import sys
import json
import os
import spacy
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel


TOTAL_NUM_TOPICS=30

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Incorrect number of arguments provided.')
        print('Provide the following: <lda_model_file_path> <input_dir_path> <output_file_path>')
        sys.exit(0)

    # get input arguments from command line
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    if not os.path.isdir(input_path):
        print('Input path must be a directory.')
    output_path = sys.argv[3]

    # load spacy nlp module
    nlp = spacy.load('en_core_web_sm')

    # load LDA model
    ldamodel = LdaModel.load(model_path)

    with open(output_path, 'w') as file_writer:

        # iterate over each subdir in input_subdir_paths
        for input_file_name in os.listdir(input_path):

            # only read JSON files
            if not input_file_name.endswith('.json'):
                continue

            print('Processing file: %s' % input_file_name)

            input_file_path = os.path.join(input_path, input_file_name)

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
                # Get the 5 most dominant topics
                for t in range(5):
                    topic_num, _ = row[t]
                    if topic_num in dominant_topic_dist:
                        dominant_topic_dist[topic_num] += 1
                    else:
                        dominant_topic_dist[topic_num] = 1

            topic_freq_list = []
            for t in range(TOTAL_NUM_TOPICS):
                freq = 0
                if t in dominant_topic_dist:
                    freq = dominant_topic_dist[t]

                topic_freq_list.append(freq)

            file_writer.write('%s' % input_file_name)
            for freq in topic_freq_list:
                file_writer.write(',%d' % freq)
            file_writer.write('\n')
