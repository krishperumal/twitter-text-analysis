"""Train LDA for all tweets in JSON format in input directory.
"""
import json
import os
import spacy
import pickle
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser('Train LDA for all tweets in JSON format in input '
                            'directory.')
    parser.add_argument('--input_dir_path', '-i', type=str,
                        help='Path to input directory containing JSON files '
                        'with tweets.')
    parser.add_argument('--mallet_path', '-m', type=str,
                        help='Path to mallet directory (download from '
                        'http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip).')
    parser.add_argument('--min_topics', type=int,
                        help='Min number of topics to train LDA model.')
    parser.add_argument('--max_topics', type=int,
                        help='Max number of topics to train LDA model.')
    parser.add_argument('--topic_num_interval', type=int,
                        help='Interval of number of topics to train LDA '
                        'model.')
    parser.add_argument('--output_dir_path', '-o', type=str,
                        help='Path to output directory containing trained '
                        'LDA model files.')
    parser.add_argument('--spacy_model', '-s', type=str,
                        default='en_core_web_sm',
                        help='Name of spacy model to use.'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # get input arguments from command line
    input_path = args.input_dir_path
    if not os.path.isdir(input_path):
        print('Input path must be a directory.')
    output_path = args.output_dir_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # load spacy nlp module
    nlp = spacy.load(args.spacy_model)

    tweet_docs = []
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

    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20,
    # id2word=dictionary, passes=20)
    mallet_path = args.mallet_path

    ldamodel_list = []
    output_file_path = os.path.join(output_path, 'lda_hyperparam_output.txt')
    with open(output_file_path, 'w') as file_writer:

        for num_topics in range(args.min_topics, args.max_topics+1,
                                args.topic_num_interval):

            print('Running LDA with %2d topics' % num_topics)

            ldamodel = gensim.models.wrappers.LdaMallet(mallet_path,
                                                        corpus=corpus,
                                                        num_topics=num_topics,
                                                        id2word=dictionary,
                                                        prefix=os.path.join(output_path, 'lda_model_n' + str(num_topics)))

            # ldamodel.save(os.path.join(output_path,
            #                            'lda_model_n%02d' % num_topics))

            with open(os.path.join(output_path,
                                   'lda_model_n%02d.pkl' % num_topics), 'wb') \
                    as pickle_writer:
                pickle.dump(ldamodel, pickle_writer)

            topics = ldamodel.print_topics(num_topics=num_topics, num_words=20)
            file_writer.write('=================='
                              '\nLDA with %2d topics\n'
                              '==================\n'
                              % num_topics)
            for t in topics:
                file_writer.write(str(t)+'\n')

            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=ldamodel,
                                                 texts=tweet_docs,
                                                 dictionary=dictionary,
                                                 coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            file_writer.write('\nCoherence Score: %.2f\n' % coherence_lda)

            ldamodel_list.append((ldamodel, coherence_lda))
