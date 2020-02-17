import sys
import json
import os
import spacy
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Incorrect number of arguments provided.')
        print('Provide the following: <input_dir_path> <output_dir_path>')
        sys.exit(0)

    # get input arguments from command line
    input_path = sys.argv[1]
    if not os.path.isdir(input_path):
        print('Input path must be a directory.')
    output_path = sys.argv[2]
    if not os.path.isdir(output_path):
        print('Output path must be a directory.')

    # load spacy nlp module
    nlp = spacy.load('en_core_web_sm')

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
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word=dictionary, passes=20)

    # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    mallet_path = '/Users/krishperumal/models/mallet-2.0.8/bin/mallet'  # update this path

    ldamodel_list = []
    output_file_path = os.path.join(output_path, 'lda_hyperparam_output.txt')
    with open(output_file_path, 'w') as file_writer:

        for num_topics in range(35, 50+1, 5):

            print('Running LDA with %2d topics' % num_topics)

            ldamodel = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics,
                                                        id2word=dictionary)
            ldamodel.save(os.path.join(output_path, 'lda_model_n%02d' % num_topics))

            topics = ldamodel.print_topics(num_topics=num_topics, num_words=20)
            file_writer.write('=================='
                              '\nLDA with %2d topics\n'
                              '==================\n'
                              % num_topics)
            for t in topics:
                file_writer.write(str(t)+'\n')

            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=ldamodel, texts=tweet_docs, dictionary=dictionary,
                                                 coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            file_writer.write('\nCoherence Score: %.2f\n' % coherence_lda)

            ldamodel_list.append((ldamodel, coherence_lda))
