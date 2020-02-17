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

    if len(sys.argv) < 3:
        print('Incorrect number of arguments provided.')
        print('Provide the following: <lda_model_file_path> <output_file_path>')
        sys.exit(0)

    # get input arguments from command line
    model_path = sys.argv[1]
    output_path = sys.argv[2]

    # load LDA model
    ldamodel = LdaModel.load(model_path)

    topics = ldamodel.print_topics(num_topics=TOTAL_NUM_TOPICS, num_words=50)

    with open(output_path, 'w') as file_writer:

        for t in topics:
            file_writer.write(str(t) + '\n')
