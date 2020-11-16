"""Get n-gram (uni, bi and trigram) frequencies for all tweets in input directory or file.
"""
import json
import os
import spacy
import csv
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser('Get n-gram (uni, bi and trigram) frequencies '
                            'for all tweets in input directory or file.')
    parser.add_argument('--input_path', '-i', type=str,
                        help='Path to input directory or file containing '
                        'tweets in JSON format.'
    parser.add_argument('--output_dir_path', '-o', type=str,
                        help='Path to output directory containing CSV files '
                        'with n-gram frequencies (one each for uni, bi and '
                        'trigrams)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # get input arguments from command line
    input_path = args.input_path
    output_path = args.output_dir_path

    # load spacy nlp module
    nlp = spacy.load('en_core_web_sm')

    # check if input path is a directory or list
    input_file_paths = []
    # if input path is a directory, make a list of all files inside it
    if os.path.isdir(input_path):
        input_file_paths = [os.path.join(input_path, f) 
                            for f in os.listdir(input_path)]
    # if input path is not a directory, just get the single input file path
    else:
        input_file_paths = [input_path]

    # initialize ngram frequency dictionaries
    unigram_freq = {}
    bigram_freq = {}
    trigram_freq = {}

    # iterate over each file in input_file_paths
    for input_file_path in input_file_paths:

        print('Processing file: %s' % input_file_path)

        # load data from json file
        with open(input_file_path, 'r') as file_reader:
            data = json.load(file_reader)

        # iterate over each tweet in json, and form ngram frequency dictionary
        for tweet in data['results']:
            if 'extended_tweet' in tweet:
                tweet_text = tweet['extended_tweet']['full_text']
            else:
                tweet_text = tweet['text']

            doc = nlp(tweet_text.strip())
            word_list = [w for w in doc]

            for i in range(len(word_list)):
                # check if current token is a stopword, punctuation or space 
                # character
                if word_list[i].is_stop or word_list[i].is_punct or \
                    word_list[i].is_space:
                    continue

                # get unigram text
                unigram_text = word_list[i].text

                # update unigram frequency dictionary
                if unigram_text not in unigram_freq:
                    unigram_freq[unigram_text] = 1
                else:
                    unigram_freq[unigram_text] = unigram_freq[unigram_text]+1

                # check if next word exists (for calculating bigram frequency)
                if i < len(word_list) - 1:
                    # get bigram text
                    bigram_text = unigram_text + '__' + word_list[i+1].text

                    # update bigram frequency dictionary
                    if bigram_text not in bigram_freq:
                        bigram_freq[bigram_text] = 1
                    else:
                        bigram_freq[bigram_text] = bigram_freq[bigram_text]+1

                    # check if next two words exist (for calculating trigram 
                    # frequency)
                    if i < len(word_list) - 2:
                        # get trigram text
                        trigram_text = bigram_text + '__' + word_list[i+2].text

                        # update trigram frequency dictionary
                        if trigram_text not in trigram_freq:
                            trigram_freq[trigram_text] = 1
                        else:
                            trigram_freq[trigram_text] = \
                                trigram_freq[trigram_text]+1

    unigram_freq_list = sorted(unigram_freq.items(), key=lambda k: k[1], 
                               reverse=True)
    bigram_freq_list = sorted(bigram_freq.items(), key=lambda k: k[1], 
                              reverse=True)
    trigram_freq_list = sorted(trigram_freq.items(), key=lambda k: k[1], 
                               reverse=True)

    # write frequencies to output_path with name 
    # "<input_name>_<ngram_n>-gram.txt"
    # unigram
    output_file_name = 'unigram.csv'
    output_file_path = os.path.join(output_path, output_file_name)
    with open(output_file_path, 'w') as file_writer:
        csv_writer = csv.DictWriter(file_writer, 
                                    fieldnames=['unigram', 'frequency'])
        csv_writer.writerows(unigram_freq_list)

    # bigram
    output_file_name = 'bigram.csv'
    output_file_path = os.path.join(output_path, output_file_name)
    with open(output_file_path, 'w') as file_writer:
        csv_writer = csv.DictWriter(file_writer, 
                                    fieldnames=['bigram', 'frequency'])
        csv_writer.writerows(bigram_freq_list)

    # trigram
    output_file_name = 'trigram.csv'
    output_file_path = os.path.join(output_path, output_file_name)
    with open(output_file_path, 'w') as file_writer:
        csv_writer = csv.DictWriter(file_writer, 
                                    fieldnames=['trigram', 'frequency'])
        csv_writer.writerows(trigram_freq_list)
