import sys
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer


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

    # tfidf score will not penalize stopwords when number of documents is small; so, in our case, since the number of
    # documents is only 10 (one for each week), we filter our terms that occur in more than 7 weeks out of 10
    vectorizer = TfidfVectorizer(max_df=0.7)

    # get all subdir paths; each subdir contains tweets for one group of tweets that are to be considered as a single
    # document for TFIDF
    input_subdir_paths = [os.path.join(input_path, d) for d in os.listdir(input_path)
                          if not d.startswith('.') and os.path.isdir(os.path.join(input_path, d))]

    tweet_docs = []
    # iterate over each subdir in input_subdir_paths
    for input_subdir_path in input_subdir_paths:

        print('Processing dir: %s' % input_subdir_path)

        tweet_doc = ''
        for input_file_name in os.listdir(input_subdir_path):
            input_file_path = os.path.join(input_subdir_path, input_file_name)

            # load data from json file
            with open(input_file_path, 'r') as file_reader:
                data = json.load(file_reader)

            # iterate over each tweet in json, and form ngram frequency dictionary
            for tweet in data['results']:
                if 'extended_tweet' in tweet:
                    tweet_text = tweet['extended_tweet']['full_text']
                else:
                    tweet_text = tweet['text']
                tweet_doc += tweet_text + ' '
        tweet_docs.append(tweet_doc)

    X = vectorizer.fit_transform(tweet_docs)
    X = X.toarray()

    feature_names = vectorizer.get_feature_names()
    feature_indices = range(len(feature_names))

    for i in range(X.shape[0]):
        scores = X[i]
        sorted_indices_scores = sorted(zip(feature_indices, scores), key=lambda k:k[1], reverse=True)

        output_file_name = os.path.basename(input_subdir_paths[i])+'_tfidf_keywords.txt'
        output_file_path = os.path.join(output_path, output_file_name)
        with open(output_file_path, 'w') as file_writer:
            for (idx, score) in sorted_indices_scores:
                file_writer.write('%s,%.2f\n' % (feature_names[idx], score))
