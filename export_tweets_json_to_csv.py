"""Export tweets in JSON file to CSV.
"""
from argparse import ArgumentParser
import json
import csv


def parse_arguments():
    parser = ArgumentParser('Export tweets in JSON file to CSV.')
    parser.add_argument('--input_file_path', '-i', type=str,
                        help='Path to input JSON file containing tweets.')
    parser.add_argument('--output_file_path', '-o', type=str,
                        help='Path to output CSV file containing tweets.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    with open(args.input_file_path, 'r', encoding='utf-8') as file_reader:
        data = json.load(file_reader)

    with open(args.output_file_path, 'w', encoding='utf-8') as file_writer:
        csv_writer = csv.DictWriter(file_writer, fieldnames=['tweet_text'])
        csv_writer.writeheader()
        
        for tweet in data['results']:
            if 'extended_tweet' in tweet:
                csv_writer.writerow(tweet['extended_tweet']['full_text']))
            else:
                csv_writer.writerow(tweet['text'])
