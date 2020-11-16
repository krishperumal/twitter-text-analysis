"""Write LDA topics to CSV file.
"""
from gensim.models.ldamodel import LdaModel
import csv
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser('Write LDA topics to CSV file.')
    parser.add_argument('--model_path', '-m', type=str,
                        help='Path to LDA trained model file.')
    parser.add_argument('--model_num_topics', '-t', type=str,
                        help='Number of topics in trained LDA model.')
    parser.add_argument('--output_file_path', '-o', type=str,
                        help='Path to output CSV file containing topic 
                        'information.'
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_arguments()
    
    # load LDA model
    ldamodel = LdaModel.load(args.model_path)

    topics = ldamodel.print_topics(num_topics=args.model_num_topics, 
                                   num_words=50)

    with open(args.output_file_path, 'w', encoding='utf-8') as file_writer:
        csv_writer = csv.DictWriter(file_writer, 
                                    fieldnames=['topic_id', 'topic_words'])
        csv_writer.writeheader()
        for t in range(len(topics)):
            csv_writer.writerow([t, topics[t]]
