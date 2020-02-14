import sys
import json

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file_reader:
        data = json.load(file_reader)

    i = 1
    for tweet in data['results']:
        print('==Tweet %d' % i)
        if 'extended_tweet' in tweet:
            print(tweet['extended_tweet']['full_text'])
        else:
            print(tweet['text'])
        print('==End of Tweet %d' % i)
        i += 1
