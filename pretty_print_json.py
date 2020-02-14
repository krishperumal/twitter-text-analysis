import sys
import json

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file_reader, \
            open(sys.argv[2], 'w') as file_writer:
        data = json.load(file_reader)
        json.dump(data, file_writer, indent=4, sort_keys=True)
