import numpy as np
from binary_nb import BinaryNBModel

def read_file(path):
    values = []

    with open(path, "r") as file:
        line = file.readline()
        while line:

            stripped = line.strip()
            split = stripped.split()

            sentiment_result = split[0]
            words = split[1:]

            values.append(
                {
                    'result': sentiment_result,
                    'words': words
                }
            )

            line = file.readline()

    return values

def select_test_data(data):
    test = []
    training = []
    data = np.array(data)
    np.random.shuffle(data)

    for i in range(int(len(data) * 2 / 3)):
        training.append(data[i])

    for i in range(int(len(data) * 2 / 3) + 1, len(data)):
        test.append(data[i])

    return [training, test]

def build_encoding_format(data):
    encoding_format = np.array([])
    for data_piece in data:
        encoding_format = np.append(encoding_format, np.array(data_piece['words']))

    encoding_format = np.unique(encoding_format, False)
    return encoding_format


def encode_data(encoding_format, data):
    encoded_data = []
    for i in range(len(data)):
        encoded_data.append({ 'identifier': data[i]['result'], 'features': [] })

        for j in range(len(encoding_format)):
            encoded_data[-1]['features'].append(0)
        for word in data[i]['words']:
            if word in encoding_format:
                index = np.where(encoding_format == word)[0][0]
                encoded_data[-1]['features'][index] = 1

    grouped_data = {}
    for d in encoded_data:
        if d['identifier'] in grouped_data.keys():
            grouped_data[d['identifier']].append(d['features'])
        else:
            grouped_data[d['identifier']] = [d['features']]

    return grouped_data

def main():
    values = read_file('simple-food-reviews.txt')
    [training_data, test_data] = select_test_data(values)

    encoding_format = build_encoding_format(training_data)
    training_data = encode_data(encoding_format, training_data)
    test_data = encode_data(encoding_format, test_data)

    BinaryNBModel.execute(training_data, test_data, print_results=True)

if __name__ == "__main__":
    main()