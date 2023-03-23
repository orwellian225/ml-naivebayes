import numpy as np
from binary_nb import NaiveBayesModel

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
    for data_piece in data:
        encoded_data.append({
            'event': data_piece['result'],
            'features': np.zeros(len(encoding_format))
        })
        
        for word in data_piece['words']:
            if word in encoding_format:
                index = np.where(encoding_format == word)
                encoded_data[-1]['features'][index] = 1
            else:
                encoding_format.append(word)
                encoded_data[-1]['features'][-1] = 1

        # print(data_piece)
        # print(encoded_data[-1])

    return encoded_data

def main():
    values = read_file('simple-food-reviews.txt')
    [training_data, test_data] = select_test_data(values)

    encoding_format = build_encoding_format(training_data)
    training_data = encode_data(encoding_format, training_data)

    nbm = NaiveBayesModel()
    nbm.train(training_data)

if __name__ == "__main__":
    main()
