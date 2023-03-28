import numpy as np
from binary_nb import BinaryNBModel
import math as m

CLASS_IDENTIFIER = 64

def read_file(path):
    values = []

    with open(path, "r") as file:
        line = file.readline()
        while line:

            stripped = line.strip()
            split = stripped.split(',')

            line_class = split[CLASS_IDENTIFIER]
            line_features = [int(num_str) for num_str in split[:CLASS_IDENTIFIER]]

            values.append(
                {
                    'class': line_class,
                    'features': line_features
                }
            )

            line = file.readline()

    return values

def visualize_digit(digit_encoding):
    result = ""
    pixel_value = ["□", "■"]

    sqr_dimension = int(m.sqrt(len(digit_encoding)))

    for i in range(sqr_dimension):
        for j in range(sqr_dimension):
            result += pixel_value[int(digit_encoding[i * sqr_dimension + j])]
        result += "\n"

    return result

def format_data(data):
    result = {}
    for d in data:
        if d['class'] in result.keys():
            result[d['class']].append(d['features'])
        else:
            result[d['class']] = [d['features']]
    return result

def select_test_data(data):
    test = []
    training = []
    data = np.array(data)
    np.random.shuffle(data)

    for i in range(int(len(data) * 0.8)):
        training.append(data[i])

    for i in range(int(len(data) * 0.8) + 1, len(data)):
        test.append(data[i])

    return [training, test]

def main():
    data = read_file("./smalldigits.csv")
    [training_data, test_data] = select_test_data(data)
    training_data = format_data(training_data)
    test_data = format_data(test_data)
    BinaryNBModel.execute(training_data, test_data, print_correctness=True, print_confusion=True)

if __name__ == "__main__":
    main()