import numpy as np
import random as rand

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
        

def main():
    values = read_file('simple-food-reviews.txt')
    [training_data, test_data] = select_test_data(values)

    print(training_data)
    print("")
    print(test_data)


if __name__ == "__main__":
    main()
