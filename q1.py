def read_file(path):
    values = []

    with open(path, "r") as file:
        values = file.readlines()

    for value in values:
        value = value.strip('\n')

    return values

def main():
    print(read_file('simple-food-reviews.txt'))

if __name__ == "__main__":
    main()
