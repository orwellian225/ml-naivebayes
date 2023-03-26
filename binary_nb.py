class NBClass:
    def __init__(self, id):
        self.id = id
        self.count = 1 # a class will only be initialized when it is found, therefore for a class to exist there must be at least one
        self.feature_count = {}
        self.prob = 0
        self.feature_prob = {}

    def inc_count(self, num):
        self.count += num

    def inc_feature_count(self, feature_id, num):
        if feature_id not in self.feature_count.keys():
            self.feature_count[feature_id] = num
        else:
            self.feature_count[feature_id] += num

    def calc_prob(self, n_events, smoothing_enabled):
        self.prob = self.count / n_events
        for feature in self.feature_count.keys():
            self.feature_prob[feature] = [0, 0]

            self.feature_prob[feature][1] = self.feature_count[feature] / self.count
            self.feature_prob[feature][0] = 1 - self.feature_prob[feature][1]

            # if smoothing_enabled and self.feature_prob[feature][1] == 1:
            #     self.feature_prob[feature][1] = (self.feature_count[feature] + 1) / (self.count + 2 * 1)
            #     self.feature_prob[feature][0] = 1 - self.feature_prob[feature][1]

    def to_string_count(self):
        return f"{self.id} -> count: {self.count}, feature_counts: {self.feature_count}"

    def to_string_prob(self):
        return f"{self.id} -> prob: {self.prob}, feature_counts: {self.feature_prob}"

class NaiveBayesModel:

    def print_prob(self):
        for nbclass in self.classes.keys():
            print(self.classes[nbclass].to_string_prob())

    def print_count(self):
        for nbclass in self.classes.keys():
            print(self.classes[nbclass].to_string_count())

    def train(self, data, smoothing_enabled):
        # Data Structure
        # {
        #   class: 0,
        #   features: {
        #     'feature_id': 0, 
        #     'feature_id': 0, 
        #       ...
        #   }
        # }

        self.classes = {}
        self.n_events = len(data)
        for data_piece in data:
            current_class = data_piece['class']
            # Event ==================================================== 
            if current_class not in self.classes.keys():
                self.classes[current_class] = NBClass(current_class)
            else:
                self.classes[data_piece['class']].inc_count(1)
            # End Event ==================================================== 

            # Feature ==================================================== 
            features = data_piece['features']
            for feature in features.keys():
                self.classes[current_class].inc_feature_count(feature, features[feature])
                # features[feature] will add 1 to count if it has a value of 1, else it will not increment the count
            # End Feature ==================================================== 
            self.classes[current_class].calc_prob(self.n_events, smoothing_enabled)


    def classify(self, features):
        results = {}
        for nbclass in self.classes.keys():
            results[nbclass] = 0

            numerator = 1
            denominator = 0

            for feature in features.keys():
                numerator *= self.classes[nbclass].feature_prob[feature][features[feature]]

                product = 1
                for second_nbclass in self.classes.keys():
                    product *= self.classes[second_nbclass].prob * self.classes[second_nbclass].feature_prob[feature][features[feature]]
                denominator += product

            numerator *= self.classes[nbclass].prob
            results[nbclass] = numerator / denominator

        max = {
            'class': '',
            'confidence': -1
        }

        print(results)
        for nbclass in results.keys():
            if results[nbclass] > max['confidence']:
                max['class'] = nbclass
                max['confidence'] = results[nbclass]

        return max
            


    def test_model(self, test_data):
        result = ""
        for data in test_data:
            classification = self.classify(data['features'])
            result += f"actual class: {data['class']}, guessed class: {classification['class']}, confidence: {classification['confidence']}\n"

        return result

