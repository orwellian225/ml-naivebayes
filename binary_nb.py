def feature_set_id(feature_set):
    binary_identifier = ""
    for f in feature_set:
        binary_identifier += str(int(f))

    return str(int(binary_identifier, 2))


class NaiveBayesModel:

    def train(self, data):
        # Data Structure
        # {
        #   event: 0,
        #   features: []
        # }

        self.feature_set_count = {}
        self.event_count = {}
        self.event_probs = {}
        self.n_events = 0
        self.n_feature_sets = 0
        for data_piece in data:
            # Event ==================================================== 
            self.n_events += 1
            if data_piece['event'] not in self.event_count.keys():
                self.event_count[data_piece['event']] = 1
            else:
                self.event_count[data_piece['event']] += 1
            # End Event ==================================================== 

            # Feature Sets ==================================================== 
            # Create a unique identifier from the feature set
            # This means that duplicate feature sets will just increment
            # the counter for that feature set
            f_identifier = feature_set_id(data_piece['features'])
            self.n_feature_sets += 1
            if f_identifier not in self.feature_set_count.keys():
                self.feature_set_count[f_identifier] = 1
            else:
                self.feature_set_count[f_identifier] += 1
            # End Feature Sets ==================================================== 

        for event in self.event_count.keys():
            self.event_probs[event] = self.event_count[event] / self.n_events 

        print(self.event_count)
        print(self.event_probs)

    def classify(self, features):
        p_features = self.feature_set_count[feature_set_id(features)] / self.n_feature_sets


