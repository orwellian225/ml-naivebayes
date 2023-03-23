class Event:
    def __init__(self, identifier) -> None:
        self.identifier = identifier
        self.probability = 0
        self.count = 0

    def update_count(self, k):
        self.count += k

    def update_probability(self, n):
        self.probability = self.count / n

class Feature:
    def __init__(self, identifier) -> None:
        self.identifier = identifier
        self.probability = 0

class NaiveBayesModel:
    def __init__(self) -> None:
        self.events = []
        self.features = []


    def train(self, data):
        # Data Structure
        # {
        #   event: 0,
        #   features: []
        # }

        nbm = self
        for data_piece in data:
            found_event = False
            for event in nbm.events:
                if event.identifier == data_piece['event']:
                    event.update_count(1)
                    found_event = True
                    break

            if not found_event:
                nbm.events.append(Event(data_piece['event']))
