class BNBClass:
	def __init__(self, identifier, count):
		self.identifier = identifier
		self.count = count
		self.features = []

class BinaryNBModel:
	def __init__(self) -> None:
		self.classes = []
		self.count = 0

	def execute(training_data, test_data):
		nbm = BinaryNBModel()
		nbm.train_model(training_data)
		test_results_unsmoothed = nbm.classify_data(test_data, 0)
		test_results_smoothed = nbm.classify_data(test_data, 1)

		print(BinaryNBModel.generate_confusion_matrix(test_results_unsmoothed))
		print(BinaryNBModel.generate_confusion_matrix(test_results_smoothed))

		#print(nbm.to_string())  

		print("\nUnsmoothed")
		print(BinaryNBModel.generate_confusion_matrix(test_results_unsmoothed), "\n")
		#for test_result in test_results_unsmoothed:
		#    print(f"actual class = {test_result['actual_class']}, generated class = {test_result['generated_class']}, probability = {test_result['probability']}")

		print("\nSmoothed")
		print(BinaryNBModel.generate_confusion_matrix(test_results_smoothed), "\n")
		#for test_result in test_results_smoothed:
		#    print(f"actual class = {test_result['actual_class']}, generated class = {test_result['generated_class']}, probability = {test_result['probability']}")

	def to_string(self):
		result = f"n = {self.count}\n"

		for nbclass in self.classes:
			result += f"\tn({nbclass.identifier}) = {nbclass.count}\n"
			for j in range(len(nbclass.features)):
				result += f"\t\tFeature {j}: "
				for type in nbclass.features[j]:
					result += f"n({nbclass.features[j].index(type)}) = {type} "
				result += "\n"

		return result

	def generate_confusion_matrix(result_data):
		matrix = {}

		for result in result_data:
			if result['actual_class'] in matrix.keys():
				if result['generated_class'] in matrix[result['actual_class']].keys():
					matrix[result['actual_class']][result['generated_class']] += 1
				else:
					matrix[result['actual_class']][result['generated_class']] = 1
			else:
				matrix[result['actual_class']] = {}
				matrix[result['actual_class']][result['generated_class']] = 1

		return matrix

	def classify_data(self, data, smoothing_constant):

		# Data format:
		# [{
		#	'class_identifier': []
		# }, ...]

		results = []
		for identifier in data.keys():
			for features in data[identifier]:
				classification = self.classify_features(features, smoothing_constant)
				results.append({
					'generated_class': classification['class'],
					'probability': classification['probability'],
					'actual_class': identifier,
					'features': data[identifier]
				})
		return results

	def classify_features(self, features, smoothing_constant):
		results = []
		for nbclass in self.classes:
			results.append({
				'class': nbclass.identifier,
				'probability': 0
			})

			denominator = 0
			for nbcI in self.classes:
				product = 1
				for j in range(len(features)):
					# laplace smoothing goes here
					alpha = smoothing_constant

					current_prob = (nbcI.features[j][features[j]]) / (nbcI.count)
					if current_prob == 0:
						current_prob = (nbcI.features[j][features[j]] + alpha) / (nbcI.count + alpha * len(nbcI.features[j]))
					product *= current_prob
				denominator += product * (nbcI.count / self.count)

			product = 1
			for j in range(len(features)):
				product *= (nbclass.features[j][features[j]]) / (nbclass.count)
			numerator = product * (nbclass.count / self.count)

			results[-1]['probability'] = numerator / denominator if denominator != 0 else 0.0 

		max_class = {
			'class': '',
			'probability': -100
		}
		for result in results:
			if result['probability'] > max_class['probability']:
				max_class = result

		return max_class

	def train_model(self, data):

		# Data format:
		# [{
		#	'class_identifier': [[...]]
		# }]
		for key in data.keys():
			# Construct Class data
			self.count += len(data[key])
			self.classes.append(BNBClass(identifier=key, count=len(data[key])))

			# Construct Feature given Class data
			current_class = self.classes[-1]
			for features in data[key]:
				for j in range(len(features)):
					try:
						current_class.features[j][features[j]] += 1
					except IndexError:
						current_class.features.append([0,0])
						current_class.features[-1][features[j]] += 1
