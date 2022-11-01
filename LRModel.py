class LRModel:
	def __init__(self):
		self.weight = 0
		self.bias = 0
		self.dataset = []

	def estimatePrice(self, mileage):
		self.dataset.append([mileage, self.weight * mileage + self.bias])
		return self.dataset[-1][1]

	def trainModel(self):
		pass