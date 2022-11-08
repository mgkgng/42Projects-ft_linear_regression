import numpy as np

class LRModel:
	def __init__(self, alpha=0.001, max_iter=1000):
		self.thetas = [0, 0]
		self.alpha = alpha
		self.max_iter = max_iter

	def estimatePrice(self, mileage):
		return self.weight * mileage + self.bias

	def trainModel(self, x, y):
		x_prime = np.insert(x, 0, 1, 1)
		for n in range(self.max_iter):
			res = (x_prime @ (x_prime @ self.thetas - y)) / y.shape[0]
			self.thetas = self.thetas - self.alpha * res
