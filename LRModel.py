import numpy as np
import matplotlib.pyplot as plt

class LRModel:
	def __init__(self, alpha=0.001, max_iter=100000):
		self.weight = -2
		self.bias = 500000
		self.alpha = alpha
		self.max_iter = max_iter

	def estimatePrice(self, mileage):
		return self.weight * mileage + self.bias

	def trainModel(self, x, y):
		for _ in range(self.max_iter):
			self.weight, self.bias = self.step_gradient(x, y)

	def cost_func(self, x, y):
		cost = self.predict(x) - y
		return (cost.T @ cost) / (cost.size * 2)

	def step_gradient(self, x, y):
		for _ in range(x.size):
			pass

	def predict(self, x):
		x_prime = np.insert(x, 0, 1, 1)
		return  np.insert(x, 0, 1, 1) @ np.array([[self.bias], [self.weight]])

	def loss_elem(self, x, y):
		pass

	def loss(self, x, y):
		x_prime = np.insert(x, 0, 1, 1)
		print(x_prime.shape)
		return x_prime.T @ (x_prime @ np.array([[self.bias], [self.weight]])) / (x_prime.shape[0] * 2)

	def plot(self, x, y):
		plt.plot(x, y, 'o')
		plt.plot(x, self.predict(x))
		plt.show()