import numpy as np
import matplotlib.pyplot as plt
import math

class LRModel:
	def __init__(self, alpha=0.001, max_iter=100000):
		self.thetas = np.array([[0.], [0.]])
		self.alpha = alpha
		self.max_iter = max_iter

	def estimatePrice(self, mileage):
		return math.ceil(round((self.thetas[1] * mileage + self.thetas[0]).item(0)) / 5) * 5

	def trainModel(self, x, y):
		for _ in range(self.max_iter):
			grad = self.loss(x, y)
			self.thetas = self.thetas - grad

	def loss(self, x, y):
		x_prime = np.insert(x, 0, 1, 1)
		y_hat = x_prime @ self.thetas
		cost = y_hat - y
		return (x_prime.T @ cost) / (x_prime.shape[0] * 2)

	def plot(self, x, y):
		plt.plot(x, y, 'o')
		plt.plot(x, self.predict(x))
		plt.show()