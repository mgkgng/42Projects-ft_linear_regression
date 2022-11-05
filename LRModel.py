import numpy as np
import matplotlib.pyplot as plt

class LRModel:
	def __init__(self, alpha=0.001, max_iter=1000):
		self.thetas = np.zeros(2, 1)
		print(self.thetas)
		self.alpha = alpha
		self.max_iter = max_iter
		self.x = []
		self.y_hat = []

	def estimatePrice(self, mileage):
		np.array(self.x, mileage)
		np.array(self.y_hat, mileage * self.thetas[1] + self.thetas[0])
		return self.y_hat[-1]

	def fit(self, y):
		x = np.insert(self.x, 0, 1, 1)
		for n in range(self.max_iter):
			res_gradient = np.array(x.T @ (x @ self.thetas - y) / y.shape[0])
			self.thetas = self.thetas - self.alpha * res_gradient

	def plot(self):
		plt.show()