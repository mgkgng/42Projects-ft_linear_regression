import pandas
import numpy as np
from LRModel import LRModel

def zscore(x):
	print("pardon?", np.mean(x), np.std(x))
	vfunc = np.vectorize(lambda e : (e - np.mean(x)) / np.std(x))
	return vfunc(x)

def minmax(x):
	min_elem = np.min(x)
	max_elem = np.max(x)
	vfunc = np.vectorize(lambda x : (x - min_elem) / (max_elem - min_elem))
	return vfunc(x)

if __name__=="__main__":
	# data = np.array(pandas.read_csv('data.csv'))
	data = np.array(pandas.read_csv('data.csv'))
	# zdata = zscore(data)
	print(data)
	print(np.mean(data))
	print(np.std(data))
	X = data[:,0].reshape(-1, 1)
	zX = zscore(data[:,0].reshape(-1, 1))
	mean_data = np.mean(X)
	std_data = np.std(X)
	Y = data[:,1].reshape(-1, 1)
	lr = LRModel()
	print(lr.loss(X, Y))
	lr.trainModel(zX, Y)
	print(lr.thetas)
	std_val = (80000 - mean_data) / std_data
	print("sorry?", lr.estimatePrice(std_val))
	lr.plot(zX, Y)
	# while True:
	# 	s = input("Enter the mileage to get the estimate the price of your car, or insert QUIT\n>>> ")
	# 	if s == "QUIT":
	# 		exit()
	# 	elif not s.isdigit():
	# 		print("Wrong input")
	# 	else:
	# 		mileage = int(s)
	# 		print("The estimated price is ${}".format(lr.estimatePrice(mileage)))
	# 		print(lr.thetas)
	# 		lr.plot(X, Y)
