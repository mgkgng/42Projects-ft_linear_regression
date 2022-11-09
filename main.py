import pandas
import numpy as np
from LRModel import LRModel

def zscore(x):
	vfunc = np.vectorize(lambda e : (e - np.mean(x)) / np.std(x))
	return vfunc(x)

if __name__=="__main__":
	data = np.array(pandas.read_csv('data.csv'))
	X = data[:,0].reshape(-1, 1)
	Y = data[:,1].reshape(-1, 1)
	lr = LRModel()
	print(lr.loss(X, Y))
	# lr.trainModel(zscore(X), zscore(Y))
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
