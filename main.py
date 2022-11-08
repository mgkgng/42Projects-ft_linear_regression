import pandas
import numpy as np
from LRModel import LRModel

if __name__=="__main__":
	data = np.array(pandas.read_csv('data.csv'))
	X = data[:,0].reshape(-1, 1)
	Y = data[:,1].reshape(-1, 1)
	xx = LRModel()
	xx.trainModel(X, Y)
	while True:
		s = input("Enter the mileage to get the estimate the price of your car, or insert QUIT\n>>> ")
		if s == "QUIT":
			exit()
		elif not s.isdigit():
			print("Wrong input")
		else:
			mileage = int(s)
			print("The estimated price is ${}".format(xx.estimatePrice(mileage)))
