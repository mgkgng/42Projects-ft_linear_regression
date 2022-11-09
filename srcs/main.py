import pandas
import numpy as np
from LRModel import LRModel

def zscore(x):
	vfunc = np.vectorize(lambda e : (e - np.mean(x)) / np.std(x))
	return vfunc(x)

if __name__=="__main__":
	data = np.array(pandas.read_csv('../assets/data.csv'))
	X = data[:,0].reshape(-1, 1)
	mean_data, std_data = np.mean(X), np.std(X)
	Y = data[:,1].reshape(-1, 1)

	lr = LRModel()
	lr.trainModel(zscore(X), Y)

	while True:
		s = input("Enter the mileage to get the estimate the price of your car, or insert QUIT\n>>> ")
		if s == "QUIT":
			exit()
		elif not s.isdigit():
			print("Wrong input")
		else:
			mileage = int(s)
			std_val = (mileage - mean_data) / std_data
			estimation = lr.estimatePrice(std_val)
			print(f"The estimated price is ${estimation if estimation > 0 else 0}." + (" Sorry!" if estimation <= 0 else ""))