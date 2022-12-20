import pandas, sys
import numpy as np
from LRModel import LRModel

def unison_shuffled_copies(a, b):
	p = np.random.permutation(len(a))
	return a[p], b[p]

def data_spliter(x, y, proportion):
	shuffled = unison_shuffled_copies(x, y)
	testNb = int(x.shape[0] * proportion)
	splitX = np.vsplit(shuffled[0], [testNb, x.shape[0]])
	splitY = np.vsplit(shuffled[1], [testNb, x.shape[0]])
	return splitX[0], splitX[1], splitY[0], splitY[1]

def zscore(x):
	vfunc = np.vectorize(lambda e : (e - np.mean(x)) / np.std(x))
	return vfunc(x)

if __name__=="__main__":
	data = np.array(pandas.read_csv('../../assets/data.csv'))
	X, Y = data[:,0].reshape(-1, 1), data[:,1].reshape(-1, 1)
	trainX, testX, trainY, testY = data_spliter(X, Y, 0.8)
	mean_X, std_X = np.mean(trainX), np.std(trainX)

	lr = LRModel()
	lr.trainModel(zscore(trainX), trainY)
	zfunc = np.vectorize(lambda x : (x - mean_X) / std_X)
	precision = lr.calculatePrecision(zfunc(testX), testY)

	while True:
		s = input("Enter the mileage to get the estimate the price of your car, or insert QUIT\n>>> ")
		if s == "QUIT":
			sys.exit("Bye bye~")
		elif not s.isdigit():
			print("Wrong input")
		else:
			estimation = lr.estimatePrice((int(s) - mean_X) / std_X)
			print()
			print(f"The estimated price is ${estimation if estimation > 0 else 0}." + (" Sorry!" if estimation <= 0 else ""))
			print(f"precision rate: {precision}%")
			print()

