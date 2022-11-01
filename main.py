import sys

from estimatePrice import estimatePrice
from trainModel import trainModel

if __name__=="__main__":
	weight, bias = 0
	dataset = []
	while True:
		s = input("Enter the mileage to get the estimate the price of your car, or insert QUIT\n>>> ")
		if s == "QUIT":
			exit()
		elif not s.isdigit():
			print("Wrong input")
		else:
			mileage = int(s)
			price = estimatePrice(mileage, weight, bias)
			dataset.append([mileage, price])
			[weight, bias] = trainModel(dataset, weight, bias)
			print("The estimated price is ${}".format(price))
