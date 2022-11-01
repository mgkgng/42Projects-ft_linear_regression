from LRModel import LRModel

if __name__=="__main__":
	xx = LRModel()
	while True:
		s = input("Enter the mileage to get the estimate the price of your car, or insert QUIT\n>>> ")
		if s == "QUIT":
			exit()
		elif not s.isdigit():
			print("Wrong input")
		else:
			mileage = int(s)
			print("The estimated price is ${}".format(xx.estimatePrice(mileage)))
			xx.trainModel()