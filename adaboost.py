import numpy as np
import math

class adaBoost:

	def __init__(self):
		self.data = np.matrix([])
		self.labels = np.matrix([])
		self.classifierArray = np.matrix([])

	def loadData(self):
		self.data = np.matrix('[1. 2.1; 2. 1.1; 1.3 1.; 1. 1.; 2. 1.]')
		self.labels = [1,1,-1,-1,1]

	def guessClass(self,data,dim,threshold,inequality):

		# get dimensions and make a return vector
		n,m = np.shape(data)
		classed = np.ones((n))

		# if a data doesn't meet threshold make it -1
		if inequality == '<=':
			for point in range(0,n):
				if data[point,dim] <= threshold:
					classed[point] = -1
		else:
			for point in range(0,n):
				if data[point,dim] > threshold:
					classed[point] = -1
		return classed

	def trainClassifier(self,data,labels,weights,steps):

		# setup
		dataMatrix = np.matrix(data)
		labelMatrix = np.matrix(labels).T
		n,m = np.shape(data)
		bestClassifier = {}
		bestClassGuess = np.zeros((n,1))
		minError = float('inf')

		# the work
		for dim in range(0,m):
			# find min and max of x or y coordinates
			rangeMin = dataMatrix[:,dim].min()
			rangeMax = dataMatrix[:,dim].max()
			stepSize = (rangeMax - rangeMin) / float(steps)

			# for every little subdivision of the range
			for j in range(-1,steps + 1):
				for inequality in ['<=','>']:

					# set the threshold value by incrementing one step over minrange
					# and classify based on the threshold
					threshold = (rangeMin + (j * stepSize))
					classGuess = self.guessClass(dataMatrix,dim,threshold,inequality)

					# find the error in predictedClasses
					errorArray = np.ones((n,1))

					# if wrong prediction, set to 0
					for point in range(0,n):
						if classGuess[point] == labelMatrix[point]:
							errorArray[point] = 0
					# print "WEIGHTS"
					# print np.matrix(weights)
					# print "ERRORARRAY"
					# print np.matrix(errorArray)

					weightedError = np.matrix(weights).T * np.matrix(errorArray)

					# print "WEIGHTED ERROR"
					# print weightedError

					# if weighted error is smallest, then put all our current 
					# stuff in a dictionary
					if weightedError < minError:
						minError = weightedError
						# print "MIN ERROR"
						# print minError
						bestClassGuess = classGuess.copy()
						bestClassifier['dim'] = dim
						bestClassifier['threshold'] = threshold
						bestClassifier['inequality'] = inequality
		# print "BEST CLASS GUESS"
		# print bestClassGuess
		# print bestClassifier
		# print minError
		return bestClassifier,minError,bestClassGuess

	def boost(self,maxFeatures):
		weakClassGuessers = []
		n,m = np.shape(self.data)

		# setup weight vector
		weights = np.ones((n,1))
		weights = weights * (1. / n)

		aggregateClassGuess = np.zeros((n,1))

		# the work
		for i in range (0,maxFeatures):
			
			# print("ITERATION", i)

			# train best classifier for these weights
			bestClassifier,error,classGuess = self.trainClassifier(self.data,self.labels,weights,10)

			# calculate weight of the classifier
			alpha = float(math.log(1.0 - error) / max(error,1e-16))
			bestClassifier['alpha'] = alpha

			# add classifier to weakClassGuess
			weakClassGuessers.append(bestClassifier)

			# calculate new weights
			exponent = np.multiply(1 * alpha * np.matrix(self.labels), classGuess)
			# print "EXPONENT VECTOR"
			# print np.exp(exponent.T)
			# print "ORIGINAL WEIGHTS"
			# print weights
			weights = np.multiply(weights,np.exp(exponent.T))
			weights = weights * (1 / weights.sum())
			# print "AFTER WEIGHTS"
			# print weights
			# update aggregateClassGuess
			aggregateClassGuess = aggregateClassGuess + np.matrix((-1 * alpha * classGuess)).T
			# print "CLASS GUESS"
			# print aggregateClassGuess

			# aggregateErrors
			aggregateErrors = np.multiply(np.sign(aggregateClassGuess) != np.matrix(self.labels).T, np.ones((n,1)))
			errorRate = aggregateErrors.sum() / n
			# print "total error",errorRate,"\n"
			if errorRate == 0.0: break

		self.classifierArray = weakClassGuessers

	def classify(self,data):
		dataMatrix = np.matrix(data)
		n,m = np.shape(dataMatrix)
		aggregateClassGuess = np.matrix(np.zeros((n,1)))

		# for every classifier we train, use it to classguess and then scale by
		# alpha and add to aggregate guess
		for i in range (0,len(self.classifierArray)):
			classGuess = self.guessClass(dataMatrix,self.classifierArray[i]['dim'],self.classifierArray[i]['threshold'],self.classifierArray[i]['inequality'])
			aggregateClassGuess = aggregateClassGuess + (-1 * self.classifierArray[i]['alpha'] * classGuess)
			# print aggregateClassGuess
		return np.sign(aggregateClassGuess)


adabooster = adaBoost()
adabooster.loadData()
adabooster.boost(30)
print adabooster.classify([0,0])
