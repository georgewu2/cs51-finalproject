import numpy as np
import math

# use a list of features as data??? maybe dim is features and total matrix is images?
def loadSimpleData():
	data = np.matrix('[1. 2.1; 2. 1.1; 1.3 1.; 1. 1.; 2. 1.]')
	classLabels = [1,1,-1,-1,1]
	return data,classLabels

data,classLabels = loadSimpleData()

def classify(data,dim,threshold,inequality):

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

def classifierTrainer(data,labels,weights,steps):

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
				classGuess = classify(dataMatrix,dim,threshold,inequality)

				# find the error in predictedClasses
				errorArray = np.ones((n,1))

				# if wrong prediction, set to 0
				for point in range(0,n):
					if classGuess[point] == labelMatrix[point]:
						errorArray[point] = 0
				# print "WEIGHTS"
				# print weights
				# print "ERRORARRAY"
				# print errorArray

				weightedError = np.dot(weights.T,errorArray)
				# print "WEIGHTED ERROR"
				# print weightedError

				# if weighted error is smallest, then put all our current 
				# stuff in a dictionary
				if weightedError < minError:
					minError = weightedError
					bestClassGuess = classGuess.copy()
					bestClassifier['dim'] = dim
					bestClassifier['threshold'] = threshold
					bestClassifier['inequality'] = inequality

	return bestClassifier,minError,bestClassGuess

def adaboost(data,labels,iterations):
	weakClassGuess = []
	n,m = np.shape(data)

	# setup weight vector
	weights = np.ones((n,1))
	weights = weights * (1. / n)

	aggregateClassGuess = np.zeros((n,1))

	# the work
	for i in range (0,iterations):
		
		# print("ITERATION", i)
		# train a classifier
		bestClassifier,error,classGuess = classifierTrainer(data,labels,weights,10)

		# calculate weight of the classifier
		alpha = float(0.5 * math.log(1.0 - error) / max(error,1e-16))
		bestClassifier['alpha'] = alpha

		# add classifier to weakClassGuess
		weakClassGuess.append(bestClassifier)

		# calculate new weights
		exponent = np.multiply(-1 * alpha * np.matrix(labels), classGuess)
		# print "EXPONENT VECTOR"
		# print exponent
		weights = np.multiply(weights,np.exp(exponent.T))
		weights = weights / weights.sum()

		# update aggregateClassGuess
		aggregateClassGuess = aggregateClassGuess + classGuess.T

		# aggregateErrors
		aggregateErrors = np.multiply(np.sign(aggregateClassGuess) != np.matrix(classLabels).T, np.ones((n,1)))
		errorRate = aggregateErrors.sum() / m
		print "total error",errorRate,"\n"
		if errorRate == 0.0: break

	return weakClassGuess




print adaboost(data,classLabels,10)
