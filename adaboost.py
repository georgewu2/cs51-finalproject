import numpy as np
import math
from collections import Counter
import Image

class adaBoost:

	def __init__(self):
		self.data 			 = np.matrix([])
		self.labels 		 = np.matrix([])
		self.classifierArray = np.matrix([])

	def get_frame_vector(self, video_frame, flatten=True):
		im = Image.open(video_frame)

		# convert to grayscale
		im_gray = im.convert('L')

		# convert to a matrix
		im_matrix = np.matrix(im_gray)
		if flatten == True:
			return im_matrix.flatten('F')
		else:
			return im_matrix

	# LOAD DATA SHOULD TAKE IN A SET?? FOR CASCADE
	def loadData(self,positiveDir="/positive",negativeDir="/negative"):

		positiveSet = []
		negativeSet = []

		for directory in (positiveDir,negativeDir):

			images = os.listdir(os.getcwd() + directory)
			
			# get rid of the .DS_Store file
			images.pop(0)

			# add each vector to the list
			for i in images:
				directory.append(self.get_frame_vector(images + i,False))



		self.data = np.vstack( [ positiveSet , negativeSet ] )
		npos,mpos = np.shape(positiveSet)
		nneg,mneg = np.shape(negativeSet)

		self.labels = ([1 for x in range(npos)])
		self.labels.extend([-1 for x in range(nneg)])

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

					weightedError = np.matrix(weights).T * np.matrix(errorArray)

					# if weighted error is smallest, then put all our current 
					# stuff in a dictionary
					if weightedError < minError:
						minError = weightedError
						bestClassGuess = classGuess.copy()
						bestClassifier['dim'] = dim
						bestClassifier['threshold'] = threshold
						bestClassifier['inequality'] = inequality

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

			print classGuess
			# calculate weight of the classifier
			alpha = float(math.log(1.0 - error) / max(error,1e-16))
			print alpha
			bestClassifier['alpha'] = alpha

			# add classifier to weakClassGuess
			weakClassGuessers.append(bestClassifier)

			# calculate new weights
			exponent = np.multiply(1 * alpha * np.matrix(self.labels), classGuess)
			weights = np.multiply(weights,np.exp(exponent.T))
			weights = weights * (1 / weights.sum())

			# update aggregateClassGuess
			aggregateClassGuess = aggregateClassGuess + np.matrix((-1 * alpha * classGuess)).T
			print np.matrix((-1 * alpha * classGuess)).T
			# print aggregateClassGuess
			# aggregateErrors
			aggregateErrors = np.multiply(np.sign(aggregateClassGuess) != np.matrix(self.labels).T, np.ones((n,1)))
			errorRate = aggregateErrors.sum() / n
			# print aggregateErrors

			if errorRate == 0.0: 
				print "NOERROR"
				break

		self.classifierArray = weakClassGuessers

	def classify(self,data):
		classifiedDict = {}
		for i in data:
			dataMatrix = np.matrix(data)
			n,m = np.shape(dataMatrix)
			aggregateClassGuess = np.matrix(np.zeros((n,1)))

			# for every classifier we train, use it to classguess and then scale by
			# alpha and add to aggregate guess
			for i in range (0,len(self.classifierArray)):
				classGuess = self.guessClass(dataMatrix,self.classifierArray[i]['dim'],self.classifierArray[i]['threshold'],self.classifierArray[i]['inequality'])
				aggregateClassGuess = aggregateClassGuess + (-1 * self.classifierArray[i]['alpha'] * classGuess)
				# print aggregateClassGuess
			classifiedDict[i] = np.sign(aggregateClassGuess)
		return classifiedDict

class cascade:

	def __init__(self):
		self.subwindow = []
		self.falsePositiveRate 	   = 1.0
		self.detectionRate        = 1.0
		self.positiveSet 		   = self.loadPositives()
		self.negativeSet  	   = self.loadNegatives()
		self.cascadedClassifier = {}

	def loadPositives(self):
		0# load images that have faces

	def loadNegatives(self):
		0# load non-face images

	def cascadedClassifierGuess(self,data):
		classifiedDict = {}

		# for every data point
		for i in data:

			dataMatrix = np.matrix(data)
			n,m = np.shape(dataMatrix)
			aggregateClassGuess = np.matrix(np.zeros((n,1)))

			# go through each classifier in our cascaded classifier
			for (layer,classifier) in self.cascadedClassifier:

				# get a classguess
				for i in range (0,len(classifier)):
					classGuess = adaboost.guessClass(dataMatrix,classifier[i]['dim'],classifier[i]['threshold'],classifier[i]['inequality'])
					aggregateClassGuess = aggregateClassGuess + (-1 * classifier[i]['alpha'] * classGuess)

				# if a layer returns a negative result, automatically return negative
				if np.sign(aggregateClassGuess) == -1:
					classifiedDict[i] = -1
					break

			# else, if every classifier says it's good, then return 1
			classifiedDict[i] = 1

		return classifiedDict

	def adjustThreshold(self,classifier):
		classifier['threshold'] -= .05

	def trainCascadedClassifier(self,f,d,Ftarget):

		# while your false positive rate is too high
		while self.falsePositiveRate > Ftarget:

			n = 0
			newFalsePositiveRate = self.falsePositiveRate

			adabooster = adaBoost()

			# we're trying to get our false positive rate down
			while newFalsePositiveRate > (f * self.falsePositiveRate):
				n += 1

				# make a new adabooster and boost to get a classifier with n features
				adabooster.loadData(self.positiveSet,self.negativeSet)
				adabooster.boost(n)

				# add our new classifier to our cascadedClassifier
				self.cascadedClassifier[n] = adabooster.classifierArray

				# find our classifier's false positive and detection rate
				negativeSetGuesses = self.cascadedClassifierGuess(self.negativeSet)
				ncnt = Counter()
				for k,v in negativeSetGuesses.items():
					ncnt[v] += 1
				newFalsePositiveRate = ncnt[1] / len(negativeSetGuesses)

				positiveSetGuesses = self.cascadedClassifierGuess(self.positiveSet)
				pcnt = Counter()
				for k,v in positiveSetGuesses.items():
					pcnt[v] += 1
				newDetectionRate = pcnt[1] / len(positiveSetGuesses)

				# adjust the most recently added classifier
				while newDetectionRate < d * self.detectionRate:

					# IMPLEMENT THIS
					self.adjustThreshold(self.cascadedClassifier[n])

					# re-test and see if we have a good detection rate
					positiveSetGuesses = self.cascadedClassifierGuess(self.positiveSet)
					cnt = Counter()
					for k,v in positiveSetGuesses.items():
						cnt[v] += 1
					newDetectionRate = cnt[1] / len(positiveSetGuesses)
			
			# replace our current negative set with only false detections
			self.negativeSet = []

			if newFalsePositiveRate  > self.falsePositiveRate:
				negativeSetGuesses = self.cascadedClassifierGuess(self.negativeSet)
				self.negativeSet = [k for (k,v) in negativeSetGuesses.iteritems() if v == 1]

adabooster = adaBoost()
adabooster.loadData( [ [12,3] , [10,4] , [2,3.5] ] , [ [2,10] , [3,11] , [14,12] ] )
adabooster.boost(30)
print adabooster.classify([-3,1])
