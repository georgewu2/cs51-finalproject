import numpy as np
import math
from collections import Counter
import Image
import os
from features import Features

def get_frame_vector(video_frame, flatten=True):
	im = Image.open(video_frame)

	# convert to grayscale
	im_gray = im.convert('L')

	# convert to a matrix
	im_matrix = np.matrix(im_gray)
	if flatten == True:
		return im_matrix.flatten('F')
	else:
		return im_matrix

class adaBoost:

	def __init__(self):
		self.data 			 = np.matrix([])
		self.labels 		 = np.matrix([])
		self.classifierArray = np.matrix([])
		self.featuresMatrix = np.zeros((1,8628))

	def loadData(self,positiveDir="testimgspos/",negativeDir="testimgsneg/"):

		positiveSet = []
		negativeSet = []

		positiveImages = os.listdir(os.getcwd() + "/" + positiveDir)
		# # get rid of the .DS_Store file
		positiveImages.pop(0)

		# add each vector to the list
		for i in positiveImages:
			positiveSet.append(get_frame_vector(positiveDir + i,False))

		negativeImages = os.listdir(os.getcwd() + "/" + negativeDir)
		# # get rid of the .DS_Store file
		negativeImages.pop(0)

		# add each vector to the list
		for i in negativeImages:
			negativeSet.append(get_frame_vector(negativeDir + i,False))

		bigSet = []
		bigSet.extend(positiveSet)
		bigSet.extend(negativeSet)

		self.data = bigSet

		self.labels = ([1 for x in range(len(positiveSet))])
		self.labels.extend([-1 for x in range(len(negativeSet))])

		x = 0
		for img in self.data:
			print x
			x += 1
			integralImageWithFeatures = Features(img)
			self.featuresMatrix = np.vstack([self.featuresMatrix,integralImageWithFeatures.f])
		self.featuresMatrix = np.delete(self.featuresMatrix,0,0)
		n,m = np.shape(self.featuresMatrix)
		print n,m

	def guessClass(self,featuresMatrix,feature,threshold,inequality):

		# get dimensions and make a return vector
		try:
			n,m = np.shape(featuresMatrix)
		except ValueError:
			n = 1

		classed = (np.ones((1,n)))[0]

		if n == 1:
			if inequality == '<=':
					if featuresMatrix[feature] <= threshold:
						classed = -1

			else:
					if featuresMatrix[feature] > threshold:
						classed = -1
			return classed

		else:
			# if a data doesn't meet threshold make it -1
			if inequality == '<=':
				for img in range(0,n):
					if featuresMatrix[img][feature] <= threshold:
						classed[img] = -1

			else:
				for img in range(0,n):
					if featuresMatrix[img][feature] > threshold:
						classed[img] = -1
			return classed

	def trainClassifier(self,data,labels,weights,steps):

		# setup
		labelMatrix = np.matrix(labels).T
		try:
			n,m,o = np.shape(data)
		except ValueError:
			try: 
				n,m = np.shape(data)
			except:
				n = len(data)


		bestClassifier = {}
		bestClassGuess = np.zeros((n,1))
		minError = float('inf')


		# MAKE A GIANT MATRIX WITH ROWS BEING INTEGRALIMAGEFEATURES.F 
		# TAKE EACH COLUMN PER FEATURE!!! YEAH!!! SO YOU CAN DO MIN AND MAX
		# ON EACH COLUMN AND DO THE SAME THING WE'VE BEEN DOING!
		# FUCK YEAH

		# the work
		for feature in range(0,8628):
			# find min and max of x or y coordinates
			rangeMin = self.featuresMatrix[:,feature].min()
			rangeMax = self.featuresMatrix[:,feature].max()

			# print "FEATURE", feature, self.featuresMatrix[:,feature]

			stepSize = (rangeMax - rangeMin) / float(steps)

			# for every little subdivision of the range
			for j in range(-1,steps + 1):
				for inequality in ['<=','>']:

					# set the threshold value by incrementing one step over minrange
					# and classify based on the threshold
					threshold = (rangeMin + (j * stepSize))
					classGuess = self.guessClass(self.featuresMatrix,feature,threshold,inequality)
					
					# find the error in predictedClasses
					errorArray = np.ones((n,1))

					# if wrong prediction, set to 0
					for img in range(0,len(self.featuresMatrix)):
						if classGuess[img] == labelMatrix[img]:
							errorArray[img] = 0

					weightedError = np.matrix(weights).T * np.matrix(errorArray)

					# if weighted error is smallest, then put all our current 
					# stuff in a dictionary
					if weightedError < minError:
						# print "CLASSGUESS", classGuess
						minError = weightedError
						bestClassGuess = classGuess.copy()
						bestClassifier['feature'] = feature
						bestClassifier['threshold'] = threshold
						bestClassifier['inequality'] = inequality

		return bestClassifier,minError,bestClassGuess

	def boost(self,maxFeatures):
		weakClassGuessers = []

		try:
			n,m,o = np.shape(self.data)
		except ValueError:
			try: 
				n,m = np.shape(self.data)
			except ValueError:
				n = len(self.data)

		# setup weight vector
		weights = np.ones((n,1))
		weights = weights * (1. / n)

		aggregateClassGuess = np.zeros((n,1))

		# the work
		for i in range (0,maxFeatures):
			
			# print("ITERATION", i)

			# train best classifier for these weights
			bestClassifier,error,classGuess = self.trainClassifier(self.data,self.labels,weights,10)

			print bestClassifier

			# calculate weight of the classifier
			alpha = max(1e-16,float(math.log(1.0 - error) / max(error,1e-16)))
			
			bestClassifier['alpha'] = alpha

			# print "ALPHA", alpha

			# add classifier to weakClassGuess
			weakClassGuessers.append(bestClassifier)

			# calculate new weights
			exponent = np.multiply(1 * alpha * np.matrix(self.labels), classGuess)

			# print "EXPONENT", exponent

			weights = np.multiply(weights,np.exp(exponent.T))
			weights = weights * (1 / weights.sum())

			# print "WEIGHTS", weights

			# update aggregateClassGuess
			aggregateClassGuess = aggregateClassGuess + np.matrix((-1 * alpha * classGuess)).T

			# print "AGGREGATE CLASS GUESS",aggregateClassGuess
			# aggregateErrors
			aggregateErrors = np.multiply(np.sign(aggregateClassGuess) != np.matrix(self.labels).T, np.ones((n,1)))
			errorRate = aggregateErrors.sum() / n
			# print aggregateErrors

			# print "ERRORRATE", errorRate

			if errorRate == 0.0: 
				print "NOERROR"
				break

		self.classifierArray = weakClassGuessers

		# print self.classifierArray

	def classify(self,data):
		classifiedDict = {}
		for i in data:
			features = Features(i)
			featuresMatrix = features.f
			n = len(data)
			print "LENGTH",n
			aggregateClassGuess = 0

			# for every classifier we train, use it to classguess and then scale by
			# alpha and add to aggregate guess
			for classifier in range (0,len(self.classifierArray)):
				classGuess = self.guessClass(featuresMatrix,self.classifierArray[classifier]['feature'],self.classifierArray[classifier]['threshold'],self.classifierArray[classifier]['inequality'])
				# print "CLASS GUESS",classGuess
				aggregateClassGuess = aggregateClassGuess + (self.classifierArray[classifier]['alpha'] * classGuess)
				# print "AGG CLASS GUESS", aggregateClassGuess
			
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

	def loadPositives(self,positiveDir="testimgspos/"):
		positiveSet = []

		positiveImages = os.listdir(os.getcwd() + "/" + positiveDir)
		# # get rid of the .DS_Store file
		positiveImages.pop(0)

		# add each vector to the list
		for i in positiveImages:
			positiveSet.append(get_frame_vector(positiveDir + i,False))

		return positiveSet

	def loadNegatives(self,negativeDir="testimgsneg/"):
		negativeSet = []

		negativeImages = os.listdir(os.getcwd() + "/" + negativeDir)
		# # get rid of the .DS_Store file
		negativeImages.pop(0)

		# print negativeImages
		# add each vector to the list
		for i in negativeImages:
			print negativeDir + i
			negativeSet.append(get_frame_vector(negativeDir + i,False))

		return negativeSet

	def cascadedClassifierGuess(self,data,adabooster):
		classifiedDict = {}

		# for every data point
		for i in data:

			features = Features(i)
			featuresMatrix = features.f
			n = len(data)
			aggregateClassGuess = 0

			# go through each classifier in our cascaded classifier
			# print self.cascadedClassifier
			for layer,classifier in self.cascadedClassifier.items():

				# get a classguess
				for i in range (0,len(classifier)):
					classGuess = adabooster.guessClass(featuresMatrix,classifier[i]['feature'],classifier[i]['threshold'],classifier[i]['inequality'])
					# print "CLASS GUESS", classGuess
					aggregateClassGuess = aggregateClassGuess + (-1 * classifier[i]['alpha'] * classGuess)

				# if a layer returns a negative result, automatically return negative
				# print "AGG GUESS",aggregateClassGuess
				if np.sign(aggregateClassGuess) == -1:
					classifiedDict[i] = -1
					break

			# else, if every classifier says it's good, then return 1
			classifiedDict[i] = 1

		return classifiedDict

	def adjustThreshold(self,classifier):
		classifier['threshold'] -= 1

	def trainCascadedClassifier(self,f,d,Ftarget):

		adabooster = adaBoost()
		adabooster.loadData()

		# while your false positive rate is too high
		while self.falsePositiveRate > Ftarget:

			n = 0
			newFalsePositiveRate = self.falsePositiveRate


			# we're trying to get our false positive rate down
			while newFalsePositiveRate > (f * self.falsePositiveRate):
				print "FALSE POSITIVE RATE", self.falsePositiveRate

				n += 1

				# make a new adabooster and boost to get a classifier with n features
				adabooster.boost(n)

				# add our new classifier to our cascadedClassifier
				self.cascadedClassifier[n] = adabooster.classifierArray

				# find our classifier's false positive and detection rate
				negativeSetGuesses = self.cascadedClassifierGuess(self.negativeSet,adabooster)
				ncnt = Counter()
				for k,v in negativeSetGuesses.items():
					ncnt[v] += 1
				newFalsePositiveRate = ncnt[1] / len(negativeSetGuesses)

				positiveSetGuesses = self.cascadedClassifierGuess(self.positiveSet,adabooster)
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

cascader = cascade()
cascader.trainCascadedClassifier(.1,.9,.1)

# adabooster = adaBoost()
# adabooster.loadData()
# adabooster.boost(10)

# positiveImages = os.listdir(os.getcwd() + "/testconfirmedpos")
# positiveImages.pop(0)

# positiveSet = [v for k,v in (adabooster.classify(map(lambda x : get_frame_vector("testconfirmedpos/" + x,False),positiveImages))).items()]

# negativeImages = os.listdir(os.getcwd() + "/testconfirmedneg")
# negativeImages.pop(0)

# negativeSet = [v for k,v in (adabooster.classify(map(lambda x : get_frame_vector("testconfirmedneg/" + x,False),negativeImages))).items()]

# print "POSITIVE",positiveSet
# print "NEGATIVE",negativeSet