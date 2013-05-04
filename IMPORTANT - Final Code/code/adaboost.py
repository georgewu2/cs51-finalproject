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

	def loadData(self,positiveDir="../data/sameface/",negativeDir="../data/randombg/"):

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

	def loadDataFromMatrices(self,positives,negatives):

		bigSet = []
		bigSet.extend(positives)
		bigSet.extend(negatives)
		self.data = bigSet

		self.labels = ([1 for x in range(len(positives))])
		self.labels.extend([-1 for x in range(len(negatives))])

		self.featuresMatrix = np.zeros((1,8628))

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

	def trainClassifier(self,data,labels,weights,steps,weakClassGuessers):

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
		for feature in range(0,100):
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
						if not weakClassGuessers:
							# print "CLASSGUESS", classGuess
							minError = weightedError
							bestClassGuess = classGuess.copy()
							bestClassifier['feature'] = feature
							bestClassifier['threshold'] = threshold
							bestClassifier['inequality'] = inequality
						else:
							if feature not in [weakClassGuessers[x]['feature'] for x in range(0,len(weakClassGuessers))]:
								minError = weightedError
								bestClassGuess = classGuess.copy()
								bestClassifier['feature'] = feature
								bestClassifier['threshold'] = threshold
								bestClassifier['inequality'] = inequality

		return bestClassifier,minError,bestClassGuess

	def boost(self,maxFeatures):

		self.classifierArray = []
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
			bestClassifier,error,classGuess = self.trainClassifier(self.data,self.labels,weights,10,weakClassGuessers)

			print bestClassifier

			# calculate weight of the classifier
			alpha = float(math.log(1.0 - error) / max(error,1e-16))
			
			bestClassifier['alpha'] = alpha

			# print "ALPHA", alpha
			if alpha == 0:
				break

			# add classifier to weakClassGuess
			weakClassGuessers.append(bestClassifier)

			# calculate new weights
			exponent = np.multiply(alpha * np.matrix(self.labels), classGuess)

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

		print self.classifierArray

	def classify(self,i):
		features = Features(i)
		featuresMatrix = features.f
		aggregateClassGuess = 0

		# for every classifier we train, use it to classguess and then scale by
		# alpha and add to aggregate guess
		for classifier in range (0,len(self.classifierArray)):
			classGuess = self.guessClass(featuresMatrix,self.classifierArray[classifier]['feature'],self.classifierArray[classifier]['threshold'],self.classifierArray[classifier]['inequality'])
			print "CLASS GUESS",classGuess
			aggregateClassGuess = aggregateClassGuess + (self.classifierArray[classifier]['alpha'] * classGuess)
			print "AGG CLASS GUESS", aggregateClassGuess
		
		if np.sign(aggregateClassGuess) == 1:
			return True
		else: 
			return False

class Cascade:

	def __init__(self):
		self.subwindow = []
		self.falsePositiveRate 	   = 1.0
		self.detectionRate        = 1.0
		self.positiveSet 		   = self.loadPositives()
		self.negativeSet  	   = self.loadNegatives()
		self.cascadedClassifier = {}

	def loadPositives(self,positiveDir="../data/sameface/"):
		positiveSet = []

		positiveImages = os.listdir(os.getcwd() + "/" + positiveDir)
		# # get rid of the .DS_Store file
		positiveImages.pop(0)

		# add each vector to the list
		for i in positiveImages:
			positiveSet.append(get_frame_vector(positiveDir + i,False))

		return positiveSet

	def loadNegatives(self,negativeDir="../data/randombg/"):
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
			# print "DATA WE ARE GUESSING",i
			classifiedDict[i] = 1

			features = Features(i)
			featuresMatrix = features.f
			n = len(data)
			aggregateClassGuess = 0

			# go through each classifier in our cascaded classifier
			# print self.cascadedClassifier
			for layer,classifier in self.cascadedClassifier.items():

				# get a classguess
				for x in range (0,len(classifier)):
					classGuess = adabooster.guessClass(featuresMatrix,classifier[x]['feature'],classifier[x]['threshold'],classifier[x]['inequality'])
					# print "CLASS GUESS", classGuess
					aggregateClassGuess = aggregateClassGuess + (-1 * classifier[x]['alpha'] * classGuess)
					# print "AGGREGATE GUESS", aggregateClassGuess

				# if a layer returns a negative result, automatically return negative
				# print "AGG GUESS",aggregateClassGuess
				if np.sign(aggregateClassGuess) == -1:
					classifiedDict[i] = -1
					break

			# else, if every classifier says it's good, then return 1
			# print "CLASSIFIED DICT", classifiedDict
		# print classifiedDict
		return classifiedDict

	def adjustThreshold(self,classifier,n):
		for i in range(0,n):
			if classifier[n][i]['inequality'] == "<=":
				classifier[n][i]['threshold'] += 2
			else:
				classifier[n][i]['threshold'] -= 2

	def trainCascadedClassifier(self,f,d,Ftarget):

		adabooster = adaBoost()
		adabooster.loadData()

		# while your false positive rate is too high
		while self.falsePositiveRate > Ftarget:

			n = 0
			newFalsePositiveRate = self.falsePositiveRate
			print "BIG LOOP FALSE POSITIVE", newFalsePositiveRate

			# we're trying to get our false positive rate down
			while newFalsePositiveRate > (f * self.falsePositiveRate):
				print "CURRENT FALSE POSITIVE RATE", newFalsePositiveRate

				n += 1

				if n > 1:
					adabooster.loadDataFromMatrices(self.positiveSet,self.negativeSet)

				# make a new adabooster and boost to get a classifier with n features
				adabooster.boost(n)

				# add our new classifier to our cascadedClassifier
				self.cascadedClassifier[n] = adabooster.classifierArray

				# find our classifier's false positive and detection rate
				negativeSetGuesses = self.cascadedClassifierGuess(self.negativeSet,adabooster)
				ncnt = Counter()
				for k,v in negativeSetGuesses.items():
					ncnt[v] += 1
				newNewFalsePositiveRate = float(ncnt[1]) / float(len(negativeSetGuesses.items()))

				# print "NEGATIVE SET GUESSES", negativeSetGuesses

				positiveSetGuesses = self.cascadedClassifierGuess(self.positiveSet,adabooster)
				pcnt = Counter()
				for k,v in positiveSetGuesses.items():
					pcnt[v] += 1
				newDetectionRate = float(pcnt[1]) / float(len(positiveSetGuesses.items()))

				# print "POSITIVE SET GUESSES", positiveSetGuesses

				# adjust the most recently added classifier
				while newDetectionRate < d * self.detectionRate:

					# IMPLEMENT THIS
					print "CASCADED CLASSIFIER", self.cascadedClassifier
					self.adjustThreshold(self.cascadedClassifier,n)

					# re-test and see if we have a good detection rate
					positiveSetGuesses = self.cascadedClassifierGuess(self.positiveSet,adabooster)
					# print "POSITIVE SET GUESSES", positiveSetGuesses
					cnt = Counter()
					for k,v in positiveSetGuesses.items():
						cnt[v] += 1
					newDetectionRate = float(cnt[1]) / float(len(positiveSetGuesses.items()))
					print "DIVIDIED", newDetectionRate, d * self.detectionRate
			
				# replace our current negative set with only false detections
				tempNegativeSet = []

				print "NEW FALSE POSITIVE RATE", newFalsePositiveRate, f, self.falsePositiveRate, newNewFalsePositiveRate
				if newFalsePositiveRate > f:
					negativeSetGuesses = self.cascadedClassifierGuess(self.negativeSet,adabooster)
					print negativeSetGuesses
					for (k,v) in negativeSetGuesses.items():
						if v == 1:
							tempNegativeSet.append(k)
					self.negativeSet = tempNegativeSet
					print self.negativeSet

				newFalsePositiveRate = newFalsePositiveRate * newNewFalsePositiveRate
				print "NEW FALSE POSITIVE RATE TO CHECK", newFalsePositiveRate


			self.falsePositiveRate = newFalsePositiveRate
			print "NEW SELF FALSE POSITIVE RATE", self.falsePositiveRate

	def cascadedClassify(self, i):
		adabooster = adaBoost()
		result = self.cascadedClassifierGuess([i],adabooster)
		if [v for (k,v) in result.items()][0] == -1:
			return False
		else:
			return True

# cascader = Cascade()
# cascader.trainCascadedClassifier(.2,.25,.7)

# positiveImages = os.listdir(os.getcwd() + "/testconfirmedpos")
# positiveImages.pop(0)

# for i in positiveImages:
# 	print "POSITIVES"
# 	print cascader.cascadedClassify(get_frame_vector("testconfirmedpos/" + i,False))

# negativeImages = os.listdir(os.getcwd() + "/testconfirmedneg")
# negativeImages.pop(0)

# for i in negativeImages:
# 	print "NEGATIVES"
# 	print cascader.cascadedClassify(get_frame_vector("testconfirmedneg/" + i,False))
