from feature import Features
import numpy


class perceptron:


	"""

	takes in original dataset of images and their corresponding labels of 
	whether there is a face or not (-1 or 1)
	"""
	def __init__(self, imgs, imglabels):	
	# perceptron initialization
		self.classws = [0]*134736

		# weights for classifying are randomly initialized to -1 or 1
		for w in self.classws:
			w = random.randint(0,1)
			if w == 0:
				w = -1

		# alpha value - to test for 0.1, 0.5, 11, 10
		self.learningRate = 0.1

		self.images = imgs # dataset of images to train on
		self.labels = imglabels # labels of images in dataset of whether there exists a face
		
		# self.imgswithvals = [] # potential guesses for each image

		# set of weights with the default weight of 1 to offset data
		self.w = [1]
		
		
		"""

		# for each image passed into dataset, get the feature value for each image
		# not sure i have to train everything before i do it
		for img in self.imags:
			pic = Feature(img)
			imgswithvals.append(classify(pic.f, self.classws)) 

		"""

		# after training, assigns threshold value for guessing
		self.threshold = train(self.images)


	"""
	
	takes in a list of vales from features and a random list of weights and returns potential guess 
	feats : list of values calculated from features.py for each feature for one image
	weights : list of same size as feats that contains random -1 or 1 weight to different features
	"""
	def classify (self, feats, weights):
		return np.dot(feats, weights)	


	"""

	takes in original (non-integral) image 
	in classifying, returns either -1 or 1 as a potential guess for one image
	"""
	def response(self,img): 
  		pic = Features(img) 

  		# classifies image and returns "stupid" guess
  		y = classify(pic.f, self.classws)   
  		if y >= 0:
   			return 1
  		else:
   			return -1


   	"""

	updates the weights status, w at time t+1 is
	    w(t+1) = w(t) + learningRate*(d-r)*x
	where d is desired output and r the perceptron response
	iterError is (d-r)
	"""
	def updateWeights(self,x,iterError):
	  
	  	lastw = self.w[len(self.w)-1]
		self.w.append(lastw + self.learningRate*iterError*x)


	""" 

	trains all the vector in data.
	returns the last weight that will become the "stupid" threshold
	"""
 	def train(self,imgs):
		learned = False
		while not learned:
			globalError = 0.00
			for i in range (0,len(imgs)): # for each sample
			    img = imgs [i]
			    r = self.response(img)    
			    if self.labels[i] != r: # if we have a wrong response
				    iterError = self.labels[i] - r # desired response - actual response
				    self.updateWeights(self.labels[i],iterError)
				    globalError += abs(iterError)
			
			if globalError <= 0.001 : # stop criteria
			   	learned = True # stop learning

		return self.w[len(self.w)-1]


	# take in the newimage return the guess
	def guess (self, image): 
		pic = Feature(image)
		guess = classify (pic.f, self.w)
		if(guess >= threshold):
			return 1
		else: 
			return -1
