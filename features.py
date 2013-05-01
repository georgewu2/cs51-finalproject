import numpy
import scipy
import Image
import integral_image
from training import Faces

class features:

	def __init__(self):

		# will be an array of all possible features for that image
		self.f = [None for x in xrange(4500000)]
		self.feature_table = [None for x in xrange(4500000)]
		self.origin_x = 0
		self.origin_y = 0
		self.patch_scale = 0.0
		self.patch_mean = 0.0
		self.patch_std = 0.0
		self.faces = Faces()
		self.img = integral_image.integral_image(self.faces.get_frame_vector("face_1.jpg"))

		print "LOL"
		self.init_helper()


	# calculates value of the feature 
	def compute_features (self, ind):
		ind *= 5
		feattype = self.feature_table[ind]
		x = self.feature_table[ind+1]
		y = self.feature_table[ind+2]
		w = self.feature_table[ind+3]
		h = self.feature_table[ind+4]

		# Scale the feature to fit the current patch.
		x = int(round(self.origin_x + x*self.patch_scale))
		y = int(round(self.origin_y + y*self.patch_scale))
		w = int(round(w*self.patch_scale))
		h = int(round(h*self.patch_scale))

		if feattype == 1:
			return self.typeI(x, y, w, h)	
		elif feattype ==2: 
			return self.typeII(x, y, w, h)
		elif feattype ==3:
			return self.typeIII(x, y, w, h)
		elif feattype ==4:
			return self.typeIV(x, y, w, h)
		else:
			print 'Tried to use feature type:' + str(ind)

	def init_helper(self):
		i = 0
		min_patch_side = 24

		nfeatures = 4320000/100

		ind = [x for x in xrange(4500000)]

		# creates first type of feature of two rectangles stacked on top of each other
		for w in xrange (1, min_patch_side+1):
			for h in xrange(1, 	min_patch_side/2 +1):
				for x in xrange(0, min_patch_side-w +1):
					for y in xrange(0, min_patch_side-2*h+1):
						self.feature_table[i] = 1
						print i
						self.feature_table[i+1] = x
						self.feature_table[i+2] = y
						self.feature_table[i+3] = w
						self.feature_table[i+4] = h
						i+=5

		# creates second type of feature of two rectangles side by side
		for w in xrange (1, min_patch_side/2 +1):
			for h in xrange(1, 	min_patch_side+1):
				for x in xrange(0, min_patch_side-2*w +1):
					for y in xrange(0, min_patch_side-h+1):
						self.feature_table[i] = 2
						self.feature_table[i+1] = x
						self.feature_table[i+2] = y
						self.feature_table[i+3] = w
						self.feature_table[i+4] = h
						i+=5

		# creates third type of feature of three rectangles side by side
		for w in xrange (1, min_patch_side/3 +1):
			for h in xrange(1, 	min_patch_side+1):
				for x in xrange(0, min_patch_side-3*w +1):
					for y in xrange(0, min_patch_side-h+1):
						self.feature_table[i] = 3
						self.feature_table[i+1] = x
						self.feature_table[i+2] = y
						self.feature_table[i+3] = w
						self.feature_table[i+4] = h
						i+=5

		# creates fourth type of feature of four rectangles in checkerboard form
		for w in xrange (1, min_patch_side/2 +1):
			for h in xrange(1, 	min_patch_side/2+1):
				for x in xrange(0, min_patch_side-2*w +1):
					for y in xrange(0, min_patch_side-2*h+1):
						self.feature_table[i] = 4
						self.feature_table[i+1] = x
						self.feature_table[i+2] = y
						self.feature_table[i+3] = w
						self.feature_table[i+4] = h
						i+=5
		self.get_features(ind, self.f)



	# sets region of interest with coordinates x, y and a width for specific features
	def set_ROI(self, x, y, w):
		self.origin_x = x
		self.origin_y = y		
		patch_scale = (float(w))/(float(min_patch_side))

		# std^2 = mean(x^2) + mean(x)^2
		mean = findIntegral(x,y,w,w)
		mean /= (float((w*w)))

		meanSqr = findIntegral(x,y,w,w)
		meanSqr /= (float((w*w)))

		if (meanSqr<=0):
			patch_std = 1
		else:
			patch_std = math.sqrt(math.pow(mean,2)+meanSqr)
		

	# takes in array ind and populates array f with features for that image
	def get_features (self, ind, f):
		for i in xrange (0, len(ind)):
			f[i] = self.compute_features(ind[i])


	"""
	  Type I feature:
	  
	  	<w->
	  ---- h
	  ++++ h

	"""
	def typeI(self, x, y, w, h):
		sumU = integral_image.findIntegral(x,y,w,h)
		sumD = integral_image.findIntegral(x,y+h,w,h)
		return (sumD-sumU)/patch_std
	
	"""
	  Type II feature:
	  
	   <w-><w->
	   ++++---- ^
	   ++++---- h
	   ++++---- v
	 """
	def typeII(self, x, y, w, h):

		sumL = integral_image.findIntegral(x,y,w,h)
		sumR = integral_image.findIntegral(x+w,y,w,h)
		return (sumL-sumR)/patch_std
	
	"""
	  Type III feature:
	  
	 	<w-><w-><w->
	   ++++----++++ ^
	   ++++----++++ h
	   ++++----++++ v
	  
	"""
	def typeIII(self, x, y, w, h):

		sumL = findInt(x,y,w,h)
		sumC = findInt(x+w,y,w,h)
		sumR = findInt(x+2*w,y,w,h)
		# We have to account for the mean, since there are more (+) than (-).
		return (sumL-sumC+sumR-patch_mean*w*h)/patch_std
	
	"""
	 Type IV feature:
	 
	 	<w-><w->
	  ++++---- ^
	  ++++---- h
	  ++++---- v
	  ----++++ ^
	  ----++++ h
	  ----++++ v
	"""
	def typeIV(self, x, y, w, h):
		sumLU = findInt(x,y,w,h)
		sumRU = findInt(x+w,y,w,h)
		sumLD = findInt(x,y+h,w,h)
		sumRD = findInt(x+w,y+h,w,h)
		return (-sumLD+sumRD+sumLU-sumRU)/patch_std
	

	# feature comes from the array feature_table
	# need to take in a feature, its original label of whether a face exists, and return a number using the integral_image func 
	def get_val (self, feature, image, label):
		0

	# iterates through different features and resize them 
	def itrfeatures(self, image): 0

	# take in the image, the feature, its number, and return the guess
	def guess (self, image, feature, val): 0

featuretest = features()
featuretest.compute_features(1)