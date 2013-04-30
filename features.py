import numpy
import scipy
import math

class features:

	# will be an array of all possible features for that image
	f = []
	feature_table = []
	origin_x = 0
	origin_y = 0
	patch_scale = 0.0
	patch_mean = 0.0
	patch_std = 0.0
	img = integral_image[]#what do i put in here?!?

	def __init__():
		i = 0
		min_patch_side = 24

		int nfeatures = 42310/100

		for i in (0, )

		# creates first type of feature of two rectangles stacked on top of each other
		for w in xrange (1, min_patch_side+1):
			for h in xrange(1, 	min_patch_side/2 +1):
				for x in xrange(0, min_patch_side-w +1):
					for y in xrange(0, min_patch_side-2*h+1):
						features.feature_table[i] = 1
						features.feature_table[i+1] = x
						features.feature_table[i+2] = y
						features.feature_table[i+3] = w
						features.feature_table[i+4] = h
						i+=5

		# creates second type of feature of two rectangles side by side
		for w in xrange (1, min_patch_side/2 +1):
			for h in xrange(1, 	min_patch_side+1):
				for x in xrange(0, min_patch_side-2*w +1):
					for y in xrange(0, min_patch_side-h+1):
						features.feature_table[i] = 1
						features.feature_table[i+1] = x
						features.feature_table[i+2] = y
						features.feature_table[i+3] = w
						features.feature_table[i+4] = h
						i+=5

		# creates third type of feature of three rectangles side by side
		for w in xrange (1, min_patch_side/3 +1):
			for h in xrange(1, 	min_patch_side+1):
				for x in xrange(0, min_patch_side-3*w +1):
					for y in xrange(0, min_patch_side-h+1):
						features.feature_table[i] = 1
						features.feature_table[i+1] = x
						features.feature_table[i+2] = y
						features.feature_table[i+3] = w
						features.feature_table[i+4] = h
						i+=5

		# creates fourth type of feature of four rectangles in checkerboard form
		for w in xrange (1, min_patch_side/2 +1):
			for h in xrange(1, 	min_patch_side/2+1):
				for x in xrange(0, min_patch_side-2*w +1):
					for y in xrange(0, min_patch_side-2*h+1):
						features.feature_table[i] = 1
						features.feature_table[i+1] = x
						features.feature_table[i+2] = y
						features.feature_table[i+3] = w
						features.feature_table[i+5] = h
						i+=5


	# calculates value of the feature 
	def compute_features (int ind):
		ind *= 5
		feattype = features.feature_table[ind]
		x = features.feature_table[ind+1]
		y = features.feature_table[ind+2]
		w = features.feature_table[ind+3]
		h = features.feature_table[ind+4]

		# Scale the feature to fit the current patch.
		x = (int) Math.round(origin_x + x*patch_scale)
		y = (int) Math.round(origin_y + y*patch_scale)
		w = (int) Math.round(w*patch_scale)
		h = (int) Math.round(h*patch_scale)

		if feattype ==1:
			return typeI(x, y, w, h);	
		elif feattype ==2: 
			return typeII(x, y, w, h);
		elif feattype ==3:
			return typeIII(x, y, w, h);
		elif feattype ==4:
			return typeIV(x, y, w, h);
		else:
			print 'Tried to use feature type:' + str(ind)


	# sets region of interest with coordinates x, y and a width for specific features
	def set_ROI(x, y, w):
		origin_x = x
		origin_y = y		
		patch_scale = ((float)w)/((float)min_patch_side)

		# std^2 = mean(x^2) + mean(x)^2
		mean = findIntegral(x,y,w,w)
		mean /= ((float)(w*w))

		meanSqr = findIntegral(x,y,w,w)
		meanSqr /= ((float)(w*w))

		if (meanSqr<=0) 
			patch_std = 1
		
		else
			patch_std = math.sqrt(math.pow(mean,2)+meanSqr)
		


	# takes in array ind and populates array f with features for that image
	def get_features (ind, f):
		for i in xrange (0, len(ind))
			f[i] = computeFeature(ind[i])

	
	# feature comes from the array feature_table
	# need to take in a feature, its original label of whether a face exists, and return a number using the integral_image func 
	def get_val (feature, image, label):

	# iterates through different features and resize them 
	def itrfeatures(image):

	# take in the image, the feature, its number, and return the guess
	def guess (image, feature, val):

