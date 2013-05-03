import numpy
# import scipy
import Image
import math
from integral_image import Integrate
from training import Faces

class Features:

	def __init__(self,img):

		# will be an array of all possible features for that image
		self.f = []

		# array of all features (the type and the coordinates of the corners)
		self.feature_table = [None for x in xrange(8628)]
		self.origin_x = 0
		self.origin_y = 0
		self.patch_scale = 1.0
		self.patch_mean = 1.0
		self.patch_std = 1.0
		self.faces = Faces()
		self.min_patch_side = 12 # smallest block size 


		self.pic = Integrate(img)

		self.img = self.pic.integral_image()

		self.set_ROI(self.pic, 0, 0, min(len(img[0]), img.size/len(img[0]))) 

		self.init_helper()

	def init_helper(self):

		ind = [x for x in xrange(43140)] 

		i = 0

		# creates first type of feature of two rectangles stacked on top of each other
		for w in xrange (1, self.min_patch_side+1):
			for h in xrange(1, 	self.min_patch_side/2 +1):
				for x in xrange(0, self.min_patch_side-w +1):		
					for y in xrange(0, self.min_patch_side-2*h+1):
						self.feature_table[i] = 1
						# for testing, print i
						self.feature_table[i+1] = x
						self.feature_table[i+2] = y
						self.feature_table[i+3] = w
						self.feature_table[i+4] = h
						i+=5

		# creates second type of feature of two rectangles side by side
		for w in xrange (1, self.min_patch_side/2 +1):
			for h in xrange(1, 	self.min_patch_side+1):
				for x in xrange(0, self.min_patch_side-2*w +1):
					for y in xrange(0, self.min_patch_side-h+1):
						self.feature_table[i] = 2
						self.feature_table[i+1] = x
						self.feature_table[i+2] = y
						self.feature_table[i+3] = w
						self.feature_table[i+4] = h
						i+=5

		# creates third type of feature of three rectangles side by side
		for w in xrange (1, self.min_patch_side/3 +1):
			for h in xrange(1, 	self.min_patch_side+1):
				for x in xrange(0, self.min_patch_side-3*w +1):
					for y in xrange(0, self.min_patch_side-h+1):
						self.feature_table[i] = 3
						self.feature_table[i+1] = x
						self.feature_table[i+2] = y
						self.feature_table[i+3] = w
						self.feature_table[i+4] = h
						i+=5

		# creates fourth type of feature of four rectangles in checkerboard form
		for w in xrange (1, self.min_patch_side/2 +1):
			for h in xrange(1, 	self.min_patch_side/2+1):
				for x in xrange(0, self.min_patch_side-2*w +1):
					for y in xrange(0, self.min_patch_side-2*h+1):
						self.feature_table[i] = 4
						self.feature_table[i+1] = x
						self.feature_table[i+2] = y
						self.feature_table[i+3] = w
						self.feature_table[i+4] = h
						i+=5
		self.get_features(ind, self.f)

	# takes in array ind and populates array f with features for that image
	def get_features (self, ind, f):
		for i in xrange (0, len(ind)):
			f.append(self.compute_features(ind[i]))

	# calculates value of the feature number ind
	# returns feature value
	def compute_features (self,ind):
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

		# print x,y,w,h
		if feattype == 1:
			return self.typeI(self.pic, x, y, w, h)	
		elif feattype ==2: 
			return self.typeII(self.pic, x, y, w, h)
		elif feattype ==3:
			return self.typeIII(self.pic, x, y, w, h)
		elif feattype ==4:
			return self.typeIV(self.pic, x, y, w, h)
		else:
			print 'Tried to use feature type:' + str(ind)


	# sets region of interest with coordinates x, y and a width for specific features
	def set_ROI(self, integral_image, x, y, w):
		self.origin_x = x
		self.origin_y = y		
		patch_scale = (float(w))/(float(self.min_patch_side))

		# std^2 = mean(x^2) + mean(x)^2
		mean = integral_image.findIntegral(x,y,w,w)
		mean /= (float((w*w)))

		meanSqr = integral_image.findIntegral(x,y,w,w)
		meanSqr /= (float((w*w)))

		if (meanSqr<=0):
			self.patch_std = 1
		else:
			self.patch_std = math.sqrt(math.pow(mean,2)+meanSqr)

	"""
	  Type I feature:
	  	<w->
	  ---- h
	  ++++ h
	"""
	def typeI(self, integral_image, x, y, w, h):
		sumU = integral_image.findIntegral(x,y,w,h)
		sumD = integral_image.findIntegral(x,y+h,w,h)
		# print int(sumD),int(sumU)
		if sumD > 100000 or sumU > 100000:
			print sumD,sumU
		return (int(sumD) - int(sumU))	/ self.patch_std

	"""
	  Type II feature:
	   <w-><w->
	   ++++---- ^
	   ++++---- h
	   ++++---- v
	"""
	def typeII(self, integral_image, x, y, w, h):

		sumL = integral_image.findIntegral(x,y,w,h)
		sumR = integral_image.findIntegral(x+w,y,w,h)

		return (int(sumL)-int(sumR))/self.patch_std
	
	"""
	  Type III feature:
	 	<w-><w-><w->
	   ++++----++++ ^
	   ++++----++++ h
	   ++++----++++ v
	"""
	def typeIII(self, integral_image, x, y, w, h):

		sumL = integral_image.findIntegral(x,y,w,h)
		sumC = integral_image.findIntegral(x+w,y,w,h)
		sumR = integral_image.findIntegral(x+2*w,y,w,h)
		# We have to account for the mean, since there are more (+) than (-).
		return (int(sumL)-int(sumC)+int(sumR)-(self.patch_mean*w*h)) / self.patch_std
	
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
	def typeIV(self, integral_image, x, y, w, h):
		sumLU = integral_image.findIntegral(x,y,w,h)
		sumRU = integral_image.findIntegral(x+w,y,w,h)
		sumLD = integral_image.findIntegral(x,y+h,w,h)
		sumRD = integral_image.findIntegral(x+w,y+h,w,h)
		return (-int(sumLD)+int(sumRD)+int(sumLU)-int(sumRU)) / self.patch_std