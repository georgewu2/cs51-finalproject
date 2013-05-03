import numpy
import scipy
from training import Faces

class Integrate:

	def __init__(self,img):
		self.faces = Faces()
		self.img = img
		self.integral_img = self.integral_image()

	
	# x is an ndarray of the input 
	# returns an ndarray that is an integral image / summed area table
	def integral_image (self):
		return self.img.cumsum(1).cumsum(0)

	# x, y are the coordinates of the corner of the box to be summed
	# w, h are the width and height of the rectangle to be summed
	def findIntegral(self, x,y,w,h):
		if (x>0 and y>0):
			A = self.integral_img[x-1, y-1]
		else:
			A = 0

		if (y>0):
			B = self.integral_img[x+w-1, y-1]
		else:
			B = 0
			
		if (x>0):
			D = self.integral_img[x-1, y+h-1]
		else:
			D = 0
			
		C = self.integral_img[x+w-1, y+h-1]

		return A+C-B-D

