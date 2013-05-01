import numpy
import scipy
import training

class Integrate:

	def __init__(self):
		original_img = self.faces.get_frame_vector("face_1.jpg")
		integral_img = self.integral_image(original_img)

	
	# x is an ndarray of the input 
	# returns an ndarray that is an integral image / summed area table
	def integral_image (self, x):
		return x.cumsum(1).cumsum(0)

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

