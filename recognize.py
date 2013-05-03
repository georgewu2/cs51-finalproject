import numpy as np
import os
from alttraining import Faces

class Eigenfaces:

	def __init__(self):
		self.faces = Faces()
		self.faces.main2()
		self.distances = []

	def normalize (self, video_frame):
		im_matrix = self.faces.get_frame_vector(video_frame).T
		return im_matrix - self.faces.meanface

	def projection (self, face):
		return self.faces.eigenfaces.T * face

	def findface(self, a):
		diff = self.faces.weights - a
		for i in diff.T:
			self.distances.append(np.linalg.norm(i-a.T))
		if min(self.distances) > 10000:
			print "Not recognized as a face"
		else:
			index = self.distances.index(min(self.distances))
			# print self.distances
			if index < 50:
				print "not smiling"
			elif index < 100:
				print "smiling"
			elif index < 150:
				print "George's face"
			else:
				print "JN's face"

	def main (self):
		images = os.listdir(os.getcwd()+"/negative")
		images.pop(0)
		for i in images:
			self.distances = []
			a = self.normalize("negative/" + i)
			b = self.projection(a)
			self.findface(b)



test = Eigenfaces()
test.main()