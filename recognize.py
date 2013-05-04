import numpy as np
import os
from training import Faces

class Eigenfaces:

	def __init__(self):
		self.faces = Faces()
		self.faces.train()
		self.distances = []

	# converts picture into vector and subtracts mean vector from it
	def normalize (self, video_frame):
		im_matrix = self.faces.get_frame_vector(video_frame).T
		return im_matrix - self.faces.meanface

	# projects vector into facespace
	def project (self, face):
		return self.faces.eigenfaces.T * face

	# determines if there is a recognized face
	def findface(self, a):
		diff = self.faces.weights - a
		
		for i in diff.T:
			self.distances.append(np.linalg.norm(i-a.T))
		
		if min(self.distances) > 10000:
			print "Not recognized as a face"
		
		else:
			index = self.distances.index(min(self.distances))
			return index < 50

	def classify (self, video_frame):
		self.distances = []
		normalized = self.normalize(video_frame)
		projection = self.project(normalized)
		self.findface(projection)