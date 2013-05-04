import numpy as np
import os
from training import Faces

class Eigenfaces:

	def __init__(self):
		self.faces = Faces()
		self.faces.train()
		self.distances = []

	def normalize (self, video_frame):
		im_matrix = self.faces.get_frame_vector(video_frame).T
		return im_matrix - self.faces.meanface

	def project (self, face):
		return self.faces.eigenfaces.T * face

	def findface(self, a):
		diff = self.faces.weights - a
		for i in diff.T:
			self.distances.append(np.linalg.norm(i-a.T))
		if min(self.distances) > 10000:
			print "Not recognized as a face"
		else:
			index = self.distances.index(min(self.distances))
			return index < 50
			# print self.distances
			#if index < 50:
			#	print "not smiling"
			#else:
			#	print "smiling"

	def classify (self, video_frame):
		self.distances = []
		normalized = self.normalize(video_frame)
		projection = self.project(normalized)
		self.findface(projection)
		



#test = Eigenfaces()
#images = os.listdir(os.getcwd()+"/picturesofjames")
#images.pop(0)
#for i in images:
#	test.classify("picturesofjames/" +i)
#test.classify("picturesofjames/img0004.jpg")
