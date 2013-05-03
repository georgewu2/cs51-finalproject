import numpy as np
from alttraining import Faces

class Eigenfaces:

	def __init__(self):
		self.faces = Faces()
		self.faces.main2()
		self.distances = []

	def normalize (self, video_frame):
		im_matrix = self.faces.get_frame_vector(video_frame).T
		print im_matrix - self.faces.meanface
		return im_matrix - self.faces.meanface

	def projection (self, face):
		# print "merp"
		# print self.faces.eigenfaces
		print self.faces.eigenfaces.T * face
		return self.faces.eigenfaces.T * face

	def findface(self, a):
		diff = self.faces.weights - a
		for i in diff.T:
			self.distances.append(np.linalg.norm(i-a.T))
		print self.distances
		print self.distances.index(min(self.distances))

	def main (self):
		a = self.normalize("picturesofjames/img0000.jpg")
		b = self.projection(a)
		self.findface(b)



test = Eigenfaces()
test.main()