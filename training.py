import numpy as np
import numpy.linalg as linalg
import Image
import os

class Faces:

	def __init__(self):
		self.listfaces = []
		self.faces = None
		self.meanface = None
		self.differencefaces = None
		self.covmatrix = None
		self.eigenfaces = None
		self.weights = None

	def get_frame_vector(self, video_frame):
		im = Image.open(video_frame)

		# convert to grayscale
		im_gray = im.convert('L')

		# convert to a matrix
		im_matrix = np.matrix(im_gray)
		return im_matrix.flatten('F')

	def get_face_images(self):
		# get all files of image directory
		images = os.listdir(os.getcwd() + "/images")
		
		# get rid of the .DS_Store file
		images.pop(0)

		# add each vector to the list
		for i in images:
			self.listfaces.append(self.get_frame_vector("images/" + i))

		# compress vector list into one matrix
		self.faces = np.concatenate(self.listfaces)

	def mean_face(self):
		# np.mean IS SO SLOW WHYYYYYYY
		self.meanface = np.mean(self.faces, axis = 0)

	def difference_faces(self):
		self.differencefaces = self.faces - self.meanface

	def covariance(self):
		self.covmatrix = np.cov(self.differencefaces)

	def get_eigenfaces(self):

		# get eigenvalues and eigenvectors
		eigenvalues, eigenvectors = linalg.eig(self.covmatrix)
		
		# find the order of the eigenvalues in descending order
		order = eigenvalues.argsort()[::-1]

		# sort eigenvalues and eigenvectors according to most significant eigenvalues
		eigenvalues = eigenvalues[order]
		eigenvectors = eigenvectors[order]

		self.eigenfaces = eigenvectors * self.differencefaces

	def get_weights(self):
		self.weights = self.faces * self.eigenfaces.T

	def main(self):
		self.get_face_images()
		# print self.faces
		self.mean_face()
		# print self.meanface
		self.difference_faces()
		# print self.differencefaces
		self.covariance()
		# print self.covmatrix
		self.get_eigenfaces()
		# print self.eigenfaces
		self.get_weights()
		print self.weights

test = Faces()
test.main()
# a = test.get_frame("stuff.jpg")
# print test.matrix_to_vector(a)
