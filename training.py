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

	def get_frame_vector(self, video_frame, flatten=True):
		im = Image.open(video_frame)

		# convert to grayscale
		im_gray = im.convert('L')

		# convert to a matrix
		im_matrix = np.matrix(im_gray)
		if flatten == True:
			return im_matrix.flatten('F')
		else:
			return im_matrix

	def get_face_images(self, path):
		# get all files of image directory
		images = os.listdir(os.getcwd() + path)
		
		# get rid of the .DS_Store file
		images.pop(0)

		# add each vector to the list
		for i in images:
			self.listfaces.append(self.get_frame_vector("data/analysis/" + i).T)

		# compress vector list into one matrix
		self.faces = np.concatenate(self.listfaces, axis = 1)

	def mean_face(self):
		self.meanface = np.mean(self.faces, axis = 1)

	def difference_faces(self):
		self.differencefaces = self.faces - self.meanface

	def covariance(self):
		self.covmatrix = self.differencefaces.T * self.differencefaces

	def get_eigenfaces(self):

		# get eigenvalues and eigenvectors
		eigenvalues, eigenvectors = linalg.eig(self.covmatrix)
		
		# find the order of the eigenvalues in descending order
		order = eigenvalues.argsort()[::-1]

		# sort eigenvalues and eigenvectors according to most significant eigenvalues
		eigenvalues = eigenvalues[order]
		eigenvectors = eigenvectors[:,order]

		# get only the 10 most significant eigenvectors
		eigenvectors = eigenvectors[:,:10]

		# create the eigenfaces
		eigenfaces = self.differencefaces * eigenvectors
		# normalize each eigenface

		# get the tranpose
		temp = eigenfaces.T
		vectors = []
		for face in temp:
			
			# get the length of the row and divide each row by it
			length = linalg.norm(face)
			nface = face / length
			vectors.append(nface)

		# get the transpose of the normalized eigenfaces
		neigenfaces = np.concatenate(vectors).T
		self.eigenfaces = neigenfaces

	def get_weights(self):
		self.weights = self.eigenfaces.T * self.faces

	def train(self):
		self.get_face_images("/data/analysis")
		self.mean_face()
		self.difference_faces()
		self.covariance()
		self.get_eigenfaces()
		self.get_weights()