import numpy as np
import Image

class Faces:

	def __init__(self):
		dictionaryfaces = {}

	def get_frame(self, video_frame):
		im = Image.open(video_frame)
		im_gray = im.convert('L')
		im_array = np.array(im_gray)
		print im_array
		return im_array


	def matrix_to_vector(self, matrix):
		return matrix.flatten('F')

	def mean_face(self, vector_list):
		return vector_list.sum(axis = 0) / float(vector_list.shape[0])

	def difference_faces(self, vector_list, mean_vector):
		vector_list - mean_vector

	def covariance(self, vector_list):
		np.cov(vector_list)

	def eigenbasis(self, matrix):
		eigenvalues = np.linalg.eig(matrix)


test = Faces()
a = test.get_frame("stuff.jpg")
print test.matrix_to_vector(a)
