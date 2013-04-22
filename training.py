import numpy as np
import Image

class Faces:

	def __init__(self):
		dictionaryfaces = {}

	def get_frame(self, video_frame):
		im = Image.open(video_frame)
		im_gray = im.convert('L')
		im_array = np.array(im_gray)
		return im_array


	def matrix_to_vector(self, matrix):
		return matrix.flatten('F')

	def mean_face(self, vector_list):
		return vector_list.sum(axis = 0) / float(vector_list.shape[0])

	def difference_faces(self, vector_list, mean_vector):
		for vector in vector_list:
			vector = vector - mean_vector
		return vector_list

	def covariance(self, vector_list):
		vector_list.cov()

	# def eigenbasis(self, matrix):

test = Faces()
a = test.get_frame("stuff.jpg")
test.matrix_to_vector(a)