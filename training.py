import numpy as np
import numpy.linalg as linalg
import Image

path = "/Users/georgieewuu/Documents/Freshman Spring/CS51/cs51-finalproject/images"

class Faces:

	def __init__(self):
		faces
		differencefaces
		meanface
		eigenfaces
		weights

	def get_frame_vector(self, video_frame):
		im = Image.open(video_frame)

		# convert to grayscale
		im_gray = im.convert('L')

		# convert to a matrix
		im_matrix = np.matrix(im_gray)
		return im_matrix.flatten('F')

	def get_face_images(self):
		images = os.listdir(path)

		li = [get_frame_vector("/images/" + img) for img in images]

		np.vstack(li)

	def mean_face(self, vector_list):
		return vector_list.sum(axis = 0) / float(vector_list.shape[0])

	def difference_faces(self, vector_list, mean_vector):
		return vector_list - mean_vector

	def covariance(self, vector_list):
		return np.cov(vector_list)

	def eigenfaces(self, matrix):

		# get eigenvalues and eigenvectors
		eigenvalues, eigenvectors = linalg.eig(matrix)
		
		# find the order of the eigenvalues in descending order
		order = eigenvalues.argsort()[::-1]

		# sort eigenvalues and eigenvectors according to most significant eigenvalues
		eigenvalues = eigenvalues[order]
		eigenvectors = eigenvectors[order]

		return eigenvectors * differencefaces

	def weights(self, eigenfaces):
		return faces * np.column_stack(eigenfaces)

test = Faces()
a = test.get_frame("stuff.jpg")
print test.matrix_to_vector(a)
