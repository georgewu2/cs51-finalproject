import numpy as np
from training import Faces

def normalize (video_frame):
	im_matrix = Faces.get_frame_vector(video_frame)
	return im_matrix - mean_face

def projection (face):
	return face * np.column_stack(eigenface)