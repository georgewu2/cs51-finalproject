import numpy
import scipy

def integral_image (x):
	return x.cumsum(1).cumsum(0)

# r0, c0 is the top left corner of the block to be summed
# r1,c1 is the bottom right corner of the block
def integrate (image, r0, c0, r1, c1):
	S = 0

	S += image[r1, c1]

	if(r0-1 >= 0) and (c0-1 >= 0):
		S +=image[r0-1, c0-1]

	if (r0-1 >=0):
		S-=image[r0-1, c1]

	if (c0-1>=0):
		S-=image[r1, c0-1]

	return S


def findIntegral(x,y,w,h):
	
	if (x>0 and y>0):
		A = img[x-1, y-1]
	else:
		A = 0

	if (y>0):
		B = img[x+w-1, y-1]
	else:
		B = 0
		
	if (x>0):
		D = img[x-1, y+h-1]
	else:
		D = 0
		
	C = img[x+w-1, y+h-1]

	return A+C-B-D

