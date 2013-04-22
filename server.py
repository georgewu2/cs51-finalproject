import cv2
NUM_FRAMES = 50
cv2.namedWindow("preview")
capture = cv2.VideoCapture(0)

if capture.isOpened(): # try to get the first frame
    rval, frame = capture.read()
else:
    rval = False

if rval:
    for i in range (-1, NUM_FRAMES):
        frame = cv2.flip(frame,1)
        cv2.imwrite("data/img%04d.jpg" % i, frame)
        cv2.imshow("preview", frame)
        rval, frame = capture.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break