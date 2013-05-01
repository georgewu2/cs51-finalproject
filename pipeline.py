import argparse
import cv2
import numpy as np

WINDOW_NAME = "Pipeline"
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

QUIT_KEY = 'c'

class Pipeline(object):
    """ Abstract Class for pipelines. """

    """ Command line arguments for our pipelines """
    parser = argparse.ArgumentParser(description="Face Detection")
    parser.add_argument("-pl", "--pipeline", type=str, default="full",
                        choices=["full"],
                        dest="pipeline_type", help="type of pipeline to run")

    @staticmethod
    def create(pipeline_type, **kwargs):
        # Filter out kwargs
        newkwargs = {}
        for key in kwargs:
            if kwargs[key] is not None:
                newkwargs[key] = kwargs[key]

        if pipeline_type == "full":
            return FullPipeline(**newkwargs)
        else:
            raise Exception("unsupported option: " + pipeline_type)

    def detect(self, frame1, frame2):
        """
        All implementing classes must implement this method. Returns frame1,
        annotated with the detected hand and optical flow, the overall
        direction of the scene, and the largest bounding box.
        """
        raise Exception("Must subclass implement")

class FullPipeline(Pipeline):

    def __init__(self):
        return


def main(**kwargs):
    # Read video stream from webcam
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print "couldn't load webcam"
        return
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Create pipeline from command line arguments
    pipeline = Pipeline.create(**kwargs)

    while True:
        retval1, frame1 = capture.read()
        retval2, frame2 = capture.read()
        if not retval1 or not retval2:
            print "could not grab frames"
            return

        # Mirror the frames to mimic webcam motion
        frame1 = cv2.flip(frame1, 1)
        frame2 = cv2.flip(frame2, 1)

        # Detect hand and direction of the scene
        largest, direction, frame_out = pipeline.detect(frame1, frame2)
        print(direction)
        cv2.imshow(WINDOW_NAME, frame_out)

        # Handlers for key presses
        c = cv2.waitKey(10)
        if chr(c & 255) is QUIT_KEY:
            break

if __name__ == "__main__":
    args = Pipeline.parser.parse_args()
    main(**vars(args))
