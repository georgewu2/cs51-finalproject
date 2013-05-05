import base64
import cv2
import os
import pipeline
from adaboost import adaBoost,Cascade
import adaboost
import threading
import tornado.ioloop
import tornado.web
import tornado.websocket
from recognize import Eigenfaces

# Directory for temporary files
DIRECTORY = os.path.join(os.path.dirname(__file__), "tmp")

# Pipeline to run on this server
PIPELINE = None
eigenfaces = Eigenfaces()
cascader = Cascade()
cascader.trainCascadedClassifier(.2,.6,.2)
# Sets the false positive threshold (0.2 in this case)


DIRECTORY = os.path.join(os.path.dirname(__file__), "tmp")
class MainHandler(tornado.web.RequestHandler):
    """
    Handles the request for the default index.html web page.
    """
    def get(self):
        self.render("index.html")

class VideoWebSocketHandler(tornado.websocket.WebSocketHandler):
    """
    Creates a web socket connection to the client for receiving frames
    and sending back the annotated frame and overall motion of the scene.
    """

    # Websocket ID
    id = 0
    # Variable for counting the image
    frames = 0
    lock = threading.Lock()
    
    def open(self):
        
        with self.__class__.lock:
            self.__class__.id += 1
            self.id = self.__class__.id
        self.file_in = DIRECTORY + "/in" + str(self.id) + ".jpeg"
        self.prev = None
        if not os.path.exists(DIRECTORY):
            os.makedirs(DIRECTORY)
        print "Websocket " + str(self.id) + " opened"

    def on_message(self, message):
        newim = self.parse_image(message)
        if self.prev == None:
            self.prev = newim
        else:
	    cv2.imwrite("../data/analysis/img%04d.jpg" % self.frames, newim)
            # cv2.imwrite("../data/sameface/img%04d.jpg" % self.frames, newim)
            # Only print message after the 100th image
            if (self.frames >= 100):
                # Run adaboost to see if there is a face or not
                if (cascader.cascadedClassify(adaboost.get_frame_vector("../data/analysis/img%04d.jpg" % self.frames, False))):
                    self.write_message(u"Yes, face")
                    # Detect if the face is smiling or not
                    if eigenfaces.classify("../data/analysis/img%04d.jpg" % self.frames):
			self.write_message(u"Yes, face")
		    else:
			self.write_message(u"Yes, smiling face")
                else:
                    self.write_message(u"No face")
                # Delete img100.jpg and replace with new file
                os.remove("../data/analysis/img%04d.jpg" % self.frames)
            if (self.frames < 100):
                self.frames += 1

            self.prev = newim

    def on_close(self):
        if os.path.isfile(self.file_in):
            os.remove(self.file_in)
        print "Websocket " + str(self.id) + " closed"

    def parse_image(self, buf):
        img_str = base64.b64decode(buf)
        img = open(self.file_in, "w+");
        img.write(base64.b64decode(buf))
        img.close()
        return cv2.imread(self.file_in)

settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
}

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/websocket", VideoWebSocketHandler),
    (r"/static", tornado.web.StaticFileHandler, dict(path=settings['static_path'])),
], **settings)

if __name__ == "__main__":
    pipeline.Pipeline.parser.add_argument("-p", "--port", type=int,
                                          default=8888, dest="port",
                                          help="port to listen to")
    args = pipeline.Pipeline.parser.parse_args()
    kwargs = vars(args)
    application.listen(kwargs["port"])
    del kwargs["port"]
    PIPELINE = pipeline.Pipeline.create(**kwargs)
    tornado.ioloop.IOLoop.instance().start()
