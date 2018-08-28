import cv2
import os

from . import DetectedObject, BoundingBox

CASCADE_FACE_DETECTOR_DEFAULT_XML_PATH = os.environ.get(
    'EIGHTTRACK_CASCADE_FACE_DETECTOR_DEFAULT_XML_PATH',
    '/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
)
'''
CASCADE_FACE_DETECTOR_DEFAULT_XML_PATH can be set to the following are known
paths for useful cascades:

  opencv-docker image: /usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml
  homebrew: /usr/local/Cellar/opencv/3.4.1_6/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml
  
'''


class CascadeDetector(object):
    '''
    A CascadeDetector is a simple wrapper around OpenCV's CascadeClassifier
    initialized with one of the included a face detection XML files.
    '''

    def __init__(self, scale_factor=1.5, min_neighbors=8, min_size=(16, 16), flags=cv2.CASCADE_SCALE_IMAGE, haar_path=CASCADE_FACE_DETECTOR_DEFAULT_XML_PATH):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.flags = flags
        self.classifier = cv2.CascadeClassifier(haar_path)

    def detect(self, frame):
        objects = self.classifier.detectMultiScale(
            frame,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=self.flags
        )
        return map(
            lambda rect: DetectedObject(
                label='face',
                score=0.99,
                bounding_box=BoundingBox(
                    rect[0],
                    rect[1],
                    rect[2],
                    rect[3]
                )
            ),
            objects
        )

    def __call__(self, frame):
        objects = self.detect(frame.pixels)
        frame.detected_objects = frame.detected_objects.union(objects)
        return frame
