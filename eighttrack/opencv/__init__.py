import cv2
import datetime
import math
import os
import random
import rtree
import uuid

from .. import BoundingBox, DetectedObject, TrackedObject, VideoFrame, TrackedObjectState

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

DEFAULT_TRACKER_IOU_THRESHOLD = float(os.environ.get(
    'EIGHTTRACK_CV2_TRACKER_IOU_THRESHOLD',
    '0.33'
))
'''
DEFAULT_TRACKER_IOU_THRESHOLD is the default IOU ratio threshold that is used
to match incoming detected object bounding boxes to existing known objects
(that are already being tracked).
'''

DEFAULT_TRACKER_RECOVERY_THRESHOLD_IN_SECONDS = float(os.environ.get(
    'EIGHTTRACK_CV2_TRACKER_RECOVERY_THRESHOLD_IN_SECONDS',
    '7'
))
'''
DEFAULT_TRACKER_RECOVERY_THRESHOLD_IN_SECONDS indicates how long a tracked
object can be missing before the object tracker gives up on it.
'''


class CascadeDetector(object):
    '''
    A CascadeDetector is a simple wrapper around OpenCV's CascadeClassifier
    initialized with one of the included a face detection XML files.
    '''

    def __init__(self, scale_factor=1.5, min_neighbors=8, min_size=(16, 16), flags=cv2.CASCADE_SCALE_IMAGE, haar_path=CASCADE_FACE_DETECTOR_DEFAULT_XML_PATH):
        if not os.path.isfile(haar_path):
            raise ValueError(
                "{} is not a valid HAAR xml file.".format(haar_path))
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


class OpencvTrackedObject(TrackedObject):
    '''
    TrackedObject subclass that wraps an Open CV object tracker and by virtue
    of being a TrackedObject facilitates things like status change
    messaging and time tracking.
    '''

    def __init__(self, object_id, bounding_box, frame, recovery_threshold_in_seconds=DEFAULT_TRACKER_RECOVERY_THRESHOLD_IN_SECONDS):
        TrackedObject.__init__(
            self,
            object_id,
            bounding_box,
            recovery_threshold_in_seconds
        )
        self._initialize_tracker(frame.pixels)

    def _initialize_tracker(self, frame):
        self._tracker = cv2.TrackerKCF_create()
        self._tracker.init(
            image=frame,
            boundingBox=self.first_known_location.as_origin_and_size()
        )

    def update(self, frame):
        ok, updated_box = self._tracker.update(frame.pixels)
        if not ok:
            self.report_missing()
            return (False, self.last_known_location)
        else:
            updated_bounding_box = BoundingBox(
                updated_box[0],
                updated_box[1],
                updated_box[2],
                updated_box[3],
            )
            self.set_last_known_location(updated_bounding_box)
            return (True, self.last_known_location)

    def attempt_recovery(self, bounding_box, frame):
        if self.state == TrackedObjectState.LOST:
            return (False, self.last_known_location)

        self.set_last_known_location(bounding_box)
        self._initialize_tracker(frame.pixels)
        return (True, self.last_known_location)


class OpencvObjectTracker(object):
    '''
    The OpencvObjectTracker is similar to Open CV's multitracker but in this
    case contains TrackedObject instances that allow keeping track of time.
    '''

    def __init__(self, recovery_threshold_in_seconds=DEFAULT_TRACKER_RECOVERY_THRESHOLD_IN_SECONDS):
        '''
        Initializes ther receiver with an empty list of tracked objects.
        '''
        self.tracked_objects = list()
        self.index = rtree.index.Index()
        self.box_iou_threshold = DEFAULT_TRACKER_IOU_THRESHOLD
        self.recovery_threshold_in_seconds = recovery_threshold_in_seconds

    def __call__(self, frame):
        self.add(frame.detected_objects, frame)
        self.update(frame)
        return VideoFrame(
            frame.pixels,
            detected_objects=frame.detected_objects,
            tracked_objects=self.tracked_objects
        )

    def get(self, bounding_box):
        '''
        Returns a tracked object for the given bounding box if one is present.
        '''
        intersections = self.index.intersection(
            bounding_box.as_left_bottom_right_top(),
            objects=True
        )
        intersections = filter(
            lambda xsect: self.tracked_objects[xsect.id].last_known_location.iou(
                bounding_box) > self.box_iou_threshold,
            intersections
        )
        intersections = sorted(
            intersections,
            key=lambda xsect: self.tracked_objects[xsect.id].last_known_location.iou(
                bounding_box
            ),
            reverse=True
        )
        if len(intersections) > 0:
            tracked_object_index = intersections[0].id
            return self.tracked_objects[tracked_object_index]

        return None

    def add(self, detected_objects, frame):
        bounding_boxes = list()
        try:
            bounding_boxes = map(lambda do: do.bounding_box, detected_objects)
        except TypeError:
            raise Exception(
                "Parameter detected_objects must be an iterable of DetectedObject instances."
            )

        boxes_and_objects = map(
            lambda box: (box, self.get(box)),
            bounding_boxes
        )

        result = list()
        for (box, tracked_object) in boxes_and_objects:
            if not tracked_object:
                # The bounding box did not intersect (above an IOU) with any
                # known objects, so this means this is a new object that needs
                # to be tracked.
                result.append(self._append_box(box, frame))
                continue

            if tracked_object.state == TrackedObjectState.TRACKING:
                continue

            # The bounding box matches a known object whose state is either
            # MISSING or LOST so an attempt should be made to recover the
            # object and have it go back to a state of TRACKING.
            (recovered, _) = tracked_object.attempt_recovery(box, frame)

            if not recovered:
                # Looks like the known object could not be recovered, so a
                # brand-new object should be created for the incoming bounding
                # box.
                result.append(self._append_box(box, frame))
                continue

        return result

    def remove_lost_objects(self):
        objects_to_remove = filter(
            lambda tracked_object: tracked_object.state == TrackedObjectState.LOST,
            self.tracked_objects
        )
        self.remove(objects_to_remove)

    def remove(self, objects_to_remove):
        for tracked_object in objects_to_remove:
            self.tracked_objects.remove(tracked_object)
        self.index = self._create_index()

    def _append_box(self, box, frame):
        tracked_obj = OpencvTrackedObject(
            str(uuid.uuid4()),
            box,
            frame,
            self.recovery_threshold_in_seconds
        )
        self._append_tracked_object(tracked_obj)
        return tracked_obj

    def _append_tracked_object(self, tracked_obj):
        self.tracked_objects.append(tracked_obj)

        # NOTE: The id of the object in the index is its index in the
        # tracked_objects list. This is because the rtree index expects an int.
        self.index.add(
            len(self.tracked_objects) - 1,
            tracked_obj.last_known_location.as_left_bottom_right_top()
        )

    def _create_index(self):
        if not self.tracked_objects:
            return rtree.index.Index()

        def index_data_generator():
            for (tracked_index, tracked_obj) in enumerate(self.tracked_objects):
                yield (
                    tracked_index,
                    tracked_obj.last_known_location.as_left_bottom_right_top(),
                    None
                )
        return rtree.index.Index('tracker', index_data_generator())

    def update(self, frame):
        for tracked_obj in self.tracked_objects:
            tracked_obj.update(frame)
        self.index = self._create_index()
        return self.tracked_objects
