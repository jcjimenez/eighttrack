name = "eighttrack"
__import__('pkg_resources').declare_namespace(__name__)

import cv2
import datetime
import math
import os
import uuid


class VideoFrame(object):
    '''
    Represents a single frame of video.
    '''

    def __init__(self, pixels, detected_objects=set(), tracked_objects=set()):
        self.pixels = pixels
        self.detected_objects = detected_objects
        self.tracked_objects = tracked_objects


class BoundingBox(object):
    def __init__(self, x, y, width, height, round_values=True):
        assert width >= 0
        assert height >= 0

        if round_values:
            x = int(round(x))
            y = int(round(y))
            width = int(round(width))
            height = int(round(height))

        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.x1 = x
        self.x2 = x + width
        self.y1 = y
        self.y2 = y + height

        self.pt1 = (self.x1, self.y1)
        self.pt2 = (self.x2, self.y2)

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            return False

        return self.x == other.x and \
            self.y == other.y and \
            self.width == other.width \
            and self.height == other.height

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.as_origin_and_size())

    def __sub__(self, other):
        local_center = self.center()
        other_center = other.center()
        return math.sqrt(
            math.pow((local_center[0] - other_center[0]), 2) +
            math.pow((local_center[1] - other_center[1]), 2)
        )

    def __str__(self):
        return str(self.as_origin_and_size())

    def center(self):
        return (self.x + self.width/2.0, self.y + self.height/2.0)

    def as_point_pair(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def as_origin_and_size(self):
        return (self.x, self.y, self.width, self.height)

    def as_left_bottom_right_top(self):
        '''
        Returns a tuple in Quadrant I coordinate space in the form:
        (left, top, bottom, right).
        '''
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def iou(self, other):
        '''
        Returns a float representing the intersection over union ratio between
        the receiver and the given bounding box (other).
        '''
        union = self.union(other)
        if union == 0:
            return 0.0
        intersection = self.intersection(other)
        return (intersection / float(union))

    def intersection(self, other):
        '''
        Returns a float representing the area of intersection between the
        reciever and the given bounding box.
        '''
        if self.x2 < other.x1 or other.x2 < self.x1:
            # No intersection in x
            return 0
        if self.y2 < other.y1 or other.y2 < self.y1:
            # No intersection in y
            return 0
        minX = max(self.x1, other.x1)
        minY = max(self.y1, other.y1)
        maxX = min(self.x2, other.x2)
        maxY = min(self.y2, other.y2)
        return float(max(0, maxX - minX) * max(0, maxY - minY))

    def union(self, other):
        '''
        Returns a float representing the area of union between the receiver and
        the given bounding box.
        '''
        return float((max(other.x2, self.x2) - min(other.x1, self.x1)) * (max(other.y2, self.y2) - min(other.y1, self.y1)))


class DetectedObject(object):
    '''
    Represents a simple detected object in a given frame of video.
    '''

    def __init__(self, label, score, bounding_box, object_id=None):
        self.label = label
        self.score = score
        self.bounding_box = bounding_box
        self.object_id = object_id if object_id else uuid.uuid4()

    def __eq__(self, other):
        if not isinstance(other, DetectedObject):
            return False

        return self.label == other.label and \
            self.score == other.score and \
            self.bounding_box == other.bounding_box and \
            self.object_id == other.object_id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.object_id)


class TrackedObjectState(object):
    '''
    TrackedObjectState represents the possible states of a tracked object.

    TRACKING: the object is confidently located by the object tracker.
    MISSING:  the object may have been occluded or dissapeared from the video.
    LOST:     the object has been missing for long enough that the tracker has
              given up on it.
    '''
    TRACKING = 1
    MISSING = 2
    LOST = 3


class TrackedObject(object):
    def __init__(self, object_id, bounding_box, recovery_threshold_in_seconds=15):
        self.object_id = object_id if object_id else uuid.uuid4()
        self.recovery_threshold_in_seconds = recovery_threshold_in_seconds
        self.state = TrackedObjectState.TRACKING

        self.first_known_location = bounding_box
        self.first_known_location_timestamp = datetime.datetime.utcnow()
        self.last_known_location = bounding_box
        self.last_known_location_timestamp = datetime.datetime.utcnow()

    def set_last_known_location(self, bounding_box):
        self.last_known_location = bounding_box
        self.last_known_location_timestamp = datetime.datetime.utcnow()
        self.state = TrackedObjectState.TRACKING

    def report_missing(self):
        '''
        Updates the receiver's state to one of the following:

        MISSING - if the current state of the receiver is TRACKING.
        LOST    - if the current state of the receiver is MISSING and enough
                  time has passed to consider it lost.
        '''
        if self.state == TrackedObjectState.LOST:
            # Nothing to do since LOST is an end state not meant for recovery.
            return

        if self.seconds_since_last_known_location() > self.recovery_threshold_in_seconds:
            self.state = TrackedObjectState.LOST
            return

        self.state = TrackedObjectState.MISSING

    def state_str(self):
        return {
            TrackedObjectState.TRACKING: "TRACKING",
            TrackedObjectState.MISSING: "MISSING",
            TrackedObjectState.LOST: "LOST",
        }.get(self.state, "UNKNOWN")

    def age_in_seconds(self):
        current_timestamp = datetime.datetime.utcnow()
        delta = current_timestamp - self.first_known_location_timestamp
        return delta.total_seconds()

    def seconds_since_last_known_location(self):
        current_timestamp = datetime.datetime.utcnow()
        delta = current_timestamp - self.last_known_location_timestamp
        return delta.total_seconds()

    def total_distance_traveled(self):
        return abs(self.last_known_location - self.first_known_location)

    def __eq__(self, other):
        if not isinstance(other, DetectedObject):
            return False

        return self.object_id == other.object_id and \
            self.state == other.state and \
            self.first_known_location == other.first_known_location and \
            self.last_known_location == other.last_known_location

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.object_id)


class Pipeline(object):
    '''
    A Pipleline represents a series made up by a video source (in the form of a
    Python generator of VideoFrame instances) followed by a list of steps
    (Python callables taking a VideoFrame instance as input).
    '''

    def __init__(self, source=None):
        '''
        Constructor that takes an optional VideoFrame generator.
        '''
        self._generator = source
        self._steps = []

    def add(self, step):
        '''
        Adds a given callable to the pipeline represented by the receiver.
        '''
        if None == self._generator:
            self._generator = step
        else:
            self._steps.append(step)
        return self

    def _assemble(self):
        # assert self._generator != None
        # assert len(self._steps) > 0
        last = self._generator
        for current in self._steps:
            transformed = map(current, last)
            last = transformed
        return last

    def run(self):
        '''
        Runs the video source generator and pipeline step callables in series.
        '''
        generator = self._assemble()
        while True:
            try:
                next(generator)
            except StopIteration:
                return self
            finally:
                pass
        return self


class VideoCaptureGenerator(object):
    '''
    A VideoCaptureGenerator is the simplest pipeline source. It is meant as an example
    generator around cv2.VideoCapture#read().
    '''

    def __init__(self, url):
        self.capture = cv2.VideoCapture(url)

    def __del__(self):
        self.capture.release()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.capture.isOpened():
            raise StopIteration()

        (ok, frame) = self.capture.read()
        if not ok:
            raise StopIteration()

        return VideoFrame(frame)


class VideoDisplaySink(object):
    '''
    A VideoDisplaySink is a simple callable that can be used as a sink to a
    pipeline.
    '''

    def __init__(self, name='video'):
        self.name = name

    def __call__(self, frame):
        cv2.imshow(self.name, frame.pixels)
        cv2.waitKey(1)
        return frame


class DetectedObjectDebugger(object):
    '''
    Draws the detected object bounding boxes on each video frame.
    '''

    def __call__(self, frame):
        for detected in frame.detected_objects:
            box = detected.bounding_box
            cv2.rectangle(
                frame.pixels,
                box.pt1,
                box.pt2,
                (0, 255, 0)
            )
        return frame
