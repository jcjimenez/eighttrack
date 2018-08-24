import unittest
import os
import sys
import cv2

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eighttrack import *
from eighttrack.cv2_tracking import *


class OpencvTrackedObjectTest(unittest.TestCase):
    def setUp(self):
        self.generator = VideoCaptureGenerator(os.path.join(
            os.path.dirname(__file__),
            'data',
            'clip.m4v'
        ))
        frame = next(self.generator)
        self.tracked_object = OpencvTrackedObject(
            "someid",
            BoundingBox(325, 85, 240, 240),
            frame
        )

    def test_initial_state(self):
        self.assertEqual(
            self.tracked_object.last_known_location,
            BoundingBox(325, 85, 240, 240)
        )

    def test_update_single_step(self):
        self.tracked_object.update(next(self.generator))
        self.assertEqual(
            self.tracked_object.last_known_location.as_origin_and_size(),
            BoundingBox(325, 85, 240, 240).as_origin_and_size()
        )
        self.assertEqual(
            self.tracked_object.state,
            TrackedObjectState.TRACKING
        )

    def test_update_last_known_location(self):
        for index in range(160):
            next(self.generator)
        for index in range(20):
            frame = next(self.generator)
            (tracking, box) = self.tracked_object.update(frame)

        self.assertEqual(
            self.tracked_object.state,
            TrackedObjectState.TRACKING
        )
        expected_bounding_box = BoundingBox(327, 85, 240, 240)
        self.assertEqual(
            self.tracked_object.last_known_location.as_origin_and_size(),
            expected_bounding_box.as_origin_and_size()
        )

    def test_update_for_missing(self):
        for index in range(220):
            next(self.generator)
        for index in range(20):
            frame = next(self.generator)
            (updated_tracking, updated_box) = self.tracked_object.update(frame)

        self.assertFalse(updated_tracking)
        self.assertEqual(
            self.tracked_object.state,
            TrackedObjectState.MISSING
        )
        expected_bounding_box = BoundingBox(325, 85, 240, 240)
        self.assertEqual(
            updated_box,
            expected_bounding_box
        )
        self.assertEqual(
            self.tracked_object.last_known_location,
            expected_bounding_box
        )

    def test_update_full_video(self):
        for index in range(220):
            next(self.generator)
        for index in range(20):
            frame = next(self.generator)
            self.tracked_object.update(frame)
        self.assertEqual(
            self.tracked_object.state,
            TrackedObjectState.MISSING
        )

        for index in range(220):
            next(self.generator)
        for frame in self.generator:
            self.tracked_object.update(frame)

        self.assertEqual(
            self.tracked_object.state,
            TrackedObjectState.TRACKING
        )
        expected_bounding_box = BoundingBox(325, 71, 240, 240)
        self.assertEqual(
            self.tracked_object.last_known_location.as_origin_and_size(),
            expected_bounding_box.as_origin_and_size()
        )


class OpencvObjectTrackerTest(unittest.TestCase):
    def setUp(self):
        self.generator = VideoCaptureGenerator(os.path.join(
            os.path.dirname(__file__),
            'data',
            'clip.m4v'
        ))
        self.first_bounding_box = BoundingBox(325, 85, 240, 240)
        self.tracker = OpencvObjectTracker()

    def test_default_state(self):
        self.assertEqual(self.tracker.tracked_objects, list())

    def test_add_single(self):
        objects_added = self.tracker.add(
            [DetectedObject('face', 0.99, self.first_bounding_box)],
            next(self.generator)
        )
        self.assertEqual(len(objects_added), 1)
        self.assertEqual(objects_added, self.tracker.tracked_objects)

        tracked_object = list(objects_added)[0]
        self.assertIsNotNone(tracked_object.object_id)
        self.assertEqual(tracked_object.state, TrackedObjectState.TRACKING)
        self.assertEqual(
            tracked_object.first_known_location,
            self.first_bounding_box
        )
        self.assertEqual(
            tracked_object.last_known_location,
            self.first_bounding_box
        )

    def test_call_single(self):
        frame = next(self.generator)
        frame.detected_objects = set(
            [DetectedObject('face', 0.99, self.first_bounding_box)])
        updated_frame = self.tracker(frame)
        self.assertIsNotNone(updated_frame)

        self.assertEqual(len(self.tracker.tracked_objects), 1)
        self.assertEqual(
            updated_frame.tracked_objects,
            self.tracker.tracked_objects
        )

        tracked_object = list(self.tracker.tracked_objects)[0]
        self.assertIsNotNone(tracked_object.object_id)
        self.assertEqual(tracked_object.state, TrackedObjectState.TRACKING)
        self.assertEqual(
            tracked_object.first_known_location,
            self.first_bounding_box
        )
        self.assertEqual(
            tracked_object.last_known_location,
            self.first_bounding_box
        )

    def test_call_two_overlapping_high_iou(self):
        frame = next(self.generator)
        frame.detected_objects = [
            DetectedObject(
                label='face',
                score=0.99,
                bounding_box=BoundingBox(320, 80, 244, 244)
            ),
            DetectedObject(
                label='face',
                score=0.99,
                bounding_box=BoundingBox(325, 85, 240, 240)
            )
        ]
        updated_frame = self.tracker(frame)
        self.assertIsNotNone(updated_frame)

        # Since the two detected objects are so close together, they are treated
        # as a single one.
        self.assertEqual(len(self.tracker.tracked_objects), 1)
        self.assertEqual(
            updated_frame.tracked_objects,
            self.tracker.tracked_objects
        )

        tracked_object = list(self.tracker.tracked_objects)[0]
        self.assertIsNotNone(tracked_object.object_id)
        self.assertEqual(tracked_object.state, TrackedObjectState.TRACKING)
        self.assertEqual(
            tracked_object.first_known_location,
            BoundingBox(320, 80, 244, 244)
        )
        self.assertEqual(
            tracked_object.first_known_location,
            tracked_object.last_known_location
        )

    def test_two_objects_crossing_paths(self):
        animation = VideoCaptureGenerator(os.path.join(
            os.path.dirname(__file__),
            'data',
            'simple_animation.mp4'
        ))
        frame = next(animation)
        frame.detected_objects = [
            DetectedObject(
                'rectangle',
                0.99,
                BoundingBox(178, 100, 230, 210)
            ),
            DetectedObject(
                'oval',
                0.99,
                BoundingBox(1190, 100, 224, 200)
            ),
        ]
        updated_frame = self.tracker(frame)
        self.assertIsNotNone(updated_frame)

        self.assertEqual(len(self.tracker.tracked_objects), 2)
        self.assertEqual(
            updated_frame.tracked_objects,
            self.tracker.tracked_objects
        )

        for frame in animation:
            self.tracker(frame)

        self.assertEqual(len(self.tracker.tracked_objects), 2)
        self.assertEqual(
            updated_frame.tracked_objects,
            self.tracker.tracked_objects
        )
        center_x = (1928/2)
        left_objects = list(filter(
            lambda to: to.last_known_location.x < center_x,
            self.tracker.tracked_objects
        ))
        self.assertTrue(len(left_objects), 1)
        self.assertEqual(
            left_objects[0].last_known_location.as_origin_and_size(),
            (654, 364, 230, 210)
        )

        right_objects = list(filter(
            lambda to: to.last_known_location.x > center_x,
            self.tracker.tracked_objects
        ))
        self.assertTrue(len(right_objects), 1)
        self.assertEqual(
            right_objects[0].last_known_location.as_origin_and_size(),
            (1190, 100, 224, 200)
        )


if __name__ == '__main__':
    unittest.main()
