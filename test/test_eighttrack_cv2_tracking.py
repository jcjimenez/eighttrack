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


if __name__ == '__main__':
    unittest.main()
