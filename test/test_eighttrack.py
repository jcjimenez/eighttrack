import unittest
import os
import sys

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eighttrack import *


class BoundingBoxTest(unittest.TestCase):
    def test_simple(self):
        box = BoundingBox(10, 20, 300, 400)
        self.assertEqual(box.x, 10)
        self.assertEqual(box.y, 20)
        self.assertEqual(box.width, 300)
        self.assertEqual(box.height, 400)

        self.assertEqual(box.x1, 10)
        self.assertEqual(box.y1, 20)
        self.assertEqual(box.x2, 310)
        self.assertEqual(box.y2, 420)

        self.assertEqual(box.as_point_pair(), (10, 20, 310, 420))
        self.assertEqual(box.as_origin_and_size(), (10, 20, 300, 400))
        self.assertEqual(box.as_left_bottom_right_top(), (10, 420, 310, 20))
        self.assertEqual(box.center(), (160.0, 220.0))

    def test_equals(self):
        box1 = BoundingBox(10, 20, 30, 40)
        box2 = BoundingBox(10, 20, 30, 40)
        self.assertEqual(box1, box2)
        box3 = BoundingBox(50, 60, 70, 80)
        self.assertNotEqual(box1, box3)

    def test_str(self):
        self.assertEqual(str(BoundingBox(10, 20, 30, 40)), "(10, 20, 30, 40)")

    def test_invalid_width(self):
        with self.assertRaises(AssertionError) as context:
            BoundingBox(0, 0, -1, 30)

    def test_invalid_height(self):
        with self.assertRaises(AssertionError) as context:
            BoundingBox(0, 0, 10, -20)

    def test_distance_zero(self):
        box1 = BoundingBox(10, 20, 300, 400)
        box2 = BoundingBox(10, 20, 300, 400)
        distance = abs(box2 - box1)
        self.assertEqual(distance, 0)

    def test_distance_x(self):
        box1 = BoundingBox(10, 20, 300, 400)
        box2 = BoundingBox(90, 20, 300, 400)
        distance = box2 - box1
        self.assertEqual(distance, 80)

    def test_distance_y(self):
        box1 = BoundingBox(10, 20, 300, 400)
        box2 = BoundingBox(10, 80, 300, 400)
        distance = box2 - box1
        self.assertEqual(distance, 60)

    def test_distance_xy(self):
        box1 = BoundingBox(10, 20, 300, 400)
        box2 = BoundingBox(60, 70, 300, 400)
        distance = box2 - box1
        self.assertAlmostEqual(distance, 70.7, delta=0.1)

    def test_iou_zero(self):
        box1 = BoundingBox(10, 20, 100, 100)
        box2 = BoundingBox(600, 70, 200, 200)
        self.assertAlmostEqual(box1.iou(box2), 0.0, delta=0.1)

    def test_iou_100(self):
        box1 = BoundingBox(10, 20, 100, 100)
        box2 = BoundingBox(10, 20, 100, 100)
        self.assertAlmostEqual(box1.iou(box2), 1.0, delta=0.1)

    def test_iou_slice_of_x(self):
        box1 = BoundingBox(10, 10, 10, 10)
        box2 = BoundingBox(19, 10, 10, 10)
        self.assertAlmostEqual(box1.iou(box2), 0.05, delta=0.01)

    def test_iou_slice_of_y(self):
        box1 = BoundingBox(10, 10, 20, 20)
        box2 = BoundingBox(10, 19, 20, 20)
        self.assertAlmostEqual(box1.iou(box2), 0.38, delta=0.01)

    def test_intersection_zero(self):
        box1 = BoundingBox(10, 20, 30, 30)
        box2 = BoundingBox(50, 60, 30, 30)
        self.assertAlmostEqual(box1.intersection(box2), 0.0, delta=0.1)

    def test_intersection_100(self):
        box1 = BoundingBox(10, 20, 30, 30)
        box2 = BoundingBox(10, 20, 30, 30)
        self.assertAlmostEqual(box1.intersection(box2), 30*30, delta=0.1)

    def test_intersection_slice_of_x(self):
        box1 = BoundingBox(10, 20, 30, 30)
        box2 = BoundingBox(39, 20, 30, 30)
        self.assertAlmostEqual(box1.intersection(box2), 30, delta=0.01)

    def test_intersection_slice_of_y(self):
        box1 = BoundingBox(10, 20, 30, 30)
        box2 = BoundingBox(10, 48, 30, 30)
        self.assertAlmostEqual(box1.intersection(box2), 2*30, delta=0.01)

    def test_union_disjoint_x_and_y(self):
        box1 = BoundingBox(10, 20, 30, 30)
        box2 = BoundingBox(50, 60, 30, 30)
        self.assertAlmostEqual(box1.union(box2), 4900.0, delta=0.1)

    def test_union_equal(self):
        box1 = BoundingBox(10, 20, 30, 30)
        box2 = BoundingBox(10, 20, 30, 30)
        self.assertAlmostEqual(box1.union(box2), 30*30, delta=0.1)


class DetectedObjectTest(unittest.TestCase):
    def test_default_state(self):
        detected_object = DetectedObject(
            'person',
            0.85,
            BoundingBox(10.6, 20.7, 30.4, 40.3)
        )
        self.assertEqual(detected_object.label, 'person')
        self.assertEqual(detected_object.score, 0.85)

        rounded_box = BoundingBox(11, 21, 30, 40)
        self.assertEqual(
            detected_object.bounding_box,
            rounded_box,
            "{} != {}".format(
                str(detected_object.bounding_box), str(rounded_box))
        )
        self.assertIsNotNone(detected_object.object_id)


class TrackedObjectTest(unittest.TestCase):
    def test_default_state(self):
        tracked_object = TrackedObject(
            "someobjectid",
            BoundingBox(11, 21, 30, 40)
        )
        self.assertEqual(tracked_object.object_id, "someobjectid")
        self.assertEqual(tracked_object.recovery_threshold_in_seconds, 15)
        self.assertEqual(
            tracked_object.first_known_location,
            BoundingBox(11, 21, 30, 40)
        )
        self.assertEqual(
            tracked_object.last_known_location,
            BoundingBox(11, 21, 30, 40)
        )

        self.assertEqual(tracked_object.state, TrackedObjectState.TRACKING)
        self.assertEqual(tracked_object.state_str(), "TRACKING")

        self.assertGreater(tracked_object.age_in_seconds(), 0.0)
        self.assertLess(tracked_object.age_in_seconds(), 1.0)
        self.assertEqual(tracked_object.total_distance_traveled(), 0.0)


class VideoCaptureGeneratorTest(unittest.TestCase):
    def test_read_frame(self):
        generator = VideoCaptureGenerator(os.path.join(
            os.path.dirname(__file__),
            'data',
            'clip.m4v'
        ))
        frame = next(generator)
        self.assertIsNotNone(frame.pixels)
        self.assertEqual(frame.detected_objects, set())

    def test_iteration_ends(self):
        generator = VideoCaptureGenerator(os.path.join(
            os.path.dirname(__file__),
            'data',
            'clip.m4v'
        ))
        frame_count = 0
        for frame in generator:
            frame_count = frame_count + 1
        self.assertEqual(frame_count, 496)

    def test_bogus_file_loop(self):
        generator = VideoCaptureGenerator('bogus_file.m4v')
        frame_count = 0
        for frame in generator:
            frame_count = frame_count + 1
        self.assertEqual(frame_count, 0)
        # frame = next(generator)
        # self.assertIsNone(frame)

    def test_bogus_file_next(self):
        generator = VideoCaptureGenerator('bogus_file.m4v')
        with self.assertRaises(StopIteration) as context:
            next(generator)


if __name__ == '__main__':
    unittest.main()
