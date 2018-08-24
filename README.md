# eighttrack
eighttrack is simple package to bootstrap an object detection and tracking pipeline. It offers a few choices of both object detectors (YOLO or Tensorflow) as well as object trackers (OpenCV built-ins).

# example
```
from eighttrack import *
from eighttrack.cv2_detection import CascadeFaceDetector
from eighttrack.cv2_tracking import OpencvObjectTracker

p = Pipeline()
p.add(VideoCaptureGenerator('test/data/clip.m4v'))
p.add(CascadeFaceDetector())
p.add(DetectedObjectDebugger())
p.add(OpencvObjectTracker())
p.add(TrackedObjectDebugger())
p.add(VideoDisplaySink())
p.run()
```
