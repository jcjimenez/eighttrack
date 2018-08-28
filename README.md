# eighttrack
eighttrack is simple package to bootstrap an object detection and tracking pipeline. It offers a few choices of both object detectors (YOLO or Tensorflow) as well as object trackers (OpenCV built-ins).

# example
You can a simple detection+tracking pipeline like so (you may need to adjust
the `haar_path` value):

```
from eighttrack import *
from eighttrack.cv2_detection import CascadeDetector
from eighttrack.cv2_tracking import OpencvObjectTracker

p = Pipeline()
p.add(VideoCaptureGenerator('test/data/clip.m4v'))
p.add(CascadeDetector(haar_path='/usr/local/Cellar/opencv/3.4.1_6/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'))
p.add(DetectedObjectDebugger())
p.add(OpencvObjectTracker())
p.add(TrackedObjectDebugger())
p.add(VideoDisplaySink())
p.run()
```
