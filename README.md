# eighttrack
eighttrack is simple package to bootstrap an object detection and tracking pipeline. It offers a few choices of both object detectors (YOLO or Tensorflow) as well as object trackers (OpenCV built-ins).

# example
```
import eighttrack as et

p = et.Pipeline()
p.add(et.VideoCaptureGenerator('test/data/clip.m4v'))
p.add(et.CascadeFaceDetector())
p.add(et.DetectedObjectDebugger())
p.add(et.VideoDisplaySink())
p.run()
```
