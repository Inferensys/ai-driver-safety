# Plugin Guide

AI Driver Safety supports detector and alert extension points without changing the core runtime.

## Detector Plugin

Implement an object with:

```python
class DetectorPlugin:
    provider = "my-detector"

    def detect(self, packet):
        return []
```

Face plugins should return `FaceObservation` objects. Object plugins should return `ObjectObservation` objects.

## Alert Policy

Alert policies consume `DetectionEvent` objects and emit alerts. The default `AlertPolicy` implements per-signal cooldowns.

## Frame Sources

Frame sources should yield `FramePacket` objects:

```python
FramePacket(frame=frame, timestamp=seconds, frame_index=index, source_id="rtsp://...")
```

This allows webcam, video, RTSP, and dataset batch sources to share the same pipeline.

