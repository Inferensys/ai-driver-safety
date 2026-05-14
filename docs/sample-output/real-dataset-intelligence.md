# AI Driver Safety Real Dataset Intelligence

The project should be evaluated as a fused driver-monitoring system: camera signals prove cabin behavior, physiological signals prove fatigue, and vehicle telemetry proves driving style.

## Dataset Signal Coverage

| Dataset | Validates | Data shape | Main gap |
| --- | --- | --- | --- |
| NITYMED: Night-Time Yawning-Microsleep-Eyeblink-driver Distraction | face_present, mouth_aspect_ratio, yawning, eyes_closed, microsleep_review_window | videos: 130, yawning_videos: 107, microsleep_videos: 21, format: MP4, 25 fps, 720p/1080p | Access requires submitting name, affiliation, and business or academic email. |
| YawDD: Yawning Detection Dataset | face_present, mouth_aspect_ratio, yawning, talking_or_singing_context, camera_view_robustness | videos: 351, mirror_camera_videos: 322, dash_camera_videos: 29, format: 640x480 RGB AVI, 30 fps | Not a full distraction or object-recognition dataset. |
| Drivers Drowsiness Database (DD-Database) | sensor_drowsiness, eye_movement_proxy_eog, heart_rate_proxy_ecg, fatigue_annotation_event, multimodal_risk_score | subjects: 10, trials: 20, duration_hours: 40, channels: 4 EEG, 2 EOG, 1 ECG, dryad_files: 41, signal_files: 20, annotation_files: 20, indexed_mb: 260.04 | No cabin video/object context. |
| UAH-DriveSet | lane_drift, short_time_to_collision, hard_maneuver, speeding, driving_style_fuzzy_score | duration_minutes: 500, driver_vehicle_pairs: 6, behaviors: normal, drowsy, aggressive, road_types: motorway, secondary | License is non-commercial and prohibits redistribution of raw or modified dataset data. |

## Project Analysis

### NITYMED: Night-Time Yawning-Microsleep-Eyeblink-driver Distraction

- Use NITYMED as the public-facing real human demo source once dataset access is approved.
- The yawning subset is short enough for README clips and should replace generated cabin visuals.
- The microsleep subset is better for event-review timelines because clips are longer and include talking, looking around, and microsleep windows.
- Because the dataset page allows distinct frames in research publications, README images should cite the dataset and avoid committing the raw full dataset.

Demo policy: Preferred source for `docs/demo/real-human-demo.*` after access approval; commit only short derived assets with attribution.

### YawDD: Yawning Detection Dataset

- Use YawDD as the first real-human visual proof because it aligns directly with yawn and mouth-state detection.
- The dash-camera set is stronger for product demos because each subject video contains silent, talking, and yawning behavior in one clip.
- The mirror-camera set is stronger for detector stress testing because the behavior classes are separated across many short clips.
- Frame-level labels are not guaranteed in the base release, so project scoring should report event timelines and manual-review windows rather than overclaiming supervised accuracy.

Demo policy: Run local video analysis and publish only screenshots explicitly allowed by the Participants Information file.

### Drivers Drowsiness Database (DD-Database)

- Use DD-Database as the physiological validation track for the original steering-wheel/heartbeat idea.
- EOG and ECG channels make it useful for comparing visual eye-closure/yawn events against sensor evidence.
- The annotation EDF files can be converted into `sensor_drowsiness` events and fused into the same risk scorer as camera events.
- The data comes from a driving simulator, so it is strong for drowsiness physiology but should not be treated as road-driving telemetry.

Demo policy: Commit metadata-only analysis and generated event summaries; keep raw EDF under `data/`.

### UAH-DriveSet

- Use UAH-DriveSet as the main proof for the fuzzy driving-style classifier.
- The dataset's normal/drowsy/aggressive behavior folders map directly to the repo's risk and fuzzy-logic framing.
- Road type and speed-limit context are necessary for thresholds because harsh acceleration on a city road and highway should not share one naive threshold.
- Vehicle events should remain separate from cabin events but fuse at the SessionSummary risk layer.

Demo policy: Publish abstract metrics/charts and local commands; keep raw telemetry and trip video under `data/`.

## Fused Solution

- README proof path: NITYMED replaces generated visuals with real in-car yawning and microsleep footage.
- Vision benchmark path: YawDD validates real human yawn/mouth signals and the annotated-video/report workflow.
- Sensor path: DD-Database validates physiological drowsiness events for the heartbeat/sensor extension.
- Vehicle path: UAH-DriveSet validates fuzzy driving-style scoring from real car telemetry.
- Fusion path: driver-risk-fusion-v1 combines camera, physiological, and vehicle-risk events into one timeline.
