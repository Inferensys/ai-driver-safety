# Datasets

Do not commit dataset media unless redistribution rights are explicit. Keep datasets under `data/`, which is gitignored.

## Active Validation Tracks

The revamp uses three real dataset tracks:

| Track | Dataset | What It Proves | Local Command |
| --- | --- | --- | --- |
| Human yawning video | YawDD | Mouth/yawn scoring on real in-car human video | `ai-driver-safety datasets prepare-yawdd` |
| Physiological drowsiness sensors | DD-Database | Drowsiness events from EEG, EOG, ECG, and annotations | `ai-driver-safety datasets prepare-dd` and `ai-driver-safety datasets dd-events` |
| Car sensor telemetry | UAH-DriveSet | Vehicle-risk events from accelerometer, GPS, lane, and vehicle telemetry | `ai-driver-safety datasets prepare-uah` and `ai-driver-safety datasets uah-events` |

## Human Yawning Video

### YawDD

Source: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset

Use for in-car yawning and mouth-tracking validation. YawDD contains real drivers in a car with different facial characteristics and illumination conditions. It has one mirror-camera set with separate normal, talking/singing, and yawning clips, plus a dash-camera set with mixed silent/talking/yawning behavior in each subject video.

Suggested layout:

```text
data/yawdd/
  ParticipantsInformation.csv
  mirror-camera/
    23-FemaleNoGlasses-Talking&Yawning.avi
  dash-camera/
    ...
```

Prepare the manifest:

```bash
ai-driver-safety datasets prepare-yawdd \
  --input data/yawdd \
  --participants-info data/yawdd/ParticipantsInformation.csv \
  --out data/manifests/yawdd.json
```

Run a local video analysis:

```bash
ai-driver-safety analyze \
  --video "data/yawdd/mirror-camera/23-FemaleNoGlasses-Talking&Yawning.avi" \
  --config configs/default.yaml \
  --out runs/yawdd-23
```

Media policy:

- Use all videos locally for research and validation.
- Public screenshots are allowed only for clips whose `Participants Information` row explicitly allows public sharing.
- Do not commit full YawDD videos or annotated full-video derivatives to the repo.

## Physiological Drowsiness Sensors

### Drivers Drowsiness Database (DD-Database)

Source: https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c

Use for drowsiness detection via physiological sensors. DD-Database contains EDF signal files with 4 EEG channels, 2 EOG channels, 1 ECG channel, and annotation files for 10 volunteers during a driving-simulator protocol designed to induce drowsiness.

Suggested layout:

```text
data/dd-database/
  dryad-files.json
  raw/
    01M_1.edf
    01M_1_annotations.edf
    01M_2.edf
    01M_2_annotations.edf
```

Fetch the Dryad file index and prepare the local manifest:

```bash
ai-driver-safety datasets dd-index --out data/dd-database/dryad-files.json
ai-driver-safety datasets prepare-dd \
  --input data/dd-database/raw \
  --download-index data/dd-database/dryad-files.json \
  --out data/manifests/dd-database.json
```

`docs/sample-output/dd-database-dryad-files.json` is committed as a metadata-only example of the real Dryad file index.

Export drowsiness events from annotation EDF files:

```bash
python -m pip install -e ".[datasets]"
ai-driver-safety datasets dd-events \
  --input data/dd-database/raw \
  --out runs/dd-database-sensors
```

Implementation notes:

- `dd-index` uses the Dryad API to record the official file list without committing raw data.
- `prepare-dd` pairs signal EDF files with matching `_annotations.edf` files.
- `dd-events` uses `pyedflib` when installed. Without the `datasets` extra, manifests still work but EDF annotation parsing is skipped.
- The generated signal is `sensor_drowsiness`, which feeds the same risk-scoring system as visual drowsiness.

## Car Sensor Telemetry

### UAH-DriveSet

Sources:

- https://github.com/Eromera/uah_driveset_reader
- https://www.selectdataset.com/dataset/4ce5b154f659b44fd9885ddcab30763a

Use for real driving telemetry. UAH-DriveSet contains more than 500 minutes of naturalistic driving by 6 driver-vehicle pairs across normal, drowsy, and aggressive behavior on motorway and secondary roads. It includes smartphone accelerometer/GPS, processed lane detection, processed vehicle detection, OpenStreetMap data, semantic information, and trip video.

Suggested layout:

```text
data/uah-driveset/
  D1/
    normal/
      motorway/
        trip-001/
          RAW_ACCELEROMETERS.txt
          RAW_GPS.txt
          PROC_LANE_DETECTION.txt
          PROC_VEHICLE_DETECTION.txt
          PROC_OPENSTREETMAP_DATA.txt
```

Prepare the manifest and export vehicle events:

```bash
ai-driver-safety datasets prepare-uah \
  --input data/uah-driveset \
  --out data/manifests/uah-driveset.json

ai-driver-safety datasets uah-events \
  --input data/uah-driveset \
  --out runs/uah-driveset-sensors
```

Generated vehicle-risk signals:

- `lane_drift`: lane-center offset exceeds the configured safety envelope.
- `short_time_to_collision`: processed lead-vehicle time-to-collision is under 2 seconds.
- `hard_maneuver`: accelerometer values show hard acceleration, braking, or lateral movement.
- `speeding`: GPS speed exceeds the nearest OpenStreetMap speed-limit signal.

License policy:

- The official reader tool states Creative Commons Attribution-NonCommercial 4.0.
- Keep raw telemetry and trip video under `data/`.
- Cite the UAH-DriveSet papers listed in the official reader when publishing results.

## Additional Drowsiness Video

### NTHU Driver Drowsiness Detection Dataset

Use for drowsiness, slow blink, yawning, day/night lighting, glasses, and sunglasses validation.

Suggested layout:

```text
data/nthu-ddd/
  train/
  eval/
  test/
```

## Distraction

### State Farm Distracted Driver Detection

Use for phone use, radio operation, drinking, reaching, passenger talk, and related driver-distraction classes.

Suggested label mapping:

```text
c0 safe driving
c1 texting right
c2 phone right
c3 texting left
c4 phone left
c5 operating radio
c6 drinking
c7 reaching behind
c8 hair and makeup
c9 talking to passenger
```

## Fine-Grained Behavior

### Drive&Act

Use later for serious multi-view, multi-modal driver behavior benchmarking. This is broader than the first release and should be treated as an evaluation benchmark rather than a quick demo dependency.
