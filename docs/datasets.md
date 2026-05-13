# Datasets

Do not commit dataset media unless redistribution rights are explicit. Keep datasets under `data/`, which is gitignored.

## Drowsiness

### NTHU Driver Drowsiness Detection Dataset

Use for drowsiness, slow blink, yawning, day/night lighting, glasses, and sunglasses validation.

Suggested layout:

```text
data/nthu-ddd/
  train/
  eval/
  test/
```

## Yawning

### YawDD

Use for in-car yawning and mouth-tracking validation. The dataset is useful for yawn detection but should be evaluated carefully because many clips contain non-yawn frames.

Suggested layout:

```text
data/yawdd/
  videos/
  annotations/
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

