# Demo Assets

README demo assets must come from project-approved human cabin recordings only.

Generate real demo assets locally:

```bash
python scripts/make_real_demo_assets.py \
  --video data/approved-demo/driver-yawning.mp4 \
  --config configs/default.yaml \
  --out-run runs/real-human-demo \
  --publish-docs \
  --source-name "Approved real driver/yawning clip" \
  --license-note "Approved for public README demo use"
```

This creates:

- `runs/real-human-demo/annotated.mp4`
- `docs/demo/real-human-demo.mp4`
- `docs/demo/real-human-demo.gif`
- `docs/screenshots/real-human-live-monitor.png`
- `docs/screenshots/real-human-event-timeline.png`
- `docs/sample-output/real-human-summary.json`
- `docs/sample-output/real-human-events.json`
- `docs/sample-output/real-human-demo-source.json`

Additional short GIFs from the same approved demo batch are kept beside the primary demo:

- `docs/demo/real-human-clip-1.gif`
- `docs/demo/real-human-clip-2.gif`
- `docs/demo/real-human-clip-3.gif`
- `docs/demo/real-human-clip-4.gif`
- `docs/demo/real-human-phone-use.gif`
- `docs/sample-output/real-human-clip-batch-summary.json`
- `docs/sample-output/phone-use-events.json`
- `docs/sample-output/phone-use-summary.json`
- `docs/screenshots/real-human-clip-batch.png`
- `docs/screenshots/phone-use-detection.png`

Generate the gallery GIFs from analyzed runs:

```bash
for n in 1 2 3 4; do
  ffmpeg -y \
    -i "runs/user-real-demo-clips/clip_${n}_202605141455/annotated.mp4" \
    -vf "fps=8,scale=480:-1:flags=lanczos" \
    "docs/demo/real-human-clip-${n}.gif"
done
```

Generate the phone-use clip after exporting `models/driver-objects.onnx`:

```bash
ai-driver-safety analyze \
  --video data/approved-demo/driver-phone-use.mp4 \
  --config configs/phone-demo.yaml \
  --out runs/phone-use-demo
```

Do not commit raw source videos. Commit only short derived demo assets when the source terms allow public display and attribution is recorded.
