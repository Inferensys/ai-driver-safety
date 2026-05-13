# Demo Assets

Generate demo assets locally:

```bash
python scripts/make_demo_assets.py
```

This creates:

- `samples/demo-driving.mp4`
- `runs/demo/annotated.mp4`
- `docs/demo/ai-driver-safety-demo.mp4`
- `docs/demo/demo.gif`
- `docs/screenshots/live-monitor.png`
- `docs/screenshots/event-timeline.png`
- `docs/sample-output/summary.json`
- `docs/sample-output/events.json`

The committed synthetic demo is repo-safe and deterministic. Replace it with approved real cabin/driving footage when a license-cleared clip is available.

