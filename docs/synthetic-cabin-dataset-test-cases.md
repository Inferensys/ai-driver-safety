# Synthetic Cabin Dataset Test Cases

This document defines 100 video-generation test cases for building an internal synthetic cabin dataset for AI Driver Safety. These clips are for algorithm development and stress testing. They should be labeled as synthetic and kept separate from real human demo proof.

## Generation Rules

- Generate raw cabin footage only. Do not add UI overlays, captions, labels, bounding boxes, watermarks, or dashboard graphics.
- Use an adult driver in every clip.
- Use a fixed in-cabin camera perspective unless the case explicitly changes it.
- Prefer 8-12 second clips at 24-30 FPS, 1280x720 or higher.
- Keep the steering wheel, driver torso, face, hands, seat belt, and windshield context visible whenever possible.
- Include natural vehicle motion, road vibration, lighting changes, and realistic cabin shadows.
- Vary vehicle type, camera placement, road type, weather, time of day, driver appearance, eyewear, clothing, and cabin materials.
- For phone/object cases, make the object visible long enough to support temporal labeling, not a single-frame flash.
- Store prompt, expected labels, duration, generator, seed, and review notes beside every generated clip.

## Suggested Label Fields

```json
{
  "clip_id": "SYN-001",
  "primary_state": "attentive | distracted | phone_use | drowsy | yawning | eyes_closed",
  "secondary_states": ["optional additional states"],
  "objects": ["phone", "cup", "food", "sunglasses", "mask"],
  "lighting": "day | dusk | night | tunnel | glare | low_light",
  "road_context": "city | highway | rural | parking | stop_go | curve | rain | snow",
  "camera_context": "dash_center | rearview_mount | steering_column | wide_cabin | off_axis",
  "expected_interval_seconds": [[2.5, 6.0]],
  "notes": "Synthetic generated clip; review labels manually before training or benchmarking."
}
```

## Prompt Template

Use this template when generating each clip:

```text
Create a realistic raw in-car cabin camera video, 10 seconds, 16:9, 1280x720 or higher, 24 fps. Show an adult driver in the front seat from a dashboard-mounted camera. No text, no subtitles, no UI overlay, no bounding boxes, no watermark. Keep the cabin, steering wheel, driver face, torso, hands, seat belt, and windshield context visible. Scenario: {case prompt}. The footage should look like natural dash/cabin camera video with realistic lighting, shadows, road vibration, and vehicle motion.
```

## 100 Test Cases

| ID | Target Labels | Prompt |
| --- | --- | --- |
| SYN-001 | `attentive` | Adult driver in daylight highway driving, both hands on wheel, eyes forward, calm posture, clear face, stable center dashboard cabin camera, dry road. |
| SYN-002 | `attentive` | Adult driver in city traffic during morning light, checking mirrors briefly but mostly looking ahead, hands on wheel, seat belt visible, normal blinking only. |
| SYN-003 | `attentive` | Adult driver on rural road with trees outside, mild cabin vibration, sunglasses on, eyes still visible through lenses, no distraction objects. |
| SYN-004 | `attentive` | Adult driver at dusk, dashboard camera slightly below eye level, warm cabin shadows, driver focused forward with normal steering corrections. |
| SYN-005 | `attentive` | Adult driver at night on well-lit urban street, face lit by dashboard and streetlights, eyes open, no phone or food, steady posture. |
| SYN-006 | `attentive` | Adult driver in rainy daytime conditions, windshield rain streaks visible, wipers moving, driver alert and looking forward. |
| SYN-007 | `attentive` | Adult driver in snow outside the windshield, bright overcast light, both hands on wheel, no yawning or phone use. |
| SYN-008 | `attentive` | Adult driver in stop-and-go traffic, frequent small braking motion, eyes forward, hands remain on wheel, no distracting activity. |
| SYN-009 | `attentive` | Adult driver on a curving mountain road, head and gaze track the road naturally, both hands on wheel, no unsafe behavior. |
| SYN-010 | `attentive` | Adult driver in a parked car before departure, seat belt fastened, looking forward and checking mirrors, no phone use. |
| SYN-011 | `eyes_closed`, `drowsy` | Adult driver on highway at night, eyelids slowly droop, eyes close for about 1 second, head remains mostly upright, hands still on wheel. |
| SYN-012 | `eyes_closed`, `drowsy` | Adult driver in low-light cabin, repeated long blinks every few seconds, mild head nodding, road lights passing through windshield. |
| SYN-013 | `drowsy` | Adult driver during late evening, head dips forward gradually, eyelids half closed, steering hand loosens, no phone visible. |
| SYN-014 | `drowsy`, `eyes_closed` | Adult driver in boring highway scene, eyes close for 2 seconds, head tilts slightly to the side, then reopens eyes. |
| SYN-015 | `drowsy` | Adult driver after sunset, slow blink pattern and blank forward stare, face visible with dashboard glow, no other distraction. |
| SYN-016 | `drowsy`, `distracted` | Adult driver in tunnel lighting, head drifts down and to the side, eyes partly hidden by shadows, hands on wheel. |
| SYN-017 | `eyes_closed` | Adult driver in daylight, brief microsleep episode with eyes fully closed for about 0.8 seconds, then startled recovery. |
| SYN-018 | `drowsy` | Adult driver wearing clear eyeglasses, eyelids droop behind lenses, repeated slow blinks, highway lane markers visible. |
| SYN-019 | `drowsy` | Adult driver in early morning blue light, heavy eyelids, head slowly nodding twice, no phone or object interaction. |
| SYN-020 | `eyes_closed`, `drowsy` | Adult driver at night with oncoming headlights creating glare, eyes close for a sustained interval, face remains detectable. |
| SYN-021 | `yawning` | Adult driver in daylight cabin, mouth opens wide in a clear yawn for 1.5 seconds, eyes still mostly open, hands on wheel. |
| SYN-022 | `yawning`, `drowsy` | Adult driver at night yawns twice, eyelids droopy, head posture tired, phone absent. |
| SYN-023 | `yawning` | Adult driver on highway in afternoon, one large yawn with visible mouth opening, then returns to forward gaze. |
| SYN-024 | `yawning`, `eyes_closed` | Adult driver yawns while eyes briefly close, face fully visible, dashboard camera centered. |
| SYN-025 | `yawning` | Adult driver wearing glasses yawns, mouth and jaw clearly visible, cabin brightly lit. |
| SYN-026 | `yawning`, `drowsy` | Adult driver in stop-and-go traffic, repeated small yawns and tired facial posture, eyes mostly forward. |
| SYN-027 | `yawning` | Adult driver with one hand on wheel yawns while the other hand rests on gear selector, no phone or food. |
| SYN-028 | `yawning`, `distracted` | Adult driver yawns while glancing down briefly, face partly angled but mouth opening still visible. |
| SYN-029 | `yawning` | Adult driver in rainy evening, yawn occurs while windshield wipers move, face lit by dashboard. |
| SYN-030 | `yawning`, `drowsy` | Adult driver at dawn with soft light, long yawn followed by slow blink, hands remain on wheel. |
| SYN-031 | `phone_use`, `distracted` | Adult driver holds a smartphone low near the center console from second 2 to second 6, repeatedly glancing down at it while driving. |
| SYN-032 | `phone_use`, `distracted` | Adult driver holds a smartphone near the steering wheel with one hand, thumb scrolling, eyes alternate between road and phone. |
| SYN-033 | `phone_use` | Adult driver raises a smartphone near chest level for 3 seconds, screen visible, other hand on steering wheel, daylight cabin. |
| SYN-034 | `phone_use`, `distracted` | Adult driver in night cabin illuminated by phone screen glow, looking down at phone for several seconds. |
| SYN-035 | `phone_use`, `distracted` | Adult driver appears to type on a phone placed near the gear selector, eyes down, steering wheel visible. |
| SYN-036 | `phone_use` | Adult driver uses phone mounted near center dashboard, hand tapping the screen, gaze moves between road and phone mount. |
| SYN-037 | `phone_use`, `distracted` | Adult driver holds phone to ear as if on a call, one hand off wheel, eyes forward but posture distracted. |
| SYN-038 | `phone_use`, `distracted` | Adult driver picks up phone from passenger seat, looks down and right, phone visible in hand for 4 seconds. |
| SYN-039 | `phone_use`, `drowsy` | Adult driver tired at night uses phone low in lap, eyelids droopy, phone screen glow visible. |
| SYN-040 | `phone_use`, `distracted` | Adult driver in bright sunlight checks a phone with glare on the screen, phone partially occluded by hand but visible. |
| SYN-041 | `phone_use`, `distracted` | Adult driver wearing sunglasses holds a phone near steering wheel, head angled down for several seconds. |
| SYN-042 | `phone_use` | Adult driver quickly receives a phone notification, phone visible for 2 seconds, brief glance down, then returns forward. |
| SYN-043 | `phone_use`, `distracted` | Adult driver records a voice message with phone in hand near mouth, gaze shifts away from road. |
| SYN-044 | `phone_use`, `distracted` | Adult driver in city traffic scrolls on phone while stopped, then continues holding it as the car starts moving. |
| SYN-045 | `phone_use`, `distracted` | Adult driver tries to plug in a phone charging cable while driving, phone and cable visible near console. |
| SYN-046 | `distracted` | Adult driver drinks from a water bottle, bottle blocks lower face for 2 seconds, one hand off wheel. |
| SYN-047 | `distracted` | Adult driver eats a snack from a wrapper, glances down, one hand off wheel, food visible. |
| SYN-048 | `distracted` | Adult driver reaches toward glove box, torso leans right, eyes away from road for several seconds. |
| SYN-049 | `distracted` | Adult driver adjusts climate controls on center console, eyes down and right, hand off wheel. |
| SYN-050 | `distracted` | Adult driver changes radio station on dashboard touch screen, gaze repeatedly leaves road. |
| SYN-051 | `distracted` | Adult driver talks animatedly to passenger off camera, head turned right, one hand gesturing. |
| SYN-052 | `distracted` | Adult driver looks at rear seat area over shoulder while car is moving, face profile visible. |
| SYN-053 | `distracted` | Adult driver reaches down to pick up a dropped object near footwell, head disappears partly below wheel. |
| SYN-054 | `distracted` | Adult driver applies lip balm or small cosmetic item while driving, mirror glance, one hand off wheel. |
| SYN-055 | `distracted` | Adult driver adjusts sunglasses with both hands briefly, steering wheel unattended for a moment. |
| SYN-056 | `distracted` | Adult driver reads a paper receipt or parking ticket, paper visible near wheel, gaze down. |
| SYN-057 | `distracted` | Adult driver searches inside jacket pocket, head and eyes down, torso shifts, road still visible. |
| SYN-058 | `distracted` | Adult driver opens a food container on passenger seat, reaches right, eyes away from road. |
| SYN-059 | `distracted` | Adult driver wipes windshield or camera-facing side window with cloth, hand crosses face and camera view. |
| SYN-060 | `distracted` | Adult driver turns around to speak to rear passenger, head rotation sustained for 2 seconds. |
| SYN-061 | `distracted` | Driver face is partially blocked by a raised hand near camera, attention uncertain, cabin otherwise bright. |
| SYN-062 | `distracted` | Driver face is hidden by sun visor shadow and brimmed cap, gaze hard to determine, car moving. |
| SYN-063 | `distracted` | Strong sunlight glare washes out the driver face for several seconds, steering wheel and body still visible. |
| SYN-064 | `distracted` | Camera is slightly misaligned toward passenger side, driver face appears near edge of frame, attention hard to track. |
| SYN-065 | `distracted` | Rearview-mirror-mounted camera has a dangling ornament partially blocking the driver face. |
| SYN-066 | `distracted` | Driver wears a face mask and sunglasses, mouth not visible, eyes hard to read, head turns down briefly. |
| SYN-067 | `distracted` | Driver leans forward close to steering wheel, face partly outside camera frame, road vibration visible. |
| SYN-068 | `distracted` | Night cabin with very low light, only dashboard glow on driver face, eye landmarks difficult. |
| SYN-069 | `distracted` | Rapid transition from bright daylight into a tunnel, driver face exposure changes sharply, gaze briefly untrackable. |
| SYN-070 | `distracted` | Camera lens has smudge or rain-like blur, face visible but landmarks unreliable, driver still moving. |
| SYN-071 | `attentive`, `vehicle_context` | Adult driver brakes hard on city road, body moves forward slightly, eyes remain on road, no distraction. |
| SYN-072 | `attentive`, `vehicle_context` | Adult driver takes a sharp curve, head and torso lean naturally, gaze follows road, hands steady. |
| SYN-073 | `distracted`, `vehicle_context` | Adult driver glances down while vehicle approaches stopped traffic, brake lights visible through windshield. |
| SYN-074 | `phone_use`, `vehicle_context` | Adult driver checks phone while car is in stop-and-go traffic, then traffic starts moving. |
| SYN-075 | `drowsy`, `vehicle_context` | Adult driver shows slow eye closure on long empty highway, lane markings drift slightly in windshield view. |
| SYN-076 | `distracted`, `vehicle_context` | Adult driver looks toward side window while navigating a roundabout, head turned for several seconds. |
| SYN-077 | `attentive`, `vehicle_context` | Adult driver performs parking maneuver, looking between mirrors and road, controlled head turns, no distraction. |
| SYN-078 | `distracted`, `vehicle_context` | Adult driver reacts to sudden honking outside, turns head sharply left, hands remain on wheel. |
| SYN-079 | `attentive`, `vehicle_context` | Adult driver on bumpy road, camera shake and torso movement, eyes remain forward. |
| SYN-080 | `distracted`, `vehicle_context` | Adult driver distracted by navigation screen while approaching highway exit, eyes shift down repeatedly. |
| SYN-081 | `phone_use`, `yawning`, `drowsy` | Adult driver yawns, then checks phone low near console for 3 seconds, night highway lighting. |
| SYN-082 | `phone_use`, `eyes_closed` | Adult driver briefly closes eyes after looking down at phone, phone remains visible in hand. |
| SYN-083 | `distracted`, `yawning` | Adult driver yawns while reaching for a bottle, mouth open and hand off wheel. |
| SYN-084 | `phone_use`, `distracted` | Adult driver on rainy city road uses phone while windshield wipers move, face lit by phone and streetlights. |
| SYN-085 | `drowsy`, `distracted` | Adult driver sleepy and looking toward passenger seat, head drops then turns away from road. |
| SYN-086 | `phone_use`, `drowsy`, `distracted` | Adult driver wearing glasses at night, phone in lap, repeated glances down and slow blinks. |
| SYN-087 | `yawning`, `phone_use` | Adult driver yawns while phone is mounted on dashboard, hand taps screen afterward. |
| SYN-088 | `eyes_closed`, `distracted` | Adult driver closes eyes while head turns toward passenger side, brief unsafe combined interval. |
| SYN-089 | `phone_use`, `distracted` | Adult driver reaches for phone from cup holder, phone visible only after second 4, then held near wheel. |
| SYN-090 | `drowsy`, `yawning`, `eyes_closed` | Adult driver shows full fatigue sequence: long yawn, slow blink, head nod, then recovery. |
| SYN-091 | `attentive` | Adult driver with beard and glasses, daytime city road, normal blinking and mirror checks, no unsafe actions. |
| SYN-092 | `attentive` | Adult driver with long hair partly covering forehead, clear eyes, both hands on wheel, rural road. |
| SYN-093 | `attentive` | Adult driver wearing baseball cap, brim casts shadow, but eyes remain visible and focused forward. |
| SYN-094 | `drowsy` | Adult driver wearing hoodie, low cabin light, eyelids droop, head nods once, no object distraction. |
| SYN-095 | `phone_use` | Adult driver with dark interior cabin and black phone, phone visible against jacket, hand and screen reflections clear. |
| SYN-096 | `distracted` | Adult driver in luxury cabin with large infotainment screen, driver interacts with screen for several seconds. |
| SYN-097 | `attentive` | Adult driver in small economy car, tight cabin, camera close to face, normal focused driving. |
| SYN-098 | `distracted` | Adult driver in truck or van cabin, high seating position, reaches toward dashboard storage area. |
| SYN-099 | `phone_use`, `distracted` | Adult driver in right-hand-drive cabin, phone held in left hand, gaze down, road visible through windshield. |
| SYN-100 | `drowsy`, `phone_use`, `vehicle_context` | Adult driver on dark highway shows fatigue and checks phone during a long straight road segment, sustained unsafe interval. |
