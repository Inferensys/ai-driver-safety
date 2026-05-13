export type Severity = "info" | "warning" | "critical";

export type DriverState =
  | "attentive"
  | "eyes_closed"
  | "drowsy"
  | "yawning"
  | "distracted"
  | "phone_use"
  | "face_missing";

export type SessionEvent = {
  event_id: string;
  timestamp: number;
  frame_index: number;
  signal: string;
  state: DriverState;
  score: number;
  severity: Severity;
  message: string;
};

export type RiskPoint = {
  timestamp: number;
  risk_score: number;
};

export type SessionSummary = {
  session_id: string;
  source: string;
  duration_seconds: number;
  processed_frames: number;
  event_counts: Record<string, number>;
  risk_timeline: RiskPoint[];
  longest_unsafe_interval_seconds: number;
  confidence_distribution: Record<string, number>;
  metrics: Record<string, number | string>;
};

export const sampleEvents: SessionEvent[] = [
  {
    event_id: "evt-eyes-001",
    timestamp: 8.28,
    frame_index: 199,
    signal: "eyes_closed",
    state: "eyes_closed",
    score: 1,
    severity: "warning",
    message: "Eyes closed beyond configured threshold",
  },
  {
    event_id: "evt-drowsy-001",
    timestamp: 9.08,
    frame_index: 218,
    signal: "drowsy",
    state: "drowsy",
    score: 1,
    severity: "critical",
    message: "Sustained eye closure indicates drowsiness",
  },
  {
    event_id: "evt-yawn-001",
    timestamp: 18.12,
    frame_index: 435,
    signal: "yawning",
    state: "yawning",
    score: 1,
    severity: "warning",
    message: "Yawn detected from mouth landmarks",
  },
  {
    event_id: "evt-look-001",
    timestamp: 29.46,
    frame_index: 707,
    signal: "distracted",
    state: "distracted",
    score: 0.83,
    severity: "warning",
    message: "Head pose indicates driver is looking away",
  },
  {
    event_id: "evt-phone-001",
    timestamp: 38.16,
    frame_index: 916,
    signal: "phone_use",
    state: "phone_use",
    score: 0.88,
    severity: "critical",
    message: "Phone use detected while driving",
  },
  {
    event_id: "evt-face-001",
    timestamp: 47.16,
    frame_index: 1132,
    signal: "face_missing",
    state: "face_missing",
    score: 0.7,
    severity: "warning",
    message: "Driver face is missing from the frame",
  },
];

export const sampleSummary: SessionSummary = {
  session_id: "demo-driving-20260514T000000Z",
  source: "samples/demo-driving.mp4",
  duration_seconds: 54,
  processed_frames: 1296,
  event_counts: {
    eyes_closed: 72,
    drowsy: 36,
    yawning: 72,
    distracted: 48,
    phone_use: 168,
    face_missing: 48,
  },
  risk_timeline: Array.from({ length: 108 }, (_, index) => {
    const timestamp = index * 0.5;
    const score =
      timestamp > 38 && timestamp < 45
        ? 0.76
        : timestamp > 8 && timestamp < 13
          ? 0.62
          : timestamp > 18 && timestamp < 23
            ? 0.34
            : timestamp > 29 && timestamp < 35
              ? 0.46
              : timestamp > 47 && timestamp < 51
                ? 0.24
                : 0.08;
    return { timestamp, risk_score: score };
  }),
  longest_unsafe_interval_seconds: 7,
  confidence_distribution: {
    eyes_closed: 0.92,
    drowsy: 0.84,
    yawning: 0.88,
    distracted: 0.72,
    phone_use: 0.88,
    face_missing: 0.7,
  },
  metrics: {
    source_fps: 24,
    avg_latency_ms: 3.2,
    p95_latency_ms: 5.8,
    estimated_runtime_fps: 312.5,
    face_provider: "synthetic",
    object_provider: "synthetic",
  },
};

