import {
  Activity,
  Bell,
  Camera,
  Download,
  Eye,
  FileJson,
  Gauge,
  Play,
  Radio,
  ShieldAlert,
  SlidersHorizontal,
  Upload,
} from "lucide-react";
import { ChangeEvent, useMemo, useState } from "react";
import { sampleEvents, sampleSummary, SessionEvent, SessionSummary } from "./sampleData";

const severityRank = {
  info: 0,
  warning: 1,
  critical: 2,
};

export function App() {
  const [events, setEvents] = useState<SessionEvent[]>(sampleEvents);
  const [summary, setSummary] = useState<SessionSummary>(sampleSummary);
  const [selectedEventId, setSelectedEventId] = useState(sampleEvents[0].event_id);
  const [mode, setMode] = useState<"review" | "live">("review");
  const [showOverlay, setShowOverlay] = useState(true);
  const [alertAudio, setAlertAudio] = useState(false);
  const [riskThreshold, setRiskThreshold] = useState(45);

  const selectedEvent = useMemo(
    () => events.find((event) => event.event_id === selectedEventId) ?? events[0],
    [events, selectedEventId],
  );

  const currentRisk = useMemo(() => {
    if (!selectedEvent) return 0;
    const point = summary.risk_timeline.reduce((closest, point) => {
      return Math.abs(point.timestamp - selectedEvent.timestamp) <
        Math.abs(closest.timestamp - selectedEvent.timestamp)
        ? point
        : closest;
    }, summary.risk_timeline[0]);
    return point?.risk_score ?? 0;
  }, [selectedEvent, summary.risk_timeline]);

  const sortedSignals = useMemo(
    () =>
      Object.entries(summary.confidence_distribution).sort(
        ([, a], [, b]) => Number(b) - Number(a),
      ),
    [summary.confidence_distribution],
  );

  function loadRunJson(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const parsed = JSON.parse(String(reader.result));
      if (Array.isArray(parsed)) {
        setEvents(parsed);
        setSelectedEventId(parsed[0]?.event_id ?? "");
      } else {
        setSummary(parsed);
      }
    };
    reader.readAsText(file);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <ShieldAlert size={24} />
          <div>
            <strong>AI Driver Safety</strong>
            <span>Driver Monitoring Studio</span>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="segmented" aria-label="Mode">
            <button className={mode === "review" ? "active" : ""} onClick={() => setMode("review")}>
              <Play size={15} />
              Review
            </button>
            <button className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}>
              <Radio size={15} />
              Live
            </button>
          </div>
          <label className="icon-button file-loader" title="Load events.json or summary.json">
            <Upload size={18} />
            <input type="file" accept="application/json,.json" onChange={loadRunJson} />
          </label>
          <button className="icon-button" title="Export current report">
            <Download size={18} />
          </button>
        </div>
      </header>

      <main className="workspace">
        <aside className="sidebar panel">
          <div className="section-title">
            <Camera size={16} />
            Session
          </div>
          <div className="session-meta">
            <strong>{summary.session_id}</strong>
            <span>{summary.source}</span>
          </div>
          <div className="metric-grid">
            <Metric label="Duration" value={`${summary.duration_seconds}s`} />
            <Metric label="Frames" value={summary.processed_frames.toLocaleString()} />
            <Metric label="Unsafe" value={`${summary.longest_unsafe_interval_seconds}s`} />
            <Metric label="FPS" value={String(summary.metrics.source_fps)} />
          </div>

          <div className="section-title">
            <SlidersHorizontal size={16} />
            Controls
          </div>
          <label className="toggle-row">
            <span>Overlay</span>
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(event) => setShowOverlay(event.target.checked)}
            />
          </label>
          <label className="toggle-row">
            <span>Alert audio</span>
            <input
              type="checkbox"
              checked={alertAudio}
              onChange={(event) => setAlertAudio(event.target.checked)}
            />
          </label>
          <label className="slider-row">
            <span>Risk threshold</span>
            <strong>{riskThreshold}%</strong>
            <input
              type="range"
              min="20"
              max="80"
              value={riskThreshold}
              onChange={(event) => setRiskThreshold(Number(event.target.value))}
            />
          </label>

          <div className="section-title">
            <FileJson size={16} />
            Artifacts
          </div>
          <div className="artifact-list">
            <span>annotated.mp4</span>
            <span>events.json</span>
            <span>summary.json</span>
            <span>report.html</span>
          </div>
        </aside>

        <section className="video-panel panel">
          <div className="video-toolbar">
            <div>
              <span className={`state-dot ${selectedEvent?.severity ?? "info"}`} />
              <strong>{selectedEvent?.state.replace("_", " ") ?? "attentive"}</strong>
              <span>{selectedEvent ? `${selectedEvent.timestamp.toFixed(2)}s` : "0.00s"}</span>
            </div>
            <div className="runtime-chip">
              <Gauge size={15} />
              Risk {(currentRisk * 100).toFixed(0)}%
            </div>
          </div>
          <div className="video-stage">
            <div className="windshield" />
            <div className="driver-head" />
            <div className="steering-wheel" />
            <div className="road-lines" />
            {showOverlay && selectedEvent ? (
              <div className={`overlay-box ${selectedEvent.severity}`}>
                <span>{selectedEvent.signal.replace("_", " ")}</span>
                <strong>{Math.round(selectedEvent.score * 100)}%</strong>
              </div>
            ) : null}
          </div>
        </section>

        <aside className="inspector panel">
          <div className="section-title">
            <Activity size={16} />
            Current State
          </div>
          <div className={`state-card ${selectedEvent?.severity ?? "info"}`}>
            <span>{selectedEvent?.severity ?? "info"}</span>
            <strong>{selectedEvent?.message ?? "No active event selected"}</strong>
          </div>
          <div className="signal-stack">
            {sortedSignals.map(([signal, value]) => (
              <SignalBar key={signal} label={signal} value={Number(value)} />
            ))}
          </div>

          <div className="section-title">
            <Bell size={16} />
            Alert Stack
          </div>
          <div className="event-list">
            {[...events]
              .sort((a, b) => severityRank[b.severity] - severityRank[a.severity])
              .slice(0, 5)
              .map((event) => (
                <button
                  key={event.event_id}
                  className={event.event_id === selectedEventId ? "event-row active" : "event-row"}
                  onClick={() => setSelectedEventId(event.event_id)}
                >
                  <span className={`state-dot ${event.severity}`} />
                  <strong>{event.signal.replace("_", " ")}</strong>
                  <span>{event.timestamp.toFixed(1)}s</span>
                </button>
              ))}
          </div>
        </aside>

        <section className="timeline panel">
          <div className="timeline-header">
            <div className="section-title">
              <Eye size={16} />
              Event Timeline
            </div>
            <span>{events.length} events</span>
          </div>
          <div className="risk-strip">
            {summary.risk_timeline.map((point) => (
              <button
                key={`${point.timestamp}-${point.risk_score}`}
                className={point.risk_score * 100 >= riskThreshold ? "risk-bar over" : "risk-bar"}
                style={{ height: `${Math.max(8, point.risk_score * 86)}px` }}
                title={`${point.timestamp.toFixed(1)}s risk ${(point.risk_score * 100).toFixed(0)}%`}
              />
            ))}
          </div>
          <div className="event-markers">
            {events.map((event) => (
              <button
                key={event.event_id}
                className={event.event_id === selectedEventId ? "marker active" : "marker"}
                style={{ left: `${(event.timestamp / summary.duration_seconds) * 100}%` }}
                onClick={() => setSelectedEventId(event.event_id)}
                title={event.message}
              />
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SignalBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="signal-row">
      <div>
        <span>{label.replace("_", " ")}</span>
        <strong>{Math.round(value * 100)}%</strong>
      </div>
      <div className="signal-track">
        <span style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}
