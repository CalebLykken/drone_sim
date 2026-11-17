// One time-series chart (uPlot) with a playback cursor, click/drag to seek,
// a synced hover crosshair, a legend, and direct labels at the line ends.

import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { cssVar, type Store, useDarkMode } from "../store";

export interface ChartSeries {
  label: string;
  data: ArrayLike<number>;
  color: string;
  dash?: number[];
  endLabel?: boolean; // label the line's end (off for reference series like targets)
}

const FONT_FAMILY = '"Barlow Semi Condensed", system-ui, sans-serif';

export function TimeChart({
  title,
  unit,
  t,
  series,
  clock,
  onSeek,
  yRange,
  refLine,
  height = 132,
}: {
  title: string;
  unit: string;
  t: Float64Array;
  series: ChartSeries[];
  clock: Store<number>;
  onSeek: (t: number) => void;
  yRange?: [number, number];
  refLine?: { value: number; label: string };
  height?: number;
}) {
  const host = useRef<HTMLDivElement>(null);
  const dark = useDarkMode();
  const seekRef = useRef(onSeek);
  seekRef.current = onSeek;

  useEffect(() => {
    const el = host.current!;
    const ink2 = cssVar("--ink-2");
    const muted = cssVar("--muted");
    const grid = cssVar("--grid");
    const px = uPlot.pxRatio;
    const axis = {
      stroke: muted,
      font: `12px ${FONT_FAMILY}`,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: grid, width: 1, size: 4 },
    };

    const drawExtras = (u: uPlot) => {
      const ctx = u.ctx;
      const { left, top, width, height: h } = u.bbox;
      ctx.save();
      if (refLine) {
        const y = Math.round(u.valToPos(refLine.value, "y", true)) + 0.5;
        if (y >= top && y <= top + h) {
          ctx.strokeStyle = muted;
          ctx.lineWidth = px;
          ctx.setLineDash([4 * px, 4 * px]);
          ctx.beginPath();
          ctx.moveTo(left, y);
          ctx.lineTo(left + width, y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = muted;
          ctx.font = `${11 * px}px ${FONT_FAMILY}`;
          ctx.textBaseline = "bottom";
          ctx.fillText(refLine.label, left + 4 * px, y - 2 * px);
        }
      }
      // Direct labels at the right end of each line, nudged apart so they never overlap.
      const labels = series
        .map((s, i) => ({ s, i, v: s.data[s.data.length - 1] }))
        .filter(({ s, i, v }) => s.endLabel !== false && u.series[i + 1].show && Number.isFinite(v))
        .map(({ s, v }) => ({ s, y: u.valToPos(v, "y", true) }))
        .sort((a, b) => a.y - b.y);
      const gap = 12 * px;
      for (let k = 1; k < labels.length; k++) labels[k].y = Math.max(labels[k].y, labels[k - 1].y + gap);
      const overflow = labels.length ? labels[labels.length - 1].y - (top + h) : 0;
      if (overflow > 0) labels.forEach((l) => (l.y -= overflow));
      ctx.font = `600 ${11 * px}px ${FONT_FAMILY}`;
      ctx.textBaseline = "middle";
      for (const { s, y } of labels) {
        const x = left + width + 4 * px;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2 * px;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 6 * px, y);
        ctx.stroke();
        ctx.fillStyle = ink2;
        ctx.fillText(s.label, x + 9 * px, y);
      }
      ctx.restore();
    };

    const fmt = (v: number | null) => (v == null ? "–" : v.toFixed(2));
    const opts: uPlot.Options = {
      width: Math.max(el.clientWidth, 200),
      height,
      padding: [6, 48, 0, 0], // room on the right for the end labels
      legend: { show: series.length > 1, live: true }, // one series: the title names it
      cursor: {
        sync: { key: "playback" },
        drag: { x: false, y: false, setScale: false },
        y: false,
        points: { size: 6 },
      },
      scales: { x: { time: false }, ...(yRange ? { y: { range: yRange } } : {}) },
      axes: [
        { ...axis, size: 24 },
        { ...axis, size: 40 },
      ],
      series: [
        { label: "Time", value: (_u, v) => (v == null ? "–" : `${v.toFixed(2)} s`) },
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: s.dash ? 1.5 : 2,
          dash: s.dash,
          value: (_u: uPlot, v: number | null) => fmt(v),
        })),
      ],
      hooks: { draw: [drawExtras] },
    };
    const u = new uPlot(opts, [t, ...series.map((s) => s.data)] as unknown as uPlot.AlignedData, el);

    // Playback cursor.
    const line = document.createElement("div");
    line.className = "chart-cursor";
    u.over.appendChild(line);
    const place = () => {
      line.style.transform = `translateX(${Math.round(u.valToPos(clock.get(), "x"))}px)`;
    };
    place();
    const unsubscribe = clock.subscribe(place);

    // Click or drag on the plot to seek.
    let dragging = false;
    const seekTo = (e: PointerEvent) => {
      const rect = u.over.getBoundingClientRect();
      const x = Math.min(Math.max(e.clientX - rect.left, 0), rect.width);
      seekRef.current(u.posToVal(x, "x"));
    };
    const down = (e: PointerEvent) => {
      dragging = true;
      u.over.setPointerCapture(e.pointerId);
      seekTo(e);
    };
    const move = (e: PointerEvent) => dragging && seekTo(e);
    const up = () => void (dragging = false);
    u.over.addEventListener("pointerdown", down);
    u.over.addEventListener("pointermove", move);
    u.over.addEventListener("pointerup", up);
    u.over.addEventListener("pointercancel", up);

    const resize = new ResizeObserver(() => {
      u.setSize({ width: Math.max(el.clientWidth, 200), height });
      place();
    });
    resize.observe(el);

    return () => {
      unsubscribe();
      resize.disconnect();
      u.destroy();
    };
  }, [t, series, height, dark, yRange, refLine, clock]);

  return (
    <figure className="chart">
      <figcaption>
        {title} <span className="chart-unit">{unit}</span>
      </figcaption>
      <div ref={host} className="chart-plot" />
    </figure>
  );
}
