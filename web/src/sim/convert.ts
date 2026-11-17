// Playback results arrive from Python as JSON with plain number arrays. Both
// backends (the in-browser worker and the local Python server) turn them into
// Float64Arrays here, so the views never see the difference.

import type { PlaybackResult, SimOptions } from "./protocol";

/** The result plus the buffers that can be transferred between threads. */
export function toPlaybackResult(raw: Record<string, unknown>, opts: SimOptions): [PlaybackResult, ArrayBuffer[]] {
  const transfer: ArrayBuffer[] = [];
  const arr = (values: unknown) => {
    const a = Float64Array.from(values as number[]);
    transfer.push(a.buffer);
    return a;
  };
  const result = {
    ...raw,
    options: opts,
    t: arr(raw.t),
    wind: arr(raw.wind),
    drones: (raw.drones as Record<string, unknown>[]).map((d) => ({
      ...d,
      pos: arr(d.pos),
      quat: arr(d.quat),
      euler: arr(d.euler),
      target: arr(d.target),
      thrust: arr(d.thrust),
    })),
  } as unknown as PlaybackResult;
  return [result, transfer];
}
