// How far a run has got, as one number for the progress bar.

export type RunProgress = {
  status: "idle" | "running" | "done" | "error";
  stages: string[]; // every step in order, ending with "decoding"
  current: number; // index of the step whose output is showing; -1 before the first
  fraction: number; // share of the decoding solve that is done, 0 to 1
};

/**
 * The share of the whole run that is done, from 0 to 1. Each step counts the
 * same; a step counts once its output is showing, and the last step, decoding,
 * counts by how far its solve has got.
 */
export function runProgress({ status, stages, current, fraction }: RunProgress): number {
  if (status === "done") return 1;
  if (status !== "running" || stages.length === 0 || current < 0) return 0;
  const last = stages.length - 1;
  const steps = current >= last ? last + Math.min(Math.max(fraction, 0), 1) : current + 1;
  return Math.min(steps / stages.length, 1);
}

export type RunSize = { species: string; size_px: number; neuron_density: number };

// Seconds a run took on the live site at REFERENCE_PX with the real number of
// neurons, measured once for each species. The human eye has far more cells.
const REFERENCE_SECONDS: Record<string, number> = { human: 40, mouse: 2, fly: 2 };
const REFERENCE_PX = 96;
const UNKNOWN_SPECIES_SECONDS = 10;

/**
 * Roughly how long a run will take, in seconds, worked out from its settings
 * before it starts. The work grows with the number of pixels and of neurons.
 * It is a guide to what to expect, not a measurement.
 */
export function estimatedSeconds({ species, size_px, neuron_density }: RunSize): number {
  const reference = REFERENCE_SECONDS[species] ?? UNKNOWN_SPECIES_SECONDS;
  return reference * (size_px / REFERENCE_PX) ** 2 * neuron_density;
}

/** A run time in round words: nobody needs an estimate to the second. */
export function roundDuration(seconds: number): string {
  if (seconds < 45) return `about ${Math.max(5, Math.round(seconds / 5) * 5)} seconds`;
  if (seconds < 90) return "about a minute";
  return `about ${Math.round(seconds / 60)} minutes`;
}
