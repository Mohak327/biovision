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
