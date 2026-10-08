// The path a picture takes through the eye: one bar for the whole run, and
// one station per stage showing that stage's picture as soon as it is reached.
import { capitalised, stageLabel, stageText } from "../presets";
import { type RunSize, estimatedSeconds, roundDuration, runProgress } from "../progress";
import type { RunState } from "../useRun";
import { StageGraph } from "./StageGraph";

type Props = {
  run: RunState;
  /** The stages this eye is expected to have, shown as placeholders before a run reports its own. */
  expected: string[];
};

export function SignalPath({ run, expected }: Props) {
  const { current, iteration, stageImages, stagePlots, status } = run;
  const stages = run.stages.length ? run.stages : expected;
  if (!stages.length) return null;
  const percent = Math.round(runProgress(run) * 100);
  return (
    <>
      <p className="path-note">
        From the mosaic to the cortex, a stage's output is not a picture. Its tile shows where
        that signal sits in the picture, with the contrast stretched; it is not a reconstruction.
        The firing-rate and spike tiles are graphs of this run's own numbers: each kind of
        cell's rate against its response, with the cells that gave each response underneath,
        and how many cells fired each number of spikes. Bar heights are on a log scale.
      </p>
      <div
        className="path-progress"
        role="progressbar"
        aria-label="Progress through the eye"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        data-state={status}
      >
        <div className="path-progress-fill" style={{ transform: `scaleX(${percent / 100})` }} />
      </div>
      <ol className="path">
        {stages.map((name, index) => {
          const reported = run.stages.length > 0;
          const done = status === "done" || (reported && index < current);
          const active = status === "running" && reported && index === current;
          const image = name === "decoding" ? (done || active ? run.frame : null) : stageImages[name];
          return (
            <li key={name} className="station" data-state={done ? "done" : active ? "active" : "waiting"}>
              <div className="station-thumb">
                {stagePlots[name] ? <StageGraph plot={stagePlots[name]} />
                  : image ? <img src={image} alt="" />
                  : <span className="ghost" aria-hidden="true" />}
              </div>
              <div className="station-name">
                {capitalised(stageLabel(name))}
                {active && name === "decoding" && iteration > 0 && <span className="station-count">step {iteration}</span>}
              </div>
              <p className="station-text">{stageText(name)}</p>
            </li>
          );
        })}
      </ol>
    </>
  );
}

/** One line for screen readers and for the space under the eyepiece. */
export function runStatus(run: RunState, speciesName: string): string {
  if (run.status === "error") return `That run failed: ${run.error}`;
  if (run.status === "running") {
    const name = run.stages[run.current];
    if (!name) return `Building the ${speciesName} eye`;
    if (name === "decoding") return `Rebuilding the picture from spikes, step ${run.iteration}`;
    return `Passing through the ${stageLabel(name)}`;
  }
  if (run.status === "done" && run.result) {
    const seconds = run.result.runtime_s;
    return `Rebuilt in ${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} s over ${run.result.iterations} steps`;
  }
  return "Pick a picture to begin";
}

/** How long the run in progress is expected to take; empty when no run is in progress. */
export function runEstimate(run: RunState, size: RunSize): string {
  if (run.status !== "running") return "";
  return `Estimated run time: ${roundDuration(estimatedSeconds(size))}`;
}
