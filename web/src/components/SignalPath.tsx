// The path a picture takes through the eye: one station per stage.
import type { RunState } from "../useRun";
import { stageLabel, stageText } from "../presets";
import { StageThumb } from "./StageThumb";

type Props = { run: RunState };

export function SignalPath({ run }: Props) {
  const { stages, current, iteration, stageImages, status } = run;
  if (!stages.length) return null;
  return (
    <ol className="path">
      {stages.map((name, index) => {
        const done = status === "done" || index < current;
        const active = status === "running" && index === current;
        const image = name === "decoding" ? run.frame : stageImages[name];
        return (
          <li key={name} className="station" data-state={done ? "done" : active ? "active" : "waiting"}>
            <div className="station-thumb">
              <StageThumb
                name={name}
                image={image}
                mosaic={run.mosaic}
                result={status === "done" ? run.result : null}
                reached={done || active}
              />
            </div>
            <div className="station-name">
              {stageLabel(name)}
              {active && name === "decoding" && iteration > 0 && <span className="station-count"> step {iteration}</span>}
            </div>
            <p className="station-text">{stageText(name)}</p>
          </li>
        );
      })}
    </ol>
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
