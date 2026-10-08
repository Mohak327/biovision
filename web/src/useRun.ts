// Drives one run at a time: starts it, follows its events, cancels the previous one.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ImageSource, type MosaicEvent, type RunSummary, type Settings, type StagePlot, pngUrl,
  streamRun,
} from "./api";

export type RunState = {
  status: "idle" | "running" | "done" | "error";
  stages: string[];
  current: number;
  iteration: number;
  fraction: number; // share of the decoding solve that is done
  original: string | null; // the picture at the size the eye sees it
  frame: string | null; // what the eye holds so far: a stage's picture, then the reconstruction
  stageImages: Record<string, string>;
  stagePlots: Record<string, StagePlot>;
  mosaic: MosaicEvent | null;
  result: RunSummary | null;
  error: string | null;
};

const IDLE: RunState = {
  status: "idle", stages: [], current: -1, iteration: 0, fraction: 0, original: null, frame: null,
  stageImages: {}, stagePlots: {}, mosaic: null, result: null, error: null,
};

const STAGE_DWELL_MS = 320; // long enough to see each stage's picture before the next

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));
const prefersStill = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function useRun(settings: Settings, source: ImageSource | null, delayMs = 250): RunState {
  const [state, setState] = useState<RunState>(IDLE);
  const controller = useRef<AbortController | null>(null);
  const lastSource = useRef<ImageSource | null>(null);

  const start = useCallback(async (runSettings: Settings, runSource: ImageSource) => {
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    // Keep the last picture on screen until this run sends its own, unless the
    // picture itself changed.
    const samePicture = lastSource.current === runSource;
    lastSource.current = runSource;
    setState((previous) => ({
      ...IDLE, status: "running", result: previous.result,
      original: samePicture ? previous.original : null,
    }));
    // The stages of the eye are computed in an instant. Each one's picture is
    // held on screen for a moment, so the run can be followed step by step.
    let lastStage = 0;
    const settle = async () => {
      const wait = prefersStill() ? 0 : lastStage + STAGE_DWELL_MS - performance.now();
      if (wait > 0) await sleep(wait);
    };
    try {
      for await (const event of streamRun(runSettings, runSource, abort.signal)) {
        if (event.type === "stage" || event.type === "frame" || event.type === "result") await settle();
        if (abort.signal.aborted) return;
        if (event.type === "original") {
          setState((s) => ({ ...s, original: pngUrl(event.image) }));
        } else if (event.type === "mosaic") {
          setState((s) => ({ ...s, mosaic: event }));
        } else if (event.type === "stage") {
          lastStage = performance.now();
          const name = event.stages[event.current];
          const picture = pngUrl(event.image);
          setState((s) => ({
            ...s, stages: event.stages, current: event.current, frame: picture,
            stageImages: { ...s.stageImages, [name]: picture },
            stagePlots: event.plot ? { ...s.stagePlots, [name]: event.plot } : s.stagePlots,
          }));
        } else if (event.type === "frame") {
          setState((s) => ({
            ...s, current: s.stages.length - 1, iteration: event.iteration, fraction: event.progress,
            frame: pngUrl(event.image),
          }));
        } else if (event.type === "result") {
          setState((s) => ({
            ...s, status: "done", result: event, frame: pngUrl(event.reconstructed),
            original: pngUrl(event.original), // also covers a server that sent no "original" event
            current: s.stages.length, iteration: event.iterations,
          }));
        } else {
          setState((s) => ({ ...s, status: "error", error: event.message }));
        }
      }
    } catch (error) {
      if (abort.signal.aborted) return;
      const message = error instanceof Error ? error.message : String(error);
      setState((s) => ({ ...s, status: "error", error: message }));
    }
  }, []);

  // Start a run shortly after the last change, so a burst of clicks runs once.
  const key = JSON.stringify(settings);
  useEffect(() => {
    if (!source) return undefined;
    const timer = window.setTimeout(() => void start(JSON.parse(key) as Settings, source), delayMs);
    return () => window.clearTimeout(timer);
  }, [key, source, delayMs, start]);

  useEffect(() => () => controller.current?.abort(), []);
  return state;
}
