// Drives one run at a time: starts it, follows its events, cancels the previous one.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ImageSource, type MosaicEvent, type RunSummary, type Settings, pngUrl, streamRun,
} from "./api";

export type RunState = {
  status: "idle" | "running" | "done" | "error";
  stages: string[];
  current: number;
  iteration: number;
  original: string | null; // the picture at the size the eye sees it
  frame: string | null; // the reconstruction so far, as an image URL
  stageImages: Record<string, string>;
  mosaic: MosaicEvent | null;
  result: RunSummary | null;
  error: string | null;
};

const IDLE: RunState = {
  status: "idle", stages: [], current: -1, iteration: 0, original: null, frame: null,
  stageImages: {}, mosaic: null, result: null, error: null,
};

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
    try {
      for await (const event of streamRun(runSettings, runSource, abort.signal)) {
        if (abort.signal.aborted) return;
        if (event.type === "original") {
          setState((s) => ({ ...s, original: pngUrl(event.image) }));
        } else if (event.type === "mosaic") {
          setState((s) => ({ ...s, mosaic: event }));
        } else if (event.type === "stage") {
          const name = event.stages[event.current];
          setState((s) => ({
            ...s, stages: event.stages, current: event.current,
            stageImages: event.image ? { ...s.stageImages, [name]: pngUrl(event.image) } : s.stageImages,
          }));
        } else if (event.type === "frame") {
          setState((s) => ({
            ...s, current: s.stages.length - 1, iteration: event.iteration, frame: pngUrl(event.image),
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
