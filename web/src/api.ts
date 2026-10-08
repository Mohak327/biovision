// Talks to the Python server. No mathematics lives here.

export type Settings = {
  species: string;
  size_px: number;
  fov_deg: number;
  window_ms: number;
  noise: boolean;
  lam: number | null;
  seed: number;
  density: number;
  neuron_density: number;
};

export type ImageSource = { kind: "sample"; name: string } | { kind: "file"; file: File };

export type SpeciesInfo = {
  name: string;
  description: string;
  receptors: string[];
  has_cortex: boolean;
  citations: string[];
  stages: string[]; // the steps of a run, ending with "decoding"
};

export type MosaicEvent = {
  type: "mosaic";
  count: number;
  positions: string;
  types: string;
  receptors: string[];
};
export type OriginalEvent = { type: "original"; image: string };
// The numbers behind the two stages that are better read as a graph than a picture.
export type RatePlot = { response: number[]; rates: Record<string, number[]>; cells: number[] };
export type SpikePlot = { edges: number[]; cells: number[] };
export type StagePlot = RatePlot | SpikePlot;
export type StageEvent = {
  type: "stage"; stages: string[]; current: number; image: string; plot: StagePlot | null;
};
// `progress` is the share of the decoding solve that is done, 0 to 1.
export type FrameEvent = { type: "frame"; iteration: number; image: string; progress: number };
export type ErrorEvent = { type: "error"; message: string };
export type RunSummary = {
  type: "result";
  metrics: {
    psnr_db: number; ssim: number; neurons: number; receptors: number;
    compression_ratio: number; mean_spikes: number;
  };
  settings: Settings & { chroma_weight: number };
  runtime_s: number;
  iterations: number;
  converged: boolean;
  residuals: number[];
  original: string;
  reconstructed: string;
  responses: string;
  channel_rmse: number[];
  spectrum: { cpd: number[]; original: number[]; reconstructed: number[]; limit_cpd: number };
  spike_histogram: { edges: number[]; counts: number[] };
  color_matrix: number[][];
  receptors: string[];
  parameters: { species: string; parameter: string; value: string; unit: string }[];
  description: string;
  citations: string[];
};
export type RunEvent = OriginalEvent | MosaicEvent | StageEvent | FrameEvent | ErrorEvent | RunSummary;
export type SweepKind = "window" | "lambda" | "density";
export type SweepRow = Record<string, number | string>;

/** Splits a stream of text into JSON values, one per line. */
export class NdjsonParser {
  private rest = "";

  push(chunk: string): unknown[] {
    const lines = (this.rest + chunk).split("\n");
    this.rest = lines.pop() ?? "";
    return lines.filter((line) => line.trim()).map((line) => JSON.parse(line));
  }

  flush(): unknown[] {
    const line = this.rest.trim();
    this.rest = "";
    return line ? [JSON.parse(line)] : [];
  }
}

function bytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

export const decodeUint8 = (base64: string): Uint8Array => bytes(base64);
export const decodeFloat32 = (base64: string): Float32Array => new Float32Array(bytes(base64).buffer);
export const pngUrl = (base64: string): string => `data:image/png;base64,${base64}`;

function form(settings: Settings, source: ImageSource, extra: Record<string, string> = {}): FormData {
  const data = new FormData();
  data.set("settings", JSON.stringify(settings));
  if (source.kind === "sample") data.set("sample", source.name);
  else data.set("image", source.file);
  for (const [key, value] of Object.entries(extra)) data.set(key, value);
  return data;
}

/** The server's error text, whether it sent a string or a list of field problems. */
async function failure(response: Response): Promise<Error> {
  let message = `The server answered ${response.status}.`;
  try {
    const { detail } = await response.json();
    if (typeof detail === "string") message = detail;
    else if (Array.isArray(detail)) {
      message = detail.map((item) => `${(item.loc ?? []).join(".")}: ${item.msg}`).join("; ");
    }
  } catch {
    // keep the status message
  }
  return new Error(message);
}

async function* stream<T>(path: string, body: FormData, signal?: AbortSignal): AsyncGenerator<T> {
  const response = await fetch(path, { method: "POST", body, signal });
  if (!response.ok || !response.body) throw await failure(response);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new NdjsonParser();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const event of parser.push(decoder.decode(value, { stream: true }))) yield event as T;
  }
  for (const event of parser.flush()) yield event as T;
}

export const streamRun = (settings: Settings, source: ImageSource, signal: AbortSignal) =>
  stream<RunEvent>("/api/runs", form(settings, source), signal);

export const streamCompare = (settings: Settings, source: ImageSource, signal?: AbortSignal) =>
  stream<RunSummary>("/api/compare", form(settings, source), signal);

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw await failure(response);
  return response.json();
}

export const fetchSpecies = () => getJson<SpeciesInfo[]>("/api/species");
export const fetchSamples = () => getJson<string[]>("/api/samples");
export const sampleUrl = (name: string) => `/api/samples/${name}`;

export async function fetchSweep(kind: SweepKind, settings: Settings, source: ImageSource): Promise<SweepRow[]> {
  const response = await fetch("/api/sweeps", { method: "POST", body: form(settings, source, { kind }) });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function fetchReport(settings: Settings, source: ImageSource, full: boolean): Promise<Blob> {
  const extra = { all_species: String(full), sweeps: String(full) };
  const response = await fetch("/api/report", { method: "POST", body: form(settings, source, extra) });
  if (!response.ok) throw await failure(response);
  return response.blob();
}
