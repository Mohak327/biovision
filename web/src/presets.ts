// Named choices in place of raw numbers, and the words the page uses.
import type { Settings } from "./api";

export type Preset = { value: number; label: string; hint: string };

export const WINDOW_PRESETS: Preset[] = [
  { value: 30, label: "glance", hint: "30 ms of spikes" },
  { value: 100, label: "look", hint: "100 ms of spikes" },
  { value: 1000, label: "stare", hint: "1 s of spikes" },
];

export const DENSITY_PRESETS: Preset[] = [
  { value: 0.25, label: "quarter", hint: "a quarter of the receptors" },
  { value: 1, label: "real", hint: "the real animal" },
  { value: 4, label: "4×", hint: "four times the receptors" },
  { value: 16, label: "16×", hint: "sixteen times the receptors" },
];

export const NEURON_PRESETS: Preset[] = [
  { value: 0.25, label: "quarter", hint: "a quarter of the cortex cells" },
  { value: 1, label: "real", hint: "the real animal" },
  { value: 4, label: "4×", hint: "four times the cortex cells" },
];

export const FOV_PRESETS: Preset[] = [
  { value: 10, label: "phone", hint: "a phone at arm's length, 10°" },
  { value: 30, label: "monitor", hint: "a desk monitor, 30°" },
  { value: 60, label: "cinema", hint: "a cinema screen, 60°" },
  { value: 120, label: "surround", hint: "most of your view, 120°" },
];

export const SIZE_PRESETS: Preset[] = [
  { value: 64, label: "draft", hint: "64 pixels, fastest" },
  { value: 96, label: "standard", hint: "96 pixels" },
  { value: 128, label: "fine", hint: "128 pixels, slowest" },
];

export const DEFAULT_SETTINGS: Settings = {
  species: "human",
  size_px: 96,
  fov_deg: 60,
  window_ms: 100,
  noise: true,
  lam: null,
  seed: 0,
  density: 1,
  neuron_density: 1,
};

export function presetLabel(presets: Preset[], value: number): string {
  return presets.find((preset) => preset.value === value)?.label ?? String(value);
}

// Colour always means a receptor type, here and in every chart.
const VIOLET = "#5b4bd6";
const GREEN = "#2e9e6b";
const AMBER = "#e0a021";
const MAGENTA = "#c13fa0";
const BLUE = "#3f7be0";

export const RECEPTOR_COLORS: Record<string, string> = {
  L: AMBER, M: GREEN, S: VIOLET, UV: MAGENTA, blue: BLUE, green: GREEN,
};

export const RECEPTOR_NAMES: Record<string, string> = {
  L: "long-wave cones", M: "middle-wave cones", S: "short-wave cones",
  UV: "ultraviolet receptors", blue: "blue receptors", green: "green receptors",
};

const STAGE_LABELS: Record<string, string> = {
  color: "receptor colours",
  optics: "optics",
  mosaic: "mosaic",
  center_surround: "retina",
  gabor: "cortex",
  rate: "firing rate",
  spikes: "spikes",
  decoding: "decoding",
};

const STAGE_TEXT: Record<string, string> = {
  color: "Light is split by what each receptor type can absorb.",
  optics: "The lens and cornea blur the picture slightly before it lands.",
  mosaic: "Receptors sample the picture only where they sit.",
  center_surround: "Each retinal cell compares its spot with the neighbourhood around it.",
  gabor: "Cortex cells respond to edges at particular angles and sizes.",
  rate: "Each response becomes a firing rate around the cell's resting rate.",
  spikes: "The cell fires a countable number of spikes in the time allowed.",
  decoding: "The picture is rebuilt from the spike counts alone.",
};

export const stageLabel = (name: string): string => STAGE_LABELS[name] ?? name.replaceAll("_", " ");
export const stageText = (name: string): string => STAGE_TEXT[name] ?? "";
