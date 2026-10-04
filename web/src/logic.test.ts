import { describe, expect, it } from "vitest";
import { NdjsonParser, decodeFloat32, decodeUint8 } from "./api";
import { linePath, niceTicks } from "./chart";
import {
  DEFAULT_SETTINGS, DENSITY_PRESETS, RECEPTOR_COLORS, SIZE_PRESETS, WINDOW_PRESETS, presetLabel,
  stageLabel,
} from "./presets";

describe("NdjsonParser", () => {
  it("returns whole lines and keeps a partial line for the next chunk", () => {
    const parser = new NdjsonParser();
    expect(parser.push('{"a":1}\n{"b"')).toEqual([{ a: 1 }]);
    expect(parser.push(':2}\n')).toEqual([{ b: 2 }]);
  });

  it("skips blank lines and flushes a final line with no newline", () => {
    const parser = new NdjsonParser();
    expect(parser.push('\n{"a":1}\n\n{"c":3}')).toEqual([{ a: 1 }]);
    expect(parser.flush()).toEqual([{ c: 3 }]);
    expect(parser.flush()).toEqual([]);
  });
});

describe("binary decoding", () => {
  it("decodes little-endian float32 and uint8 arrays from base64", () => {
    const floats = new Float32Array([0, 0.5, 1]);
    const text = btoa(String.fromCharCode(...new Uint8Array(floats.buffer)));
    expect(Array.from(decodeFloat32(text))).toEqual([0, 0.5, 1]);
    expect(Array.from(decodeUint8(btoa(String.fromCharCode(0, 1, 2))))).toEqual([0, 1, 2]);
  });
});

describe("presets", () => {
  it("defaults to the real animal with real neurons", () => {
    expect(DEFAULT_SETTINGS.density).toBe(1);
    expect(DEFAULT_SETTINGS.neuron_density).toBe(1);
    expect(DEFAULT_SETTINGS.noise).toBe(true);
    expect(DEFAULT_SETTINGS.window_ms).toBe(100);
  });

  it("offers picture detail as explicit pixel sizes", () => {
    expect(SIZE_PRESETS.map((preset) => preset.value)).toEqual([96, 128, 256, 512]);
    expect(SIZE_PRESETS.map((preset) => preset.label)).toEqual(["96 px", "128 px", "256 px", "512 px"]);
    expect(SIZE_PRESETS.some((preset) => preset.value === DEFAULT_SETTINGS.size_px)).toBe(true);
  });

  it("names every preset value, and falls back to the number", () => {
    expect(presetLabel(WINDOW_PRESETS, 100)).toBe("look");
    expect(presetLabel(DENSITY_PRESETS, 1)).toBe("real");
    expect(presetLabel(DENSITY_PRESETS, 3)).toBe("3");
  });

  it("gives each receptor type its own colour", () => {
    const colours = ["L", "M", "S", "UV", "blue", "green"].map((name) => RECEPTOR_COLORS[name]);
    expect(colours.every(Boolean)).toBe(true);
    expect(RECEPTOR_COLORS.M).toBe(RECEPTOR_COLORS.green);
    expect(new Set([RECEPTOR_COLORS.L, RECEPTOR_COLORS.M, RECEPTOR_COLORS.S, RECEPTOR_COLORS.UV]).size).toBe(4);
  });

  it("writes stage names in plain words", () => {
    expect(stageLabel("center_surround")).toBe("retina");
    expect(stageLabel("gabor")).toBe("cortex");
    expect(stageLabel("something_new")).toBe("something new");
  });
});

describe("chart geometry", () => {
  it("picks round ticks inside a linear range", () => {
    expect(niceTicks([0, 10], false, 5)).toEqual([0, 2, 4, 6, 8, 10]);
  });

  it("picks powers of ten on a log range", () => {
    expect(niceTicks([0.01, 100], true, 5)).toEqual([0.01, 0.1, 1, 10, 100]);
  });

  it("thins log ticks when the range spans many powers of ten", () => {
    expect(niceTicks([0.01, 1e5], true, 4)).toEqual([0.01, 1, 100, 10000]);
  });

  it("draws a path through every finite point and skips the rest", () => {
    const path = linePath([[0, 0], [1, Number.NaN], [2, 4]], (x) => x * 10, (y) => 100 - y);
    expect(path).toBe("M0,100L20,96");
  });
});
