import { describe, expect, it } from "vitest";
import { runProgress } from "./progress";

const STAGES = ["color", "optics", "mosaic", "decoding"];
const running = (current: number, fraction = 0) =>
  runProgress({ status: "running", stages: STAGES, current, fraction });

describe("runProgress", () => {
  it("is empty before a run and before its first step", () => {
    expect(runProgress({ status: "idle", stages: [], current: -1, fraction: 0 })).toBe(0);
    expect(runProgress({ status: "running", stages: [], current: -1, fraction: 0 })).toBe(0);
    expect(running(-1)).toBe(0);
  });

  it("counts each step once its output is showing", () => {
    expect(running(0)).toBe(0.25);
    expect(running(1)).toBe(0.5);
    expect(running(2)).toBe(0.75);
  });

  it("fills the last step by how far the solve has got", () => {
    expect(running(3, 0)).toBe(0.75);
    expect(running(3, 0.5)).toBe(0.875);
    expect(running(3, 1)).toBe(1);
  });

  it("never passes the end, and is full when the run is done", () => {
    expect(running(3, 7)).toBe(1);
    expect(running(9, 0.2)).toBeLessThanOrEqual(1);
    expect(runProgress({ status: "done", stages: STAGES, current: 0, fraction: 0 })).toBe(1);
  });

  it("is empty after an error", () => {
    expect(runProgress({ status: "error", stages: STAGES, current: 2, fraction: 0 })).toBe(0);
  });
});
