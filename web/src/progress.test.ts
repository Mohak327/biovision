import { describe, expect, it } from "vitest";
import { estimatedSeconds, roundDuration, runProgress } from "./progress";

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

describe("estimatedSeconds", () => {
  const human = { species: "human", size_px: 96, neuron_density: 1 };

  it("is the measured time at the reference size", () => {
    expect(estimatedSeconds(human)).toBe(40);
  });

  it("grows with the number of pixels and of neurons", () => {
    expect(estimatedSeconds({ ...human, size_px: 192 })).toBe(160);
    expect(estimatedSeconds({ ...human, neuron_density: 0.5 })).toBe(20);
  });

  it("is much shorter for the smaller eyes", () => {
    expect(estimatedSeconds({ ...human, species: "fly" })).toBeLessThan(5);
  });
});

describe("roundDuration", () => {
  it("rounds to words a person would use", () => {
    expect(roundDuration(2)).toBe("about 5 seconds");
    expect(roundDuration(23)).toBe("about 25 seconds");
    expect(roundDuration(70)).toBe("about a minute");
    expect(roundDuration(290)).toBe("about 5 minutes");
  });
});
