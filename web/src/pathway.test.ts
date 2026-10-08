import { describe, expect, it } from "vitest";
import anatomy from "./data/anatomy.json";
import { PIPELINE_STAGES, ROUTES, pointOnPath, routeFor, stopStatus } from "./pathway";

const SPECIES = ["human", "mouse", "fly"];
const built = anatomy as unknown as Record<string, { stops: Record<string, unknown>; faces: Record<string, number> }>;

describe("the route of seeing", () => {
  it("has a route for each species and none for an unknown one", () => {
    expect(Object.keys(ROUTES).sort()).toEqual([...SPECIES].sort());
    expect(routeFor("cat")).toBeNull();
  });

  it.each(SPECIES)("%s: runs in order from light to the last stop, with unique ids", (name) => {
    const stops = ROUTES[name].stops;
    const ids = stops.map((stop) => stop.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids[0]).toBe("light");
    expect(ids.indexOf("optics")).toBeLessThan(ids.indexOf("receptors"));
    expect(ids.length).toBeGreaterThanOrEqual(7);
    // The stops are exactly the ones the build script placed, in the model's own order.
    expect([...ids].sort()).toEqual(Object.keys(built[name].stops).sort());
  });

  it("covers each species' own route", () => {
    const ids = (name: string) => ROUTES[name].stops.map((stop) => stop.id);
    expect(ids("human")).toEqual(["light", "optics", "receptors", "bipolar", "ganglion", "nerve", "chiasm",
      "tract", "geniculate", "radiation", "cortex", "beyond"]);
    expect(ids("mouse")).toEqual(["light", "optics", "receptors", "bipolar", "ganglion", "nerve", "chiasm",
      "tract", "colliculus", "geniculate", "cortex", "beyond"]);
    expect(ids("fly")).toEqual(["light", "optics", "receptors", "lamina", "medulla", "lobula", "brain"]);
  });

  it.each(SPECIES)("%s: every stop is described and placed", (name) => {
    for (const stop of ROUTES[name].stops) {
      expect(stop.name.length).toBeGreaterThan(2);
      expect(stop.what.length).toBeGreaterThan(40);
      expect(stop.position).toHaveLength(3);
      expect(stop.position.every(Number.isFinite)).toBe(true);
      expect(stop.size).toBeGreaterThan(0);
    }
  });

  it.each(SPECIES)("%s: stops light only parts that are in the model, and every part has a look", (name) => {
    const parts = Object.keys(built[name].faces);
    expect(Object.keys(ROUTES[name].looks).sort()).toEqual([...parts].sort());
    for (const stop of ROUTES[name].stops) {
      for (const part of stop.parts) expect(parts).toContain(part);
    }
  });

  it.each(SPECIES)("%s: a modelled stop names real pipeline stages", (name) => {
    for (const stop of ROUTES[name].stops) {
      for (const stage of stop.stages) expect(PIPELINE_STAGES).toContain(stage);
    }
    const modelled = ROUTES[name].stops.flatMap((stop) => stop.stages);
    for (const stage of ["color", "optics", "mosaic", "center_surround", "rate", "spikes"]) {
      expect(modelled).toContain(stage);
    }
    expect(modelled.includes("gabor")).toBe(name !== "fly"); // only mammals have the cortex stage
  });

  it.each(SPECIES)("%s: credits its source and names what is drawn", (name) => {
    const route = ROUTES[name];
    expect(route.credit.length).toBeGreaterThan(40);
    expect(route.label.length).toBeGreaterThan(20);
    const drawn = (anatomy as unknown as Record<string, { drawn: string[] }>)[name].drawn;
    expect(/drawn/i.test(route.credit)).toBe(drawn.length > 0);
  });

  it("says in words whether a stop is in the model", () => {
    const stop = (id: string) => ROUTES.human.stops.find((one) => one.id === id)!;
    expect(stopStatus(stop("optics"))).toBe("In the model: the optics stage.");
    expect(stopStatus(stop("cortex"))).toBe("In the model: the cortex, firing rate and spikes stages.");
    expect(stopStatus(stop("chiasm"))).toBe("Not in the model.");
  });
});

describe("pointOnPath", () => {
  const path: [number, number, number][] = [[0, 0, 0], [2, 0, 0], [2, 2, 0]];

  it("moves at even speed along the line and wraps past the end", () => {
    expect(pointOnPath(path, 0)).toEqual([0, 0, 0]);
    expect(pointOnPath(path, 0.25)).toEqual([1, 0, 0]);
    expect(pointOnPath(path, 0.75)).toEqual([2, 1, 0]);
    expect(pointOnPath(path, 1)).toEqual([2, 2, 0]);
    expect(pointOnPath(path, 1.25)).toEqual([1, 0, 0]);
  });

  it("copes with two stops at the same place", () => {
    expect(pointOnPath([[0, 0, 0], [0, 0, 0], [4, 0, 0]], 0.5)).toEqual([2, 0, 0]);
  });
});
