// Geometry for the hand-drawn SVG charts.
import { scaleLinear, scaleLog } from "d3-scale";

export type Point = [number, number];

/** Round tick values inside a range: powers of ten on a log axis. */
export function niceTicks(domain: [number, number], log: boolean, count: number): number[] {
  if (!log) return scaleLinear().domain(domain).ticks(count);
  const [low, high] = domain;
  const first = Math.ceil(Math.log10(low) - 1e-9);
  const last = Math.floor(Math.log10(high) + 1e-9);
  const step = Math.max(1, Math.ceil((last - first + 1) / count));
  const ticks: number[] = [];
  for (let power = first; power <= last; power += step) ticks.push(Number(`1e${power}`));
  return ticks;
}

export function makeScale(domain: [number, number], range: [number, number], log: boolean) {
  return log ? scaleLog().domain(domain).range(range) : scaleLinear().domain(domain).range(range);
}

/** An SVG path through the finite points; others are skipped. */
export function linePath(points: Point[], x: (v: number) => number, y: (v: number) => number): string {
  return points
    .filter(([px, py]) => Number.isFinite(px) && Number.isFinite(py))
    .map(([px, py], index) => `${index ? "L" : "M"}${round(x(px))},${round(y(py))}`)
    .join("");
}

const round = (value: number): number => Math.round(value * 100) / 100;

/** The smallest and largest finite values, widened if they coincide. */
export function extent(values: number[], log: boolean): [number, number] {
  const finite = values.filter((v) => Number.isFinite(v) && (!log || v > 0));
  if (!finite.length) return log ? [0.1, 1] : [0, 1];
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  if (low === high) return log ? [low / 2, high * 2] : [low - 1, high + 1];
  return [low, high];
}

export function formatTick(value: number): string {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 10000 || magnitude < 0.01) return value.toExponential(0).replace("e+", "e");
  return String(Number(value.toPrecision(3)));
}
