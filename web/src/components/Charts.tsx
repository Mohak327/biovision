// Hand-drawn SVG charts. d3 supplies the scales; everything else is plain SVG.
import { type Point, extent, formatTick, linePath, makeScale, niceTicks } from "../chart";
import { capitalised } from "../presets";

const WIDTH = 360;
const HEIGHT = 230;
const MARGIN = { top: 12, right: 14, bottom: 42, left: 52 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;
const INNER_H = HEIGHT - MARGIN.top - MARGIN.bottom;

export type Series = { name: string; color: string; points: Point[]; dots?: boolean };

type LineProps = {
  title: string;
  series: Series[];
  xLabel: string;
  yLabel: string;
  xLog?: boolean;
  yLog?: boolean;
  marker?: { x: number; label: string };
};

export function LineChart({ title, series, xLabel, yLabel, xLog = false, yLog = false, marker }: LineProps) {
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const xDomain = extent(xs, xLog);
  const yDomain = extent(ys, yLog);
  const x = makeScale(xDomain, [0, INNER_W], xLog);
  const y = makeScale(yDomain, [INNER_H, 0], yLog);
  const usable = (p: Point) => (!xLog || p[0] > 0) && (!yLog || p[1] > 0);
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`${title}. ${yLabel} against ${xLabel}.`}>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {niceTicks(yDomain, yLog, 4).map((tick) => (
            <g key={`y${tick}`} transform={`translate(0,${y(tick)})`}>
              <line x2={INNER_W} className="chart-grid" />
              <text x={-8} dy="0.32em" textAnchor="end" className="chart-tick">{formatTick(tick)}</text>
            </g>
          ))}
          {niceTicks(xDomain, xLog, 5).map((tick) => (
            <text key={`x${tick}`} x={x(tick)} y={INNER_H + 16} textAnchor="middle" className="chart-tick">
              {formatTick(tick)}
            </text>
          ))}
          <line y1={INNER_H} y2={INNER_H} x2={INNER_W} className="chart-axis" />
          {marker && marker.x >= xDomain[0] && marker.x <= xDomain[1] && (
            <g transform={`translate(${x(marker.x)},0)`}>
              <line y2={INNER_H} className="chart-marker" />
              <text x={-5} y={10} textAnchor="end" className="chart-tick">{marker.label}</text>
            </g>
          )}
          {series.map((s) => {
            const points = s.points.filter(usable);
            return (
              <g key={s.name}>
                <path d={linePath(points, x, y)} fill="none" stroke={s.color} strokeWidth={2} />
                {s.dots && points.map((p) => <circle key={p[0]} cx={x(p[0])} cy={y(p[1])} r={3.5} fill={s.color} />)}
              </g>
            );
          })}
          <text x={INNER_W / 2} y={INNER_H + 36} textAnchor="middle" className="chart-label">{xLabel}</text>
          <text transform={`translate(${-40},${INNER_H / 2}) rotate(-90)`} textAnchor="middle" className="chart-label">
            {yLabel}
          </text>
        </g>
      </svg>
      {series.length > 1 && (
        <ul className="chart-legend">
          {series.map((s) => (
            <li key={s.name}><span className="swatch" style={{ background: s.color }} />{capitalised(s.name)}</li>
          ))}
        </ul>
      )}
    </figure>
  );
}

type BarProps = {
  title: string;
  bars: { label: string; value: number; color: string }[];
  yLabel: string;
  xLabel?: string;
  showLabels?: boolean;
};

export function BarChart({ title, bars, yLabel, xLabel, showLabels = true }: BarProps) {
  const top = Math.max(...bars.map((bar) => bar.value), 1e-9);
  const y = makeScale([0, top], [INNER_H, 0], false);
  const band = INNER_W / bars.length;
  const gap = bars.length > 12 ? 1 : band * 0.2;
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`${title}. ${yLabel}.`}>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {niceTicks([0, top], false, 4).map((tick) => (
            <g key={tick} transform={`translate(0,${y(tick)})`}>
              <line x2={INNER_W} className="chart-grid" />
              <text x={-8} dy="0.32em" textAnchor="end" className="chart-tick">{formatTick(tick)}</text>
            </g>
          ))}
          {bars.map((bar, index) => (
            <g key={`${bar.label}${index}`}>
              <rect
                x={index * band + gap / 2}
                y={y(bar.value)}
                width={Math.max(band - gap, 1)}
                height={INNER_H - y(bar.value)}
                fill={bar.color}
                rx={bars.length > 12 ? 0 : 3}
              />
              {showLabels && (
                <text x={index * band + band / 2} y={INNER_H + 16} textAnchor="middle" className="chart-tick">
                  {bar.label}
                </text>
              )}
            </g>
          ))}
          <line y1={INNER_H} y2={INNER_H} x2={INNER_W} className="chart-axis" />
          {xLabel && <text x={INNER_W / 2} y={INNER_H + 36} textAnchor="middle" className="chart-label">{xLabel}</text>}
          <text transform={`translate(${-40},${INNER_H / 2}) rotate(-90)`} textAnchor="middle" className="chart-label">
            {yLabel}
          </text>
        </g>
      </svg>
    </figure>
  );
}
