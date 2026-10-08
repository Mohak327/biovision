// Small graphs of a run's own numbers, for the two stages that are not pictures:
// the firing-rate curve with the responses that fell on it, and the spike counts.
import type { RatePlot, SpikePlot, StagePlot } from "../api";
import { formatTick } from "../chart";

const SIZE = 100; // the graph's own units; it is drawn to fill its tile
const LEFT = 8;
const RIGHT = 94;
const TOP = 22;
const BOTTOM = 84;
const LIGHT = "#e9edee"; // tiles sit on the dark well in both themes
const SOFT = "#8fa1ad";
const FAINT = "#31414d";

const isRate = (plot: StagePlot): plot is RatePlot => "response" in plot;
const across = (share: number) => LEFT + share * (RIGHT - LEFT);
const up = (share: number) => BOTTOM - share * (BOTTOM - TOP);

/** Bar heights on a log scale: most cells share one bar, and this keeps the rest visible. */
function Bars({ counts, fill }: { counts: number[]; fill: string }) {
  const most = Math.log1p(Math.max(...counts, 1));
  const width = (RIGHT - LEFT) / counts.length;
  return (
    <>
      {counts.map((count, index) => {
        const share = Math.log1p(count) / most;
        return (
          <rect key={index} x={LEFT + index * width} y={up(share)} width={Math.max(width - 0.4, 0.4)}
                height={BOTTOM - up(share)} fill={fill} />
        );
      })}
    </>
  );
}

function Frame({ top, low, high, children }: { top: string; low: string; high: string; children: React.ReactNode }) {
  return (
    <svg viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true">
      <text x={LEFT} y={12} fontSize="7.5" fill={LIGHT}>{top}</text>
      {children}
      <line x1={LEFT} y1={BOTTOM} x2={RIGHT} y2={BOTTOM} stroke={SOFT} strokeWidth="0.8" />
      <text x={LEFT} y={94} fontSize="7" fill={SOFT}>{low}</text>
      <text x={RIGHT} y={94} fontSize="7" fill={SOFT} textAnchor="end">{high}</text>
    </svg>
  );
}

/** Each kind of cell's rate against the response, over the responses this picture gave. */
function RateGraph({ plot }: { plot: RatePlot }) {
  const { response, rates, cells } = plot;
  const reach = response[response.length - 1];
  const most = Math.max(...Object.values(rates).flat(), 1e-9);
  const x = (value: number) => across((value + reach) / (2 * reach));
  return (
    <Frame top={`to ${formatTick(most)} spikes/s`} low="− response" high="+">
      {/* How many cells gave each response: most sit near zero. */}
      <Bars counts={cells} fill={FAINT} />
      <line x1={x(0)} y1={TOP} x2={x(0)} y2={BOTTOM} stroke={SOFT} strokeWidth="0.6" strokeDasharray="2 2" />
      {Object.entries(rates).map(([kind, values], index) => (
        <polyline key={kind} fill="none" stroke={index ? SOFT : LIGHT} strokeWidth="2" strokeLinejoin="round"
                  points={values.map((rate, i) => `${x(response[i])},${up(rate / most)}`).join(" ")} />
      ))}
    </Frame>
  );
}

/** How many cells fired each number of spikes. */
function SpikeGraph({ plot }: { plot: SpikePlot }) {
  const { edges, cells } = plot;
  const total = cells.reduce((sum, count) => sum + count, 0);
  return (
    <Frame top={`${total.toLocaleString()} cells (log)`} low={`${formatTick(edges[0])} spikes`}
           high={formatTick(edges[edges.length - 1])}>
      <Bars counts={cells} fill={LIGHT} />
    </Frame>
  );
}

export function StageGraph({ plot }: { plot: StagePlot }) {
  return isRate(plot) ? <RateGraph plot={plot} /> : <SpikeGraph plot={plot} />;
}
