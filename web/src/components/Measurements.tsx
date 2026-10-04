// Numbers for a report: charts of the finished run, sweeps and comparison on
// request, and the downloadable report.
import { useEffect, useState } from "react";
import {
  type ImageSource, type RunSummary, type Settings, type SweepKind, type SweepRow,
  fetchReport, fetchSweep, pngUrl, streamCompare,
} from "../api";
import { BarChart, LineChart } from "./Charts";

const INK = "var(--ink)";
const REBUILT = "var(--rebuilt)";
const CHANNELS = [
  { label: "red", color: "#d1495b" },
  { label: "green", color: "#2e9e6b" },
  { label: "blue", color: "#3f7be0" },
];

const SWEEPS: { kind: SweepKind; button: string; title: string; x: string; xLabel: string; y: string; log: boolean }[] = [
  { kind: "window", button: "Looking time", title: "Quality against looking time", x: "window_ms", xLabel: "spike window (ms)", y: "psnr_mean", log: true },
  { kind: "density", button: "Receptor count", title: "Quality against receptor density", x: "density", xLabel: "receptor density (1 = the real eye)", y: "psnr_db", log: true },
  { kind: "lambda", button: "Smoothing", title: "Quality against smoothing", x: "lam", xLabel: "smoothing strength (lambda)", y: "psnr_db", log: true },
];

type Props = { result: RunSummary; settings: Settings; source: ImageSource };

export function Measurements({ result, settings, source }: Props) {
  const [sweeps, setSweeps] = useState<Partial<Record<SweepKind, SweepRow[]>>>({});
  const [compared, setCompared] = useState<RunSummary[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  // Sweeps and comparisons describe one set of settings; clear them when it changes.
  const key = JSON.stringify(settings) + (source.kind === "sample" ? source.name : source.file.name);
  useEffect(() => {
    setSweeps({});
    setCompared([]);
    setProblem(null);
  }, [key]);

  const guard = async (label: string, work: () => Promise<void>) => {
    setBusy(label);
    setProblem(null);
    try {
      await work();
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(null);
    }
  };

  const runSweep = (kind: SweepKind) => guard(kind, async () => {
    const rows = await fetchSweep(kind, settings, source);
    setSweeps((previous) => ({ ...previous, [kind]: rows }));
  });

  const runCompare = () => guard("compare", async () => {
    setCompared([]);
    for await (const summary of streamCompare(settings, source)) {
      setCompared((previous) => [...previous, summary]);
    }
  });

  const download = (full: boolean) => guard(full ? "full report" : "report", async () => {
    const blob = await fetchReport(settings, source, full);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "biovision_report.zip";
    link.click();
    URL.revokeObjectURL(link.href);
  });

  const { spectrum, spike_histogram: spikes } = result;
  const windowMs = result.settings.window_ms;
  return (
    <section className="measure" aria-labelledby="measure-title">
      <h2 id="measure-title">Measurements</h2>
      <div className="chart-grid-layout">
        <LineChart
          title="Detail kept at each scale"
          xLabel="spatial frequency (cycles per degree)"
          yLabel="power"
          xLog
          yLog
          marker={{ x: spectrum.limit_cpd, label: "eye's limit" }}
          series={[
            { name: "original", color: INK, points: spectrum.cpd.map((f, i) => [f, spectrum.original[i]]) },
            { name: "rebuilt", color: REBUILT, points: spectrum.cpd.map((f, i) => [f, spectrum.reconstructed[i]]) },
          ]}
        />
        <BarChart
          title="Error in each colour channel"
          yLabel="root-mean-square error"
          bars={CHANNELS.map((channel, i) => ({ ...channel, value: result.channel_rmse[i] }))}
        />
        <BarChart
          title={`Spikes per neuron in ${windowMs} ms`}
          yLabel="neurons"
          xLabel={`from ${Math.round(spikes.edges[0])} to ${Math.round(spikes.edges[spikes.edges.length - 1])} spikes`}
          showLabels={false}
          bars={spikes.counts.map((value, i) => ({ label: String(i), value, color: INK }))}
        />
        <LineChart
          title="How the solver closed in"
          xLabel="step"
          yLabel="remaining error"
          yLog
          series={[{ name: "error", color: INK, points: result.residuals.map((r, i) => [i + 1, r]) }]}
        />
      </div>

      <h3>Try a range</h3>
      <p className="measure-note">Each of these runs the eye several times, so it takes a while.</p>
      <div className="actions">
        {SWEEPS.map((sweep) => (
          <button key={sweep.kind} type="button" onClick={() => runSweep(sweep.kind)} disabled={busy !== null}>
            {busy === sweep.kind ? "Running…" : `Vary ${sweep.button.toLowerCase()}`}
          </button>
        ))}
        <button type="button" onClick={runCompare} disabled={busy !== null}>
          {busy === "compare" ? "Running…" : "Compare all three eyes"}
        </button>
      </div>
      {problem && <p className="problem" role="alert">{problem}</p>}
      <div className="chart-grid-layout">
        {SWEEPS.filter((sweep) => sweeps[sweep.kind]).map((sweep) => (
          <LineChart
            key={sweep.kind}
            title={sweep.title}
            xLabel={sweep.xLabel}
            yLabel="PSNR (dB)"
            xLog={sweep.log}
            series={[{
              name: result.settings.species, color: REBUILT, dots: true,
              points: sweeps[sweep.kind]!.map((row) => [Number(row[sweep.x]), Number(row[sweep.y])]),
            }]}
          />
        ))}
      </div>
      {compared.length > 0 && (
        <ul className="compare">
          {compared.map((item) => (
            <li key={item.settings.species}>
              <img src={pngUrl(item.reconstructed)} alt={`The picture as rebuilt from the ${item.settings.species}'s spikes`} />
              <strong>{item.settings.species}</strong>
              <span>{item.metrics.psnr_db.toFixed(1)} dB from {Math.round(item.metrics.neurons).toLocaleString()} neurons</span>
            </li>
          ))}
        </ul>
      )}

      <h3>Take it with you</h3>
      <div className="actions">
        <button type="button" onClick={() => download(false)} disabled={busy !== null}>
          {busy === "report" ? "Preparing…" : "Download this eye's report"}
        </button>
        <button type="button" onClick={() => download(true)} disabled={busy !== null}>
          {busy === "full report" ? "Preparing…" : "Download the full report"}
        </button>
      </div>
      <p className="measure-note">
        The report has every figure as PNG and PDF, every table as CSV, and a written summary.
        The full report covers all three eyes and the ranges above, and takes a few minutes.
      </p>

      <details className="advanced">
        <summary>The numbers behind the {result.settings.species} eye</summary>
        <p>{result.description}</p>
        <table>
          <thead><tr><th>Parameter</th><th>Value</th><th>Unit</th></tr></thead>
          <tbody>
            {result.parameters.map((row) => (
              <tr key={row.parameter}><td>{row.parameter.replaceAll("_", " ")}</td><td>{row.value}</td><td>{row.unit}</td></tr>
            ))}
          </tbody>
        </table>
        <h4>Sources</h4>
        <ol className="sources">{result.citations.map((citation) => <li key={citation}>{citation}</li>)}</ol>
      </details>
    </section>
  );
}
