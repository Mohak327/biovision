import { useEffect, useRef, useState } from "react";
import {
  type ImageSource, type Settings, type SpeciesInfo, fetchSamples, fetchSpecies, sampleUrl,
} from "./api";
import { Controls } from "./components/Controls";
import { Eyepiece } from "./components/Eyepiece";
import { Measurements } from "./components/Measurements";
import { Pathway } from "./components/Pathway";
import { Retina } from "./components/Retina";
import { SignalPath, runEstimate, runStatus } from "./components/SignalPath";
import { SpeciesPicker } from "./components/SpeciesPicker";
import { ThemeToggle } from "./components/ThemeToggle";
import { TitleEye } from "./components/TitleEye";
import { DEFAULT_SETTINGS, capitalised } from "./presets";
import { useTheme } from "./theme";
import { useRun } from "./useRun";

const WAITING = [0, 1, 2]; // placeholders shown until the server lists its pictures
const SERVER_HELP = "The Python server is not answering. Start it with: biovision serve";

/** One measurement, or a ghost bar while the run has not produced it yet. */
function Reading({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value ?? <span className="ghost" role="img" aria-label="not ready yet" />}</dd>
    </div>
  );
}

export function App() {
  const [species, setSpecies] = useState<SpeciesInfo[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [source, setSource] = useState<ImageSource | null>(null);
  const [loadProblem, setLoadProblem] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [theme, toggleTheme] = useTheme();

  useEffect(() => {
    Promise.all([fetchSpecies(), fetchSamples()])
      .then(([speciesList, sampleList]) => {
        setSpecies(speciesList);
        setSamples(sampleList);
        setSource({ kind: "sample", name: sampleList[0] });
      })
      .catch(() => setLoadProblem(SERVER_HELP));
  }, []);

  const run = useRun(settings, source);
  const current = species.find((item) => item.name === settings.species);
  const result = run.status === "done" ? run.result : null;

  const change = (patch: Partial<Settings>) => setSettings((previous) => ({ ...previous, ...patch }));
  const metrics = result?.metrics;

  return (
    <div className="page">
      <header className="masthead">
        <div className="masthead-top">
          <h1><TitleEye species={settings.species} />SpikeSight</h1>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
        <p className="lede">Your eye turns the world into spikes. We turn the spikes back.</p>
        <p className="lede-more">
          See what a human, a mouse and a fruit fly actually keep of a picture, rebuilt with
          maths alone.
        </p>
      </header>

      {loadProblem && <p className="problem" role="alert">{loadProblem}</p>}

      <section className="chooser" aria-label="Picture and eye">
        <div className="pictures">
          <span className="chooser-label" id="picture-label">Picture</span>
          <div className="picture-list" role="radiogroup" aria-labelledby="picture-label">
            {samples.length === 0 && !loadProblem && WAITING.map((place) => (
              <div key={place} className="picture" aria-hidden="true">
                <span className="ghost picture-ghost" />
                <span className="ghost" />
              </div>
            ))}
            {samples.map((name) => {
              const active = source?.kind === "sample" && source.name === name;
              return (
                <button
                  key={name}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  className="picture"
                  onClick={() => setSource({ kind: "sample", name })}
                >
                  <img src={sampleUrl(name)} alt="" />
                  <span>{capitalised(name)}</span>
                </button>
              );
            })}
            <button
              type="button"
              role="radio"
              aria-checked={source?.kind === "file"}
              className="picture picture-upload"
              onClick={() => fileInput.current?.click()}
            >
              <span className="picture-plus" aria-hidden="true">+</span>
              <span>{source?.kind === "file" ? source.file.name : "Your own"}</span>
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="image/png,image/jpeg"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) setSource({ kind: "file", file });
                event.target.value = "";
              }}
            />
          </div>
        </div>
        <div>
          <span className="chooser-label">Eye</span>
          <SpeciesPicker species={species} waiting={!loadProblem} selected={settings.species} onSelect={(name) => change({ species: name })} />
        </div>
      </section>

      <main className="bench">
        <div className="bench-view">
          <Eyepiece
            original={run.original}
            reconstruction={run.frame}
            speciesName={settings.species}
            busy={run.status === "running"}
          />
          <p className="status" role="status" data-state={run.status}>{runStatus(run, settings.species)}</p>
          <p className="status-estimate">{runEstimate(run, settings)}</p>
          {result && !result.converged && (
            <p className="measure-note">The solver stopped before fully settling; this is its best estimate.</p>
          )}
          <dl className="readout" aria-label="Result" aria-busy={!metrics}>
            <Reading label="Match to the original" value={metrics && `${metrics.psnr_db.toFixed(1)} dB`} />
            <Reading label="Structure kept" value={metrics && metrics.ssim.toFixed(2)} />
            <Reading label="Neurons" value={metrics && Math.round(metrics.neurons).toLocaleString()} />
            <Reading label="Receptors" value={metrics && Math.round(metrics.receptors).toLocaleString()} />
          </dl>
        </div>
        <Controls settings={settings} species={current} onChange={change} />
      </main>

      <section className="journey" aria-labelledby="journey-title">
        <h2 id="journey-title">From light to spikes and back</h2>
        <SignalPath run={run} expected={current?.stages ?? []} />
        {run.mosaic && (
          <div className="journey-retina">
            <div>
              <h3>The {settings.species}'s receptors</h3>
              <p>
                {run.mosaic.count.toLocaleString()} receptors sample this picture. {current?.description}
              </p>
            </div>
            <Retina mosaic={run.mosaic} responses={result?.responses ?? null} />
          </div>
        )}
      </section>

      <Pathway species={settings.species} />

      {result && source && <Measurements result={result} settings={settings} source={source} />}

      <footer className="colophon">
        <p>
          Every stage is a published model of the eye, and the picture is rebuilt with linear algebra.
          Nothing here is trained.
        </p>
        <p className="colophon-credit">
          © 2026 <a href="https://moksh-mandala.vercel.app/" target="_blank" rel="noreferrer">Mohak Sharma</a>.
          Made with <span role="img" aria-label="love">❤️</span>
        </p>
      </footer>
    </div>
  );
}
