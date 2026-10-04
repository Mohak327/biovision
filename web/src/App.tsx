import { useEffect, useMemo, useRef, useState } from "react";
import {
  type ImageSource, type Settings, type SpeciesInfo, fetchSamples, fetchSpecies, pngUrl, sampleUrl,
} from "./api";
import { Controls } from "./components/Controls";
import { Eyepiece } from "./components/Eyepiece";
import { Measurements } from "./components/Measurements";
import { Retina } from "./components/Retina";
import { SignalPath, runStatus } from "./components/SignalPath";
import { SpeciesPicker } from "./components/SpeciesPicker";
import { DEFAULT_SETTINGS } from "./presets";
import { useRun } from "./useRun";

const SERVER_HELP = "The Python server is not answering. Start it with: biovision serve";

export function App() {
  const [species, setSpecies] = useState<SpeciesInfo[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [source, setSource] = useState<ImageSource | null>(null);
  const [loadProblem, setLoadProblem] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

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

  // The picture as given, until the run returns it at the size the eye saw.
  const sourceUrl = useMemo(() => {
    if (!source) return null;
    return source.kind === "sample" ? sampleUrl(source.name) : URL.createObjectURL(source.file);
  }, [source]);
  useEffect(() => () => {
    if (sourceUrl?.startsWith("blob:")) URL.revokeObjectURL(sourceUrl);
  }, [sourceUrl]);

  const change = (patch: Partial<Settings>) => setSettings((previous) => ({ ...previous, ...patch }));
  const metrics = result?.metrics;

  return (
    <div className="page">
      <header className="masthead">
        <h1>biovision</h1>
        <p className="lede">
          A picture goes into an eye as light and leaves as spikes. This rebuilds the
          picture from those spikes alone, so you can see what each eye keeps.
        </p>
      </header>

      {loadProblem && <p className="problem" role="alert">{loadProblem}</p>}

      <section className="chooser" aria-label="Picture and eye">
        <div className="pictures">
          <span className="chooser-label" id="picture-label">Picture</span>
          <div className="picture-list" role="radiogroup" aria-labelledby="picture-label">
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
                  <span>{name}</span>
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
              <span>{source?.kind === "file" ? source.file.name : "your own"}</span>
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
          <SpeciesPicker species={species} selected={settings.species} onSelect={(name) => change({ species: name })} />
        </div>
      </section>

      <main className="bench">
        <div className="bench-view">
          <Eyepiece
            original={result ? pngUrl(result.original) : sourceUrl}
            reconstruction={run.frame}
            speciesName={settings.species}
            busy={run.status === "running"}
          />
          <p className="status" role="status" data-state={run.status}>{runStatus(run, settings.species)}</p>
          {result && !result.converged && (
            <p className="measure-note">The solver stopped before fully settling; this is its best estimate.</p>
          )}
          <dl className="readout" aria-label="Result">
            <div><dt>Match to the original</dt><dd>{metrics ? `${metrics.psnr_db.toFixed(1)} dB` : "—"}</dd></div>
            <div><dt>Structure kept</dt><dd>{metrics ? metrics.ssim.toFixed(2) : "—"}</dd></div>
            <div><dt>Neurons</dt><dd>{metrics ? Math.round(metrics.neurons).toLocaleString() : "—"}</dd></div>
            <div><dt>Receptors</dt><dd>{metrics ? Math.round(metrics.receptors).toLocaleString() : "—"}</dd></div>
          </dl>
        </div>
        <Controls settings={settings} species={current} onChange={change} />
      </main>

      <section className="journey" aria-labelledby="journey-title">
        <h2 id="journey-title">From light to spikes and back</h2>
        <SignalPath run={run} />
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

      {result && source && <Measurements result={result} settings={settings} source={source} />}

      <footer className="colophon">
        Every stage is a published model of the eye, and the picture is rebuilt with linear algebra.
        Nothing here is trained.
      </footer>
    </div>
  );
}
