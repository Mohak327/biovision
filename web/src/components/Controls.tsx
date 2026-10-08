// Named choices instead of raw sliders, grouped by the part of the eye they change.
import { useId } from "react";
import type { Settings, SpeciesInfo } from "../api";
import {
  DENSITY_PRESETS, FOV_PRESETS, NEURON_PRESETS, type Preset, SIZE_PRESETS, WINDOW_PRESETS,
  capitalised,
} from "../presets";

type ChoiceProps = {
  legend: string;
  presets: Preset[];
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  note?: string;
};

function Choice({ legend, presets, value, onChange, disabled, note }: ChoiceProps) {
  const name = useId();
  const active = presets.find((preset) => preset.value === value);
  return (
    <fieldset className="choice" disabled={disabled}>
      <legend>{legend}</legend>
      <div className="choice-options">
        {presets.map((preset) => (
          <label key={preset.value} className="choice-option">
            <input
              type="radio"
              name={name}
              checked={preset.value === value}
              onChange={() => onChange(preset.value)}
            />
            <span>{capitalised(preset.label)}</span>
          </label>
        ))}
      </div>
      <p className="choice-hint">{note ?? active?.hint ?? `${value}`}</p>
    </fieldset>
  );
}

type Props = {
  settings: Settings;
  species: SpeciesInfo | undefined;
  onChange: (patch: Partial<Settings>) => void;
};

const NOISE_PRESETS: Preset[] = [
  { value: 1, label: "real neurons", hint: "spikes arrive at random, as in a living cell" },
  { value: 0, label: "ideal neurons", hint: "exact spike counts, with no randomness" },
];

export function Controls({ settings, species, onChange }: Props) {
  const lamId = useId();
  const seedId = useId();
  const manual = settings.lam !== null;
  return (
    <form className="controls" onSubmit={(event) => event.preventDefault()}>
      <h2>Change the eye</h2>
      <Choice
        legend="Receptors in the eye"
        presets={DENSITY_PRESETS}
        value={settings.density}
        onChange={(density) => onChange({ density })}
      />
      <Choice
        legend="Cells in the cortex"
        presets={NEURON_PRESETS}
        value={settings.neuron_density}
        onChange={(neuron_density) => onChange({ neuron_density })}
        disabled={species ? !species.has_cortex : false}
        note={species && !species.has_cortex ? `The ${species.name} has no cortex stage.` : undefined}
      />
      <Choice
        legend="How long it looks"
        presets={WINDOW_PRESETS}
        value={settings.window_ms}
        onChange={(window_ms) => onChange({ window_ms })}
      />
      <Choice
        legend="How its neurons fire"
        presets={NOISE_PRESETS}
        value={settings.noise ? 1 : 0}
        onChange={(value) => onChange({ noise: value === 1 })}
      />
      <Choice
        legend="How much of its view the picture fills"
        presets={FOV_PRESETS}
        value={settings.fov_deg}
        onChange={(fov_deg) => onChange({ fov_deg })}
      />
      <Choice
        legend="Picture detail"
        presets={SIZE_PRESETS}
        value={settings.size_px}
        onChange={(size_px) => onChange({ size_px })}
      />
      <details className="advanced">
        <summary>Advanced</summary>
        <label className="field">
          <input
            type="checkbox"
            checked={!manual}
            onChange={(event) => onChange({ lam: event.target.checked ? null : 0.001 })}
          />
          Set the smoothing from the spike noise
        </label>
        {manual && (
          <label className="field" htmlFor={lamId}>
            Smoothing strength (lambda)
            <input
              id={lamId}
              type="number"
              min={0.00001}
              step="any"
              value={settings.lam ?? 0.001}
              onChange={(event) => {
                const lam = Number(event.target.value);
                if (lam > 0) onChange({ lam });
              }}
            />
          </label>
        )}
        <label className="field" htmlFor={seedId}>
          Random seed for the spikes
          <input
            id={seedId}
            type="number"
            min={0}
            step={1}
            value={settings.seed}
            onChange={(event) => onChange({ seed: Math.max(0, Math.floor(Number(event.target.value) || 0)) })}
          />
        </label>
      </details>
    </form>
  );
}
