// Three cards, each drawn with that eye's own receptor pattern.
import { motion } from "motion/react";
import type { SpeciesInfo } from "../api";
import { RECEPTOR_COLORS } from "../presets";

const SIZE = 72;

/** Dots laid out the way each eye's receptors are: rings, a grid, or hexagons. */
function patternDots(name: string): [number, number][] {
  const dots: [number, number][] = [];
  const centre = SIZE / 2;
  if (name === "fly") {
    const step = 11;
    for (let row = 0; row * step * 0.866 < SIZE; row += 1) {
      for (let col = 0; col * step < SIZE + step; col += 1) {
        dots.push([col * step + (row % 2 ? step / 2 : 0), row * step * 0.866 + 4]);
      }
    }
  } else if (name === "mouse") {
    const step = 9;
    for (let y = step / 2; y < SIZE; y += step) for (let x = step / 2; x < SIZE; x += step) dots.push([x, y]);
  } else {
    dots.push([centre, centre]);
    let radius = 0;
    let spacing = 3;
    while (radius < centre * 1.4) {
      radius += spacing;
      const count = Math.max(6, Math.round((2 * Math.PI * radius) / spacing));
      for (let i = 0; i < count; i += 1) {
        const angle = (2 * Math.PI * i) / count;
        dots.push([centre + radius * Math.cos(angle), centre + radius * Math.sin(angle)]);
      }
      spacing *= 1.22;
    }
  }
  return dots.filter(([x, y]) => Math.hypot(x - centre, y - centre) < centre - 2);
}

const FACTS: Record<string, string> = {
  human: "Sharp, three colours",
  mouse: "Blurred, no red",
  fly: "About 500 facets, no red",
};

type Props = { species: SpeciesInfo[]; selected: string; onSelect: (name: string) => void };

export function SpeciesPicker({ species, selected, onSelect }: Props) {
  return (
    <div className="species" role="radiogroup" aria-label="Whose eye">
      {species.map((item) => {
        const dots = patternDots(item.name);
        const active = item.name === selected;
        return (
          <button
            key={item.name}
            type="button"
            role="radio"
            aria-checked={active}
            className="species-card"
            onClick={() => onSelect(item.name)}
          >
            {active && <motion.span layoutId="species-ring" className="species-ring" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
            <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true">
              <circle cx={SIZE / 2} cy={SIZE / 2} r={SIZE / 2 - 1} className="species-well" />
              {dots.map(([x, y], index) => (
                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r={item.name === "fly" ? 2.6 : item.name === "mouse" ? 1.9 : 1.2}
                  fill={RECEPTOR_COLORS[item.receptors[index % item.receptors.length]]}
                />
              ))}
            </svg>
            <span className="species-text">
              <span className="species-name">{item.name}</span>
              <span className="species-fact">{FACTS[item.name] ?? item.receptors.join(", ")}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
