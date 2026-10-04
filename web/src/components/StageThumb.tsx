// A small picture for each stage of the path. Stages whose output is not an
// image get a drawing of what they do.
import { useEffect, useRef } from "react";
import { type MosaicEvent, type RunSummary, decodeFloat32, decodeUint8 } from "../api";
import { RECEPTOR_COLORS } from "../presets";

const SIZE = 160;
const MAX_DOTS = 6000;

type DotsProps = { mosaic: MosaicEvent; responses: string | null };

/** The mosaic drawn flat: coloured by type, or lit by response when given. */
function Dots({ mosaic, responses }: DotsProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    const xy = decodeFloat32(mosaic.positions);
    const types = decodeUint8(mosaic.types);
    const levels = responses ? decodeFloat32(responses) : null;
    const stride = Math.max(1, Math.ceil(mosaic.count / MAX_DOTS));
    const radius = Math.min(5, Math.max(0.9, (SIZE / Math.sqrt(mosaic.count / mosaic.receptors.length / stride)) * 0.3));
    const kinds = mosaic.receptors.length;
    const nudge = levels ? 0 : radius * 0.9;
    context.clearRect(0, 0, SIZE, SIZE);
    for (let i = 0; i < mosaic.count; i += stride) {
      const angle = (2 * Math.PI * types[i]) / kinds;
      if (levels && levels.length === mosaic.count) {
        const grey = Math.round(40 + 215 * levels[i]);
        context.fillStyle = `rgb(${grey},${grey},${grey})`;
      } else {
        context.fillStyle = RECEPTOR_COLORS[mosaic.receptors[types[i]]] ?? "#fff";
      }
      context.beginPath();
      const x = 6 + xy[2 * i] * (SIZE - 12) + nudge * Math.cos(angle);
      const y = 6 + xy[2 * i + 1] * (SIZE - 12) + nudge * Math.sin(angle);
      context.arc(x, y, levels ? radius : radius * 0.62, 0, 2 * Math.PI);
      context.fill();
    }
  }, [mosaic, responses]);
  return <canvas ref={canvas} width={SIZE} height={SIZE} />;
}

/** Four edge detectors at different angles, as cortex cells are tuned. */
function Edges() {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    const picture = context.createImageData(SIZE, SIZE);
    const half = SIZE / 2;
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        const quadrant = (x < half ? 0 : 1) + (y < half ? 0 : 2);
        const theta = (quadrant * Math.PI) / 4;
        const dx = (x % half) - half / 2;
        const dy = (y % half) - half / 2;
        const along = dx * Math.cos(theta) + dy * Math.sin(theta);
        const envelope = Math.exp(-(dx * dx + dy * dy) / (2 * 13 * 13));
        const value = 22 + 105 * (1 + envelope * Math.cos((2 * Math.PI * along) / 20));
        const offset = 4 * (y * SIZE + x);
        picture.data[offset] = value * 0.55;
        picture.data[offset + 1] = value * 0.62;
        picture.data[offset + 2] = value * 0.68;
        picture.data[offset + 3] = 255;
      }
    }
    context.putImageData(picture, 0, 0);
  }, []);
  return <canvas ref={canvas} width={SIZE} height={SIZE} />;
}

/** Firing rate against response: flat at zero, then a straight line through rest. */
function RateCurve() {
  return (
    <svg viewBox="0 0 100 100" aria-hidden="true">
      <line x1="12" y1="88" x2="92" y2="88" stroke="#5f7482" strokeWidth="1" />
      <line x1="12" y1="12" x2="12" y2="88" stroke="#5f7482" strokeWidth="1" />
      <line x1="12" y1="52" x2="92" y2="52" stroke="#5f7482" strokeWidth="1" strokeDasharray="3 3" />
      <polyline points="12,88 30,88 88,18" fill="none" stroke="#e9edee" strokeWidth="3" strokeLinejoin="round" />
      <circle cx="59.8" cy="52" r="4" fill="#e9edee" />
    </svg>
  );
}

function SpikeBars({ counts }: { counts: number[] }) {
  const top = Math.max(...counts, 1);
  const width = 84 / counts.length;
  return (
    <svg viewBox="0 0 100 100" aria-hidden="true">
      {counts.map((count, index) => (
        <rect key={index} x={8 + index * width} y={90 - (count / top) * 78} width={Math.max(width - 0.6, 0.6)} height={(count / top) * 78} fill="#e9edee" />
      ))}
    </svg>
  );
}

type Props = {
  name: string;
  image: string | undefined | null;
  mosaic: MosaicEvent | null;
  result: RunSummary | null;
  reached: boolean;
};

export function StageThumb({ name, image, mosaic, result, reached }: Props) {
  if (image) return <img src={image} alt="" />;
  if (!reached) return null;
  if (name === "mosaic" && mosaic) return <Dots mosaic={mosaic} responses={null} />;
  if (name === "center_surround" && mosaic) return <Dots mosaic={mosaic} responses={result?.responses ?? null} />;
  if (name === "gabor") return <Edges />;
  if (name === "rate") return <RateCurve />;
  if (name === "spikes" && result) return <SpikeBars counts={result.spike_histogram.counts} />;
  return <span aria-hidden="true">•••</span>;
}
