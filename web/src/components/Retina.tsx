// The receptor mosaic as a 3D retina: one point per receptor, coloured by its
// type and lit by how strongly it responded. Drag to orbit.
import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { type MosaicEvent, decodeFloat32, decodeUint8 } from "../api";
import { RECEPTOR_COLORS, RECEPTOR_NAMES, capitalised } from "../presets";

const CURVE = 1.7; // radius of the bowl the receptors sit on; larger is flatter
const REST = 0.45; // brightness of a receptor that did not respond

type Props = { mosaic: MosaicEvent; responses: string | null };

function buildGeometry(mosaic: MosaicEvent, responses: string | null) {
  const xy = decodeFloat32(mosaic.positions);
  const types = decodeUint8(mosaic.types);
  const levels = responses ? decodeFloat32(responses) : null;
  const count = mosaic.count;
  const kinds = mosaic.receptors.length;
  // Receptors of different types can share a position; nudge each type apart
  // by a fraction of the typical spacing so every colour stays visible.
  const nudge = 0.55 / Math.sqrt(Math.max(count / kinds, 1));
  const colours = mosaic.receptors.map((name) => new THREE.Color(RECEPTOR_COLORS[name] ?? "#ffffff"));
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const angle = (2 * Math.PI * types[i]) / kinds;
    const u = (xy[2 * i] - 0.5) * 2 + nudge * Math.cos(angle);
    const v = (0.5 - xy[2 * i + 1]) * 2 + nudge * Math.sin(angle);
    positions[3 * i] = u;
    positions[3 * i + 1] = v;
    positions[3 * i + 2] = CURVE - Math.sqrt(Math.max(CURVE * CURVE - u * u - v * v, 0));
    // The square root lifts mid responses, which is where most receptors sit.
    const level = levels && levels.length === count ? REST + (1 - REST) * Math.sqrt(levels[i]) : 0.7;
    const colour = colours[types[i]];
    colors[3 * i] = colour.r * level;
    colors[3 * i + 1] = colour.g * level;
    colors[3 * i + 2] = colour.b * level;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

export function Retina({ mosaic, responses }: Props) {
  const geometry = useMemo(() => buildGeometry(mosaic, responses), [mosaic, responses]);
  const kinds = mosaic.receptors.length;
  const pointSize = Math.min(0.1, Math.max(0.024, 1.7 / Math.sqrt(mosaic.count / kinds)));
  const still = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  return (
    <div className="retina">
      <div className="retina-view" role="img" aria-label={`${mosaic.count} receptors arranged on the retina, coloured by type and lit by their response`}>
        <Canvas flat camera={{ position: [0, -1.5, 3.6], fov: 40 }} dpr={[1, 2]}>
          <points geometry={geometry}>
            <pointsMaterial vertexColors size={pointSize} sizeAttenuation toneMapped={false} />
          </points>
          <OrbitControls
            enablePan={false}
            enableZoom={false}
            minPolarAngle={Math.PI * 0.25}
            maxPolarAngle={Math.PI * 0.85}
            autoRotate={!still}
            autoRotateSpeed={0.6}
          />
        </Canvas>
      </div>
      <ul className="retina-legend">
        {mosaic.receptors.map((name) => (
          <li key={name}>
            <span className="swatch" style={{ background: RECEPTOR_COLORS[name] }} />
            {capitalised(RECEPTOR_NAMES[name] ?? name)}
          </li>
        ))}
        <li className="retina-note">Brighter means a stronger response. Drag to turn it.</li>
      </ul>
    </div>
  );
}
