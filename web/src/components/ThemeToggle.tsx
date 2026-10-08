// The theme switch: a small sun that turns into a moon. Its rays draw in as the
// page goes dark, and the light then circles the moon so it runs through its phases.
import { Canvas, useFrame } from "@react-three/fiber";
import { useReducedMotion } from "motion/react";
import { useRef, useState } from "react";
import * as THREE from "three";
import type { Theme } from "../theme";

const SUN = new THREE.Color("#f2b632");
const MOON = new THREE.Color("#dfe6ea");
const RAYS = 8;
const FRONT = new THREE.Vector3(0, 0, 3); // lights the whole face: a sun
const SIDE = new THREE.Vector3(-2.7, 0.9, 0.5); // lights one edge: a crescent
const AXIS = new THREE.Vector3(0, 1, 0); // the moon's light circles this, giving the phases
const IDLE_SPEED = 0.3; // radians a second, for the sun's rays and the moon's phases alike
const LIVELY_SPEED = 1.6; // the same, under the pointer
const SETTLE = 7; // how quickly the change settles; higher is quicker

type BodyProps = { dark: boolean; still: boolean; lively: boolean };

function Body({ dark, still, lively }: BodyProps) {
  const ball = useRef<THREE.Mesh>(null);
  const rays = useRef<THREE.Group>(null);
  const light = useRef<THREE.DirectionalLight>(null);
  const night = useRef(dark ? 1 : 0); // 0 is the sun, 1 the moon
  const phase = useRef(0); // how far round the moon its light has gone
  const moonLight = useRef(new THREE.Vector3());
  useFrame((_, delta) => {
    const goal = dark ? 1 : 0;
    night.current = still ? goal : THREE.MathUtils.damp(night.current, goal, SETTLE, delta);
    const t = night.current;
    const turn = still ? 0 : delta * (lively ? LIVELY_SPEED : IDLE_SPEED);
    if (ball.current) {
      const material = ball.current.material as THREE.MeshStandardMaterial;
      material.color.copy(SUN).lerp(MOON, t);
      material.emissive.copy(SUN);
      material.emissiveIntensity = 0.85 * (1 - t); // the sun glows; the moon only reflects
      ball.current.rotation.y = t * Math.PI;
      ball.current.scale.setScalar(1 + 0.18 * t); // the moon fills the space the rays leave
    }
    if (rays.current) {
      rays.current.scale.setScalar(Math.max(1 - t, 0.0001));
      rays.current.rotation.z += turn;
    }
    if (dark) phase.current += turn;
    moonLight.current.copy(SIDE).applyAxisAngle(AXIS, phase.current);
    light.current?.position.copy(FRONT).lerp(moonLight.current, t);
  });
  return (
    <>
      <ambientLight intensity={0.12} />
      <directionalLight ref={light} intensity={2.6} />
      <mesh ref={ball}>
        <sphereGeometry args={[1, 32, 24]} />
        <meshStandardMaterial roughness={0.9} />
      </mesh>
      <group ref={rays}>
        {Array.from({ length: RAYS }, (_, index) => {
          const angle = (2 * Math.PI * index) / RAYS;
          return (
            <mesh key={index} position={[1.62 * Math.cos(angle), 1.62 * Math.sin(angle), 0]} rotation={[0, 0, angle]}>
              <boxGeometry args={[0.42, 0.16, 0.16]} />
              <meshBasicMaterial color={SUN} />
            </mesh>
          );
        })}
      </group>
    </>
  );
}

type Props = { theme: Theme; onToggle: () => void };

export function ThemeToggle({ theme, onToggle }: Props) {
  const still = useReducedMotion() ?? false;
  const [lively, setLively] = useState(false);
  const dark = theme === "dark";
  const label = dark ? "Switch to the light theme" : "Switch to the dark theme";
  return (
    <button
      type="button"
      className="theme-toggle"
      aria-label={label}
      title={label}
      onClick={onToggle}
      onPointerEnter={() => setLively(true)}
      onPointerLeave={() => setLively(false)}
    >
      <Canvas orthographic camera={{ zoom: 9, position: [0, 0, 5] }} dpr={[1, 2]} gl={{ alpha: true }} aria-hidden="true">
        <Body dark={dark} still={still} lively={lively} />
      </Canvas>
    </button>
  );
}
