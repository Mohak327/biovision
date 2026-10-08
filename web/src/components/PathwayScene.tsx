// One species' anatomy in 3D: every named part of its model, the selected
// stop's parts lit, and the signal's route with small lights travelling along
// it. The camera flies to the selected stop and then leaves the visitor free.
import { Line, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { type ComponentRef, Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { type Point, type Route, type Stop, pointOnPath } from "../pathway";

const LIT = "#a99bff"; // the page's accent on a dark ground
const SIGNAL = "#ffffff";
const LIGHTS_PER_UNIT = 0.45; // travelling lights for each scene unit of route
const SETTLE_FRAMES = 150; // how long the camera keeps moving toward a newly chosen stop

type Props = {
  route: Route;
  stop: Stop;
  moving: boolean;
  /** Fly to the stop straight away instead of opening on the whole view. */
  flyOnOpen: boolean;
};

function Anatomy({ route, stop }: { route: Route; stop: Stop }) {
  const { scene } = useGLTF(route.model);
  const parts = useMemo(() => {
    const found: { name: string; geometry: THREE.BufferGeometry }[] = [];
    scene.updateMatrixWorld(true);
    scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh) return;
      const name = route.looks[mesh.name] ? mesh.name : mesh.parent?.name ?? "";
      if (!route.looks[name]) return;
      const geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
      geometry.computeVertexNormals();
      found.push({ name, geometry });
    });
    return found;
  }, [scene, route]);
  return (
    <>
      {parts.map(({ name, geometry }) => {
        const look = route.looks[name];
        const lit = stop.parts.includes(name);
        const opacity = !lit ? look.opacity : look.shell ? 0.4 : Math.max(look.opacity, 0.92);
        return (
          <mesh key={name} geometry={geometry} renderOrder={opacity < 1 ? 2 : 1}>
            <meshStandardMaterial
              color={lit ? LIT : look.tone}
              roughness={0.75}
              transparent={opacity < 1}
              opacity={opacity}
              depthWrite={opacity >= 0.5}
              side={THREE.DoubleSide}
              polygonOffset={lit}
              polygonOffsetFactor={-2}
              polygonOffsetUnits={-2}
            />
          </mesh>
        );
      })}
    </>
  );
}

function length(line: Point[]): number {
  return line.slice(1).reduce((sum, point, i) => sum + Math.hypot(
    point[0] - line[i][0], point[1] - line[i][1], point[2] - line[i][2]), 0);
}

/** A thin line marking one stretch of the route, and small lights that travel it. */
function Signal({ line, moving }: { line: Point[]; moving: boolean }) {
  const lights = useRef<THREE.Group>(null);
  const span = length(line);
  const count = Math.max(1, Math.round(span * LIGHTS_PER_UNIT));
  useFrame(({ clock }) => {
    lights.current?.children.forEach((light, i) => {
      const travelled = moving ? (clock.elapsedTime * 0.9) / span : 0;
      light.position.set(...pointOnPath(line, travelled + i / count));
    });
  });
  return (
    <>
      <Line points={line} color={SIGNAL} lineWidth={1.5} transparent opacity={0.55} depthTest={false} />
      <group ref={lights}>
        {Array.from({ length: count }, (_, i) => (
          <mesh key={i} renderOrder={3}>
            <sphereGeometry args={[0.055, 14, 10]} />
            <meshBasicMaterial color={SIGNAL} depthTest={false} />
          </mesh>
        ))}
      </group>
    </>
  );
}

/** Turns the camera toward the selected stop and moves in to fit it; then the visitor is free again. */
function Focus({ stop, flyOnOpen }: { stop: Stop; flyOnOpen: boolean }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const camera = useThree((state) => state.camera);
  const settling = useRef(flyOnOpen ? SETTLE_FRAMES : 0);
  const shown = useRef(stop.id);
  const goal = useMemo(() => new THREE.Vector3(), []);
  const offset = useMemo(() => new THREE.Vector3(), []);
  useEffect(() => {
    if (shown.current !== stop.id) settling.current = SETTLE_FRAMES;
    shown.current = stop.id;
  }, [stop.id]);
  useFrame(() => {
    if (!controls.current || settling.current <= 0) return;
    settling.current -= 1;
    goal.set(...stop.position);
    const target = controls.current.target;
    offset.copy(camera.position).sub(target);
    const distance = THREE.MathUtils.lerp(offset.length(), Math.min(Math.max(stop.size * 2.6, 2.2), 13), 0.05);
    target.lerp(goal, 0.05);
    camera.position.copy(target).add(offset.setLength(distance));
    controls.current.update();
  });
  return (
    <OrbitControls
      ref={controls}
      enablePan={false}
      minDistance={1.2}
      maxDistance={24}
      onStart={() => { settling.current = 0; }}
    />
  );
}

export default function PathwayScene({ route, stop, moving, flyOnOpen }: Props) {
  return (
    <Canvas camera={{ position: route.camera, fov: 36 }} dpr={[1, 2]}>
      <ambientLight intensity={1.4} />
      <directionalLight position={[-4, 6, 10]} intensity={1.7} />
      <directionalLight position={[6, -2, 4]} intensity={0.5} />
      <Suspense fallback={null}>
        <Anatomy route={route} stop={stop} />
      </Suspense>
      {route.lines.map((line, index) => <Signal key={index} line={line} moving={moving} />)}
      <Focus stop={stop} flyOnOpen={flyOnOpen} />
    </Canvas>
  );
}
