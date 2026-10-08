// The "o" of the page title: the chosen species' eye in 3D, in translucent mauve.
// The three eyes sit on a reel that rolls vertically to the next one when the
// species changes, and the eye in front follows the pointer.
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";
import * as THREE from "three";

const ORDER = ["human", "mouse", "fly"]; // the reel's order; an unknown species shows the first
const STEP = (2 * Math.PI) / ORDER.length;
const REEL = 3; // the reel's radius: wide enough that only the front eye is in view
const VIEW = 2.3; // world units the canvas is tall; the eye is 2
const ROLL = 6; // how quickly the reel settles; higher is quicker
const LOOK = 8; // the same, for following the pointer
const REACH = 400; // pixels the pointer travels for the eye to turn fully
const MAUVE = "#a78bfa";
const PALE = "#ddd3ff";
const DEEP = "#4a3a9e";
const DARK = "#150f2b";

/** Part of a sphere's surface, facing forward: an iris or a pupil of the given angular size. */
function Cap({ radius, angle, color }: { radius: number; angle: number; color: string }) {
  return (
    <mesh rotation={[Math.PI / 2, 0, 0]}>
      <sphereGeometry args={[radius, 32, 12, 0, 2 * Math.PI, 0, angle]} />
      <meshStandardMaterial color={color} roughness={0.35} />
    </mesh>
  );
}

function Glint() {
  return (
    <mesh position={[0.32, 0.38, 0.9]}>
      <sphereGeometry args={[0.11, 12, 8]} />
      <meshBasicMaterial color="#ffffff" />
    </mesh>
  );
}

/** A white of the eye, a coloured iris and a round pupil. */
function HumanEye() {
  return (
    <>
      <mesh>
        <sphereGeometry args={[1, 40, 28]} />
        <meshPhysicalMaterial color={PALE} transparent opacity={0.62} roughness={0.15} clearcoat={1} />
      </mesh>
      <Cap radius={1.01} angle={0.62} color={MAUVE} />
      <Cap radius={1.02} angle={0.27} color={DARK} />
      <Glint />
    </>
  );
}

/** A dark bead: the pupil fills almost all of the eye that shows. */
function MouseEye() {
  return (
    <>
      <mesh>
        <sphereGeometry args={[1, 40, 28]} />
        <meshPhysicalMaterial color={DEEP} transparent opacity={0.75} roughness={0.1} clearcoat={1} />
      </mesh>
      <Cap radius={1.01} angle={1.05} color={DARK} />
      <Glint />
    </>
  );
}

/** A compound eye: flat facets, with the joins between them drawn. */
function FlyEye() {
  return (
    <>
      <mesh>
        <icosahedronGeometry args={[1, 3]} />
        <meshStandardMaterial color={MAUVE} flatShading transparent opacity={0.82} roughness={0.3} metalness={0.2} />
      </mesh>
      <mesh>
        <icosahedronGeometry args={[1.01, 3]} />
        <meshBasicMaterial color={DEEP} wireframe transparent opacity={0.45} />
      </mesh>
    </>
  );
}

const EYES = [HumanEye, MouseEye, FlyEye];

function Reel({ index, still }: { index: number; still: boolean }) {
  const gaze = useRef<THREE.Group>(null);
  const reel = useRef<THREE.Group>(null);
  const shown = useRef(index);
  const goal = useRef(index * STEP); // the reel only ever rolls forward
  const angle = useRef(goal.current);
  const pointer = useRef({ x: 0, y: 0 }); // where the pointer is from the eye, -1 to 1
  const { camera, gl, size } = useThree();

  useEffect(() => {
    camera.zoom = size.height / VIEW;
    camera.updateProjectionMatrix();
  }, [camera, size]);

  useEffect(() => {
    if (still) return;
    const follow = (event: PointerEvent) => {
      const box = gl.domElement.getBoundingClientRect();
      pointer.current = {
        x: THREE.MathUtils.clamp((event.clientX - box.left - box.width / 2) / REACH, -1, 1),
        y: THREE.MathUtils.clamp((event.clientY - box.top - box.height / 2) / REACH, -1, 1),
      };
    };
    window.addEventListener("pointermove", follow);
    return () => window.removeEventListener("pointermove", follow);
  }, [gl, still]);

  useFrame((_, delta) => {
    if (shown.current !== index) {
      goal.current += ((index - shown.current + ORDER.length) % ORDER.length) * STEP;
      shown.current = index;
    }
    angle.current = still ? goal.current : THREE.MathUtils.damp(angle.current, goal.current, ROLL, delta);
    if (reel.current) reel.current.rotation.x = -angle.current;
    if (gaze.current) {
      gaze.current.rotation.y = THREE.MathUtils.damp(gaze.current.rotation.y, pointer.current.x * 0.6, LOOK, delta);
      gaze.current.rotation.x = THREE.MathUtils.damp(gaze.current.rotation.x, pointer.current.y * 0.45, LOOK, delta);
    }
  });

  return (
    <>
      <ambientLight intensity={0.9} />
      <directionalLight position={[2, 3, 4]} intensity={2} />
      <group ref={gaze}>
        <group ref={reel} position={[0, 0, -REEL]}>
          {EYES.map((Eye, place) => (
            <group key={ORDER[place]} rotation={[place * STEP, 0, 0]}>
              <group position={[0, 0, REEL]}>
                <Eye />
              </group>
            </group>
          ))}
        </group>
      </group>
    </>
  );
}

export function TitleEye({ species }: { species: string }) {
  const still = useReducedMotion() ?? false;
  const index = Math.max(ORDER.indexOf(species), 0);
  return (
    <span className="title-eye" aria-hidden="true">
      <Canvas orthographic camera={{ position: [0, 0, 6] }} dpr={[1, 2]} gl={{ alpha: true }}>
        <Reel index={index} still={still} />
      </Canvas>
    </span>
  );
}
