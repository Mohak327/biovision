// The whole route of seeing in 3D for the chosen species: the model, the
// ordered stops beside it, and what happens at the selected one. The list and
// the descriptions carry the content; the 3D view illustrates it.
import { useReducedMotion } from "motion/react";
import { Suspense, lazy, useState } from "react";
import { routeFor, stopStatus } from "../pathway";

const DEFAULT_STOP = "receptors"; // every species has it, and it is in the model

/** A link can open the model on one stop: add ?stop=chiasm (or any stop's id) to the address. */
function stopInAddress(): string | null {
  return new URLSearchParams(window.location.search).get("stop");
}

// The scene and its model loader are loaded only when this section is on the page.
const PathwayScene = lazy(() => import("./PathwayScene"));

export function Pathway({ species }: { species: string }) {
  const [linked] = useState(stopInAddress);
  const [selected, setSelected] = useState(linked ?? DEFAULT_STOP);
  const still = useReducedMotion();
  const route = routeFor(species);
  if (!route) return null;
  // A stop chosen for one species may not exist in the next one.
  const stop = route.stops.find((one) => one.id === selected)
    ?? route.stops.find((one) => one.id === DEFAULT_STOP)!;
  return (
    <section className="pathway-section" aria-labelledby="pathway-title">
      <h2 id="pathway-title">The whole route, from the light to the brain</h2>
      <p>
        Follow the signal through the {species}'s own anatomy. Stops marked "in the model" are the
        ones biovision computes. Drag to turn, scroll to zoom, or pick a stop.
      </p>
      <div className="pathway">
        <div className="pathway-scene" role="img" aria-label={route.label}>
          <Suspense fallback={<p className="pathway-wait">Loading the anatomy.</p>}>
            <PathwayScene
              key={species}
              route={route}
              stop={stop}
              moving={!still}
              flyOnOpen={route.stops.some((one) => one.id === linked)}
            />
          </Suspense>
        </div>
        <div className="pathway-side">
          <ol className="pathway-stops">
            {route.stops.map((one) => {
              const open = one.id === stop.id;
              return (
                <li key={one.id}>
                  <button
                    type="button"
                    className="stop"
                    aria-expanded={open}
                    aria-controls={open ? "pathway-detail" : undefined}
                    onClick={() => setSelected(one.id)}
                  >
                    <span>{one.name}</span>
                    {one.stages.length > 0 && <span className="stop-mark">in the model</span>}
                  </button>
                  {/* The description opens under its own stop. */}
                  {open && (
                    <div className="stop-detail" id="pathway-detail" aria-live="polite">
                      <p>{one.what}</p>
                      <p className="pathway-status">{stopStatus(one)}</p>
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        </div>
      </div>
      <p className="pathway-credit">{route.credit}</p>
    </section>
  );
}
