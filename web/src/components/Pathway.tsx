// The whole route of seeing in 3D for the chosen species: the model, the
// ordered stops beside it, and what happens at the selected one. The list and
// the descriptions carry the content; the 3D view illustrates it.
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Suspense, lazy, useState } from "react";
import { routeFor, stopStatus } from "../pathway";

const DEFAULT_STOP = "receptors"; // every species has it, and it is in the model
const DRAWER = { duration: 0.26, ease: [0.2, 0, 0, 1] as const }; // quick out, soft landing
const CLOSED = { height: 0, opacity: 0 };
const OPEN = { height: "auto", opacity: 1 };

/** A link can open the model on one stop: add ?stop=chiasm (or any stop's id) to the address. */
function stopInAddress(): string | null {
  return new URLSearchParams(window.location.search).get("stop");
}

// The scene and its model loader are loaded only when this section is on the page.
const PathwayScene = lazy(() => import("./PathwayScene"));

export function Pathway({ species }: { species: string }) {
  const [linked] = useState(stopInAddress);
  const [selected, setSelected] = useState(linked ?? DEFAULT_STOP);
  // Which descriptions are open. Any number can be; the selected stop starts open.
  const [opened, setOpened] = useState<ReadonlySet<string>>(() => new Set([linked ?? DEFAULT_STOP]));
  const still = useReducedMotion();
  const route = routeFor(species);
  if (!route) return null;
  // A stop chosen for one species may not exist in the next one.
  const stop = route.stops.find((one) => one.id === selected)
    ?? route.stops.find((one) => one.id === DEFAULT_STOP)!;
  // A click shows the stop in the model and opens its description, or shuts it if it was open.
  const pick = (id: string) => {
    setSelected(id);
    setOpened((current) => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  };
  return (
    <section className="pathway-section" aria-labelledby="pathway-title">
      <h2 id="pathway-title">The whole route, from the light to the brain</h2>
      <p>
        Follow the signal through the {species}'s own anatomy. Stops marked "in the model" are the
        ones biovision computes. Drag to turn, scroll to zoom, or pick a stop to read about it.
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
              const open = opened.has(one.id);
              return (
                <li key={one.id}>
                  <button
                    type="button"
                    className="stop"
                    aria-expanded={open}
                    aria-current={one.id === stop.id ? "true" : undefined}
                    aria-controls={open ? `stop-detail-${one.id}` : undefined}
                    onClick={() => pick(one.id)}
                  >
                    <span>{one.name}</span>
                    {one.stages.length > 0 && <span className="stop-mark">in the model</span>}
                    <svg className="stop-chevron" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
                      <path d="M2.5 4.25 6 7.75l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.6"
                            strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                  {/* The description slides open under its own stop, and shut on a second click. */}
                  <AnimatePresence initial={false}>
                    {open && (
                      <motion.div
                        className="stop-drawer"
                        id={`stop-detail-${one.id}`}
                        aria-live="polite"
                        initial={CLOSED}
                        animate={OPEN}
                        exit={CLOSED}
                        transition={still ? { duration: 0 } : DRAWER}
                      >
                        <div className="stop-detail">
                          <p>{one.what}</p>
                          <p className="pathway-status">{stopStatus(one)}</p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
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
