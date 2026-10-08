// The whole route of seeing for each species: every stop from the light to the
// brain, where it sits in that species' 3D model, and which stages of the
// biovision pipeline (if any) stand for it. Positions come from
// tools/build_anatomy.py, which also builds the models.
import anatomy from "./data/anatomy.json";
import { stageLabel } from "./presets";

export type Point = [number, number, number];

/** The pipeline's stage names, as the Python library spells them. */
export const PIPELINE_STAGES = [
  "color", "optics", "mosaic", "photons", "rods", "center_surround", "gabor", "rate", "spikes",
] as const;
export type PipelineStage = (typeof PIPELINE_STAGES)[number];

export interface Stop {
  id: string;
  name: string;
  /** What happens there, in a sentence or three. */
  what: string;
  /** The parts of the model that light up at this stop. */
  parts: string[];
  /** The pipeline stages that stand for this stop; empty if biovision does not model it. */
  stages: PipelineStage[];
  /** Where the stop sits in the model and how big it is there (scene units). */
  position: Point;
  size: number;
}

/** How one named part of a model is drawn. A shell is the see-through rest of the brain. */
export interface Look { tone: string; opacity: number; shell?: boolean }

export interface Route {
  model: string;
  /** What the picture shows, for someone who cannot see it. */
  label: string;
  /** Where the anatomy came from, and which parts are drawings. */
  credit: string;
  stops: Stop[];
  looks: Record<string, Look>;
  /** The lines the signal travels along. */
  lines: Point[][];
  camera: Point;
}

type Described = Omit<Stop, "position" | "size" | "parts" | "stages"> & { parts?: string[]; stages?: PipelineStage[] };

// Tones for tissue on the dark well. Receptor colours are kept for receptors.
const SHELL: Look = { tone: "#6f8492", opacity: 0.09, shell: true };
const NUCLEUS: Look = { tone: "#7f93a1", opacity: 1 };
const FIBRE: Look = { tone: "#d9e1e5", opacity: 1 };
const CORTEX: Look = { tone: "#a9b8c1", opacity: 0.7 };
const EYEBALL: Look = { tone: "#e6ecef", opacity: 0.22 };
const RETINA: Look = { tone: "#93a6b2", opacity: 0.55 };
const GLASS: Look = { tone: "#f6f8f8", opacity: 0.45 };

const HUMAN: Described[] = [
  {
    id: "light", name: "Light",
    what: "Light from the scene arrives at the eye. Nothing about it is a nerve signal yet. biovision starts from a stored picture: three numbers for each pixel, not the full spectrum of real light.",
  },
  {
    id: "optics", name: "Cornea and lens", parts: ["cornea", "lens", "iris"], stages: ["optics"],
    what: "The cornea bends the light most and the lens adjusts the focus, throwing an image onto the back of the eye. The image is slightly blurred, and blue light more than the rest, because the eye cannot focus every wavelength at once.",
  },
  {
    id: "receptors", name: "Rods and cones", parts: ["retina"], stages: ["color", "mosaic", "photons"],
    what: "The retina lines the back of the eye. About 4.6 million cones of three types (long, middle and short wavelengths) are packed most tightly at the centre of gaze, among about 92 million rods that work in dim light. Each absorbs light only where it sits, so the image becomes a mosaic of samples. The cells are far too small to see at this scale. The photon-noise stage runs only when a light level is set.",
  },
  {
    id: "bipolar", name: "Bipolar and horizontal cells", parts: ["retina"], stages: ["rods"],
    what: "Inside the retina, bipolar cells carry each receptor's signal on, some answering to more light (ON) and some to less (OFF), while horizontal cells spread signals sideways to build the surround the next cells compare against. The rods' signals join the cones' pathway here. biovision has no cells of its own for this layer: it has a rods stage, used only when a light level is set, and the surround is part of the retina stage.",
  },
  {
    id: "ganglion", name: "Retinal ganglion cells", parts: ["retina"], stages: ["center_surround"],
    what: "Roughly a million ganglion cells are the eye's only output. Each compares a small centre with its surroundings. Together they recode the three cone signals as brightness, red against green and blue against yellow, in ON and OFF cells. This is biovision's retina stage.",
  },
  {
    id: "nerve", name: "Optic nerve", parts: ["optic_nerve"],
    what: "The ganglion cells' fibres leave the eye together as the optic nerve. Where they leave there are no receptors, which is the blind spot. biovision passes the retina's signals on unchanged.",
  },
  {
    id: "chiasm", name: "Optic chiasm", parts: ["chiasm"],
    what: "The two nerves meet. Fibres from the half of each retina nearer the nose cross to the other side, a little over half of them, so each side of the brain receives the opposite half of the view from both eyes. The line here follows fibres that stay on their own side.",
  },
  {
    id: "tract", name: "Optic tract", parts: ["optic_tract"],
    what: "Behind the chiasm each tract carries one half of the view, from both eyes. Most of its fibres go to the lateral geniculate nucleus. The rest go to the midbrain, where they help steer the eyes, set the pupil and keep the daily clock.",
  },
  {
    id: "geniculate", name: "Lateral geniculate nucleus", parts: ["geniculate"],
    what: "A relay in the thalamus, in six layers, each fed by one eye. Its cells keep the retina's centre and surround. The brightness, red-green and blue-yellow channels biovision uses were measured here, in the macaque. biovision has no separate stage for it.",
  },
  {
    id: "radiation", name: "Optic radiation",
    what: "A fan of fibres from the geniculate nucleus to the back of the brain; part of it loops forward through the temporal lobe on the way. The atlas has no part for it, so nothing lights up, and the line shows only the direction, not the real loop.",
  },
  {
    id: "cortex", name: "Primary visual cortex (V1)", parts: ["visual_cortex"], stages: ["gabor", "rate", "spikes"],
    what: "V1 lies at the very back of the brain, mostly on the inner face of the occipital lobe. Each of its simple cells answers to an edge at one place, of one angle and size. biovision's cortex stage is a bank of such cells, and the spikes it counts are theirs. The atlas gives the occipital lobe as one piece, so the whole lobe is lit; V1 is part of it.",
  },
  {
    id: "beyond", name: "Beyond V1",
    what: "From V1 the signals go on to V2, V4 and dozens of other areas, broadly in two streams: one down into the temporal lobe for recognising things, one up into the parietal lobe for where things are and for acting on them. biovision stops at V1 and rebuilds the picture from its spikes.",
  },
];

const MOUSE: Described[] = [
  {
    id: "light", name: "Light",
    what: "Light from the scene arrives at the eye. A mouse sees ultraviolet, which an ordinary picture does not record, so biovision estimates it from the picture's blue.",
  },
  {
    id: "optics", name: "Cornea and lens", parts: ["cornea", "lens"], stages: ["optics"],
    what: "A mouse's eye is about 3 mm across and a nearly round lens fills most of it. The eyes face sideways, so together they see most of the way round, with a strip in front that both cover. The image is coarse: a mouse resolves about half a cycle per degree, around a hundred times less than a person.",
  },
  {
    id: "receptors", name: "Rods and cones", parts: ["retina"], stages: ["color", "mosaic", "photons"],
    what: "About 97% of a mouse's receptors are rods. Its cones have two pigments, one for ultraviolet and one for green, and almost no response to red. biovision models the cone pathway in daylight only and gives each cone one pigment; in the real retina many cones hold both. The photon-noise stage runs only when a light level is set.",
  },
  {
    id: "bipolar", name: "Bipolar and horizontal cells", parts: ["retina"],
    what: "As in other mammals, bipolar cells carry the receptors' signals on in ON and OFF types and horizontal cells build the surround. biovision has no cells of its own for this layer; the surround is part of the retina stage.",
  },
  {
    id: "ganglion", name: "Retinal ganglion cells", parts: ["retina"], stages: ["center_surround"],
    what: "The eye's output cells, of more than forty types in the mouse. Each compares a centre several degrees wide with its surroundings. biovision's retina stage has one kind for each cone pigment.",
  },
  {
    id: "nerve", name: "Optic nerve", parts: ["optic_nerve"],
    what: "The ganglion cells' fibres leave the eye as the optic nerve. The atlas is of a brain taken out of the skull, so the eyes and this stretch of nerve are drawn here, not scanned.",
  },
  {
    id: "chiasm", name: "Optic chiasm", parts: ["chiasm"],
    what: "Because a mouse's eyes face sideways, all but a few percent of the fibres cross here to the other side of the brain. In a person about half cross.",
  },
  {
    id: "tract", name: "Optic tract", parts: ["optic_tract", "brachium"],
    what: "The crossed fibres run up the side of the brain. They pass the geniculate nucleus and continue over it to the superior colliculus.",
  },
  {
    id: "colliculus", name: "Superior colliculus", parts: ["colliculus"],
    what: "The main target of a mouse's eye: most of its ganglion cells send their fibre here, to the roof of the midbrain. It holds a map of the view and turns the eyes, head and body toward or away from what appears. biovision does not model it.",
  },
  {
    id: "geniculate", name: "Lateral geniculate nucleus", parts: ["geniculate"],
    what: "The relay to the cortex, in the thalamus. A smaller share of the retina's output takes this path than in a person. biovision has no separate stage for it.",
  },
  {
    id: "cortex", name: "Primary visual cortex (V1)", parts: ["visual_cortex"], stages: ["gabor", "rate", "spikes"],
    what: "A patch a few millimetres across at the back of the cortex. Its cells answer to edges at one angle, as a person's do, but prefer coarse patterns of about 0.04 cycles per degree. biovision's cortex stage is a bank of such cells, and the spikes it counts are theirs.",
  },
  {
    id: "beyond", name: "Beyond V1",
    what: "Around V1 lie about ten smaller visual areas, and the colliculus has its own routes to the rest of the brain. biovision stops at V1 and rebuilds the picture from its spikes.",
  },
];

const FLY: Described[] = [
  {
    id: "light", name: "Light",
    what: "Light from the scene arrives at the eye. A fly sees ultraviolet, which an ordinary picture does not record, so biovision estimates it from the picture's blue.",
  },
  {
    id: "optics", name: "Facets and lenses", parts: ["eye"], stages: ["optics"],
    what: "Each eye is a dome of about 750 facets. Every facet has its own small lens and takes in a patch of the world about five degrees wide, so the whole eye forms a picture of about 750 points. The dome here is drawn, facet by facet; brain atlases do not include the eye.",
  },
  {
    id: "receptors", name: "Photoreceptors R1 to R8", parts: ["eye"], stages: ["color", "mosaic", "photons"],
    what: "Under each lens sit eight receptor cells. R1 to R6 respond to a broad band of light and serve contrast and motion; R7 and R8, stacked in the middle, come in ultraviolet, blue and green kinds and serve colour. Flies see almost no red. biovision gives every facet one ultraviolet, one blue and one green receptor. The photon-noise stage runs only when a light level is set.",
  },
  {
    id: "lamina", name: "Lamina", parts: ["lamina"], stages: ["center_surround", "rate", "spikes"],
    what: "The first layer of the brain, just under the eye, with one unit for each point in the view. Its cells subtract what the neighbouring points see, which sharpens the picture. This is biovision's retina stage for the fly, and the model's output is taken here. Real lamina cells signal with smooth changes of voltage, not spikes; the model counts spikes all the same. The lamina is torn away when a brain is prepared for an atlas, so it is drawn here.",
  },
  {
    id: "medulla", name: "Medulla", parts: ["medulla"],
    what: "The largest part of the optic lobe, with one column for each facet. R7 and R8 end here, brightening and darkening are split into separate channels, and the comparison of colours and the detection of motion begin. biovision does not model it.",
  },
  {
    id: "lobula", name: "Lobula and lobula plate", parts: ["lobula", "lobula_plate"],
    what: "The lobula plate works out which way things are moving, in four layers, one for each direction; the lobula picks out features such as a small moving object or something looming. biovision works on still pictures and has no motion pathway.",
  },
  {
    id: "brain", name: "Central brain", parts: ["brain"],
    what: "The optic lobe's outputs end in small knots of tissue at the side of the central brain, and from there reach the circuits that steer walking and flight. biovision does not model any of it.",
  },
];

const BUILT = anatomy as unknown as Record<string, {
  stops: Record<string, { position: number[]; size: number }>;
  routes: number[][][];
  camera: number[];
}>;

function route(name: string, described: Described[], rest: Omit<Route, "model" | "stops" | "lines" | "camera">): Route {
  const built = BUILT[name];
  return {
    ...rest,
    model: `/models/${name}.glb`,
    stops: described.map((stop) => ({
      parts: [], stages: [], ...stop,
      position: built.stops[stop.id].position as Point,
      size: built.stops[stop.id].size,
    })),
    lines: built.routes as Point[][],
    camera: built.camera as Point,
  };
}

export const ROUTES: Record<string, Route> = {
  human: route("human", HUMAN, {
    label: "Three-dimensional model of the human eyes and brain, showing the route from the eye along the optic nerve to the visual cortex at the back",
    credit: "Human anatomy: BodyParts3D 4.0, © The Database Center for Life Science, licensed CC BY-SA 2.1 Japan. Every part is from that atlas. The skull, the eye muscles and the optic radiation are left out.",
    looks: {
      brain: SHELL, visual_cortex: CORTEX, geniculate: NUCLEUS, optic_tract: FIBRE, chiasm: FIBRE,
      optic_nerve: FIBRE, eyeball: EYEBALL, retina: RETINA, cornea: GLASS, lens: GLASS,
      iris: { tone: "#5d6f7b", opacity: 1 },
    },
  }),
  mouse: route("mouse", MOUSE, {
    label: "Three-dimensional model of the mouse brain with drawn eyes, showing the route from the eye across the optic chiasm to the superior colliculus and the visual cortex",
    credit: "Mouse brain: Allen Mouse Brain Common Coordinate Framework, version 3 (Wang and others, Cell, 2020), © Allen Institute for Brain Science, used under its terms for noncommercial use. The eyes and the nerves from them to the chiasm are drawn, not scanned, and are placed by eye.",
    looks: {
      brain: SHELL, visual_cortex: CORTEX, geniculate: NUCLEUS, colliculus: NUCLEUS, optic_tract: FIBRE,
      brachium: FIBRE, chiasm: FIBRE, optic_nerve: FIBRE, eyeball: EYEBALL, retina: RETINA, cornea: GLASS,
      lens: GLASS,
    },
  }),
  fly: route("fly", FLY, {
    label: "Three-dimensional model of the fruit fly brain with drawn compound eyes, showing the route from the facets through the lamina, medulla and lobula to the central brain",
    credit: "Fly brain: the neuropil regions of Ito and others (Neuron, 2014) on the JFRC2 template brain, from Virtual Fly Brain, licensed CC BY 4.0. The compound eyes and the lamina are drawn, not scanned: a schematic from published dimensions.",
    looks: {
      brain: { tone: "#6f8492", opacity: 0.3, shell: true }, medulla: { tone: "#a9b8c1", opacity: 0.45 }, lobula: NUCLEUS, lobula_plate: { tone: "#bcc8ce", opacity: 1 },
      lamina: { tone: "#d9e1e5", opacity: 0.6 }, eye: { tone: "#93a6b2", opacity: 0.4 },
    },
  }),
};

export const routeFor = (species: string): Route | null => ROUTES[species] ?? null;

/** Whether biovision computes a stop, in the words of the page's stage tiles. */
export function stopStatus(stop: Stop): string {
  if (!stop.stages.length) return "Not in the model.";
  const names = stop.stages.map(stageLabel);
  const list = names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}` : names[0];
  return `In the model: the ${list} ${names.length > 1 ? "stages" : "stage"}.`;
}

const round = (value: number) => Math.round(value * 1e6) / 1e6;

/** The point a fraction `t` of the way along a line of points, at even speed. `t` wraps past 1. */
export function pointOnPath(points: Point[], t: number): Point {
  const lengths = points.slice(1).map((point, i) => Math.hypot(
    point[0] - points[i][0], point[1] - points[i][1], point[2] - points[i][2]));
  const total = lengths.reduce((sum, length) => sum + length, 0);
  let remaining = (t === 1 ? 1 : ((t % 1) + 1) % 1) * total;
  for (let i = 0; i < lengths.length; i += 1) {
    if (remaining <= lengths[i] || i === lengths.length - 1) {
      const f = lengths[i] === 0 ? 0 : Math.min(remaining / lengths[i], 1);
      const a = points[i];
      const b = points[i + 1];
      return [round(a[0] + (b[0] - a[0]) * f), round(a[1] + (b[1] - a[1]) * f), round(a[2] + (b[2] - a[2]) * f)];
    }
    remaining -= lengths[i];
  }
  return points[points.length - 1];
}
