"""
Build the 3D anatomy the web app shows: one model of the route of seeing for
each species, from open atlases of real anatomy plus a few drawn parts.

    Human:  BodyParts3D 4.0, (c) The Database Center for Life Science,
            licensed CC BY-SA 2.1 Japan. Every part is from the atlas.
    Mouse:  Allen Mouse Brain Common Coordinate Framework v3 (Wang et al.,
            Cell 2020), (c) Allen Institute for Brain Science. Its terms of
            use allow copying, display and derivative works for research and
            other noncommercial purposes, with citation. The atlas is of a
            brain taken out of the skull, so the eyes and the nerves from
            them to the chiasm are drawn here, not scanned.
    Fly:    The neuropil domains of Ito et al. (Neuron 2014) painted on the
            JFRC2 template brain, from Virtual Fly Brain, licensed CC BY 4.0.
            Brain atlases leave out the compound eye and the lamina; both are
            drawn here.

Only the needed files are read from the remote archives, and they are kept in
tools/.cache. Each model is scaled to the same size on screen, so a scene unit
is a different length for each species.

Writes web/public/models/<species>.glb and that species' entry in
web/src/data/anatomy.json. One species at a time (the meshes are large):

    pip install -e ".[anatomy]"
    python tools/build_anatomy.py human
    python tools/build_anatomy.py mouse
    python tools/build_anatomy.py fly
"""
import csv
import gzip
import io
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from remotezip import RemoteZip
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / ".cache"
MODELS = ROOT / "web" / "public" / "models"
ANATOMY = ROOT / "web" / "src" / "data" / "anatomy.json"
BP3D = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/"
ALLEN = ("https://download.alleninstitute.org/informatics-archive/current-release/"
         "mouse_ccf/annotation/ccf_2017/structure_meshes/")
VFB = "https://www.virtualflybrain.org/data/VFB/i/0003/{}/VFB_00017894/volume.nrrd"
RADIUS = 5.0  # every model is scaled so its bounding box has this half-diagonal, in scene units


@dataclass
class Model:
    """One species' anatomy in the atlas's own units and axes."""
    parts: dict                 # part name -> trimesh.Trimesh
    stops: dict                 # stop id -> (point, size); size None means "as big as the part"
    routes: list                # polylines the signal follows, each a list of points
    axes: np.ndarray            # rows: the scene's x (animal's left), y (up), z (front) in atlas axes
    camera: tuple               # direction from the model's centre to the camera, in scene axes
    budget: dict                # part name -> most faces to keep
    unit: str                   # the atlas's unit of length
    drawn: list = field(default_factory=list)   # parts that are drawings, not scans
    facts: dict = field(default_factory=dict)   # numbers the page quotes


# ---------------------------------------------------------------- fetching

def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "biovision build_anatomy"})
    return urllib.request.urlopen(request, timeout=300).read()


def cached(name, get):
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get())
    return path


def bp3d_listing():
    """BodyParts3D concept name -> the element files that make it up."""
    owners = {}
    names = {}
    for listing in ("partof_element_parts.txt", "isa_element_parts.txt"):
        text = cached(listing, lambda: fetch(BP3D + listing)).read_text(encoding="utf-8")
        for row in csv.reader(io.StringIO(text), delimiter="\t"):
            if len(row) >= 3 and row[0].startswith("FMA"):
                owners.setdefault(row[1], set()).add(row[2])
    for concept, elements in owners.items():  # an element's name is its most specific concept
        for element in elements:
            if element not in names or len(elements) < names[element][0]:
                names[element] = (len(elements), concept)
    return owners, {element: name for element, (_, name) in names.items()}


def bp3d_meshes(elements):
    """BodyParts3D element meshes, in millimetres."""
    missing = [e for e in elements if not (CACHE / "bp3d" / f"{e}.obj").exists()]
    for archive in ("partof_BP3D_4.0_obj_99.zip", "isa_BP3D_4.0_obj_99.zip"):
        if not missing:
            break
        with RemoteZip(BP3D + archive) as remote:
            inside = {Path(n).stem: n for n in remote.namelist()}
            for element in list(missing):
                if element in inside:
                    cached(f"bp3d/{element}.obj", lambda: remote.read(inside[element]))
                    missing.remove(element)
    if missing:
        raise SystemExit(f"BodyParts3D has no mesh for {missing}")
    return {e: trimesh.load(CACHE / "bp3d" / f"{e}.obj", force="mesh", process=False) for e in elements}


def allen_mesh(structure_id):
    """One structure of the Allen mouse atlas, both sides, in micrometres."""
    path = cached(f"mouse/{structure_id}.obj", lambda: fetch(f"{ALLEN}{structure_id}.obj"))
    return trimesh.load(path, force="mesh", process=False)


def vfb_surface(domain):
    """The surface of one painted neuropil domain of the JFRC2 fly brain, in micrometres.

    Virtual Fly Brain gives each domain as a volume of voxels; the surface is
    taken from it at half resolution (marching cubes on a smoothed mask).
    """
    raw = cached(f"fly/nrrd/VFB_0003{domain}.nrrd", lambda: fetch(VFB.format(domain))).read_bytes()
    head, _, body = raw.partition(b"\n\n")
    header = dict(line.split(": ", 1) for line in head.decode().splitlines() if ": " in line)
    sizes = [int(n) for n in header["sizes"].split()]
    voxel = float(header["space directions"].split(",")[0].strip("( "))
    volume = np.frombuffer(gzip.decompress(body), dtype=np.uint8).reshape(sizes, order="F")
    step = 2
    coarse = (volume[::step, ::step, ::step] > 0).astype(np.float32)
    coarse = np.pad(coarse, 1)  # close the surface where a domain touches the edge of the volume
    from scipy.ndimage import gaussian_filter
    vertices, faces, _, _ = measure.marching_cubes(gaussian_filter(coarse, 1.0), 0.5)
    return trimesh.Trimesh((vertices - 1) * step * voxel, faces, process=True)


# ---------------------------------------------------------------- geometry

def join(meshes):
    return trimesh.util.concatenate(list(meshes))


def whole_pieces(mesh, share=0.05):
    """A mesh without its stray scraps: only the connected pieces holding a real share of its faces."""
    pieces = mesh.split(only_watertight=False)
    return join(piece for piece in pieces if len(piece.faces) >= share * len(mesh.faces))


def side(mesh, axis, positive):
    """The half of a two-sided mesh on one side of its own middle along an axis."""
    middle = mesh.bounds.mean(axis=0)[axis]
    keep = (mesh.vertices[:, axis] > middle) == positive
    return trimesh.Trimesh(mesh.vertices, mesh.faces[keep[mesh.faces].all(axis=1)], process=True)


def far_end(mesh, away_from, share=0.1):
    """Centre of the part of a mesh farthest from a point."""
    distance = np.linalg.norm(mesh.vertices - away_from, axis=1)
    return mesh.vertices[distance >= np.quantile(distance, 1 - share)].mean(axis=0)


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def frame(axis, up):
    """Three unit vectors at right angles: the axis, and across and up from it."""
    axis = unit(axis)
    across = unit(np.cross(up, axis))
    return axis, across, np.cross(axis, across)


def directions(axis, up, across_deg, up_deg):
    """Unit vectors `across_deg`, `up_deg` away from an axis (angles in degrees, arrays allowed)."""
    axis, across, up = frame(axis, up)
    angle = np.radians(np.hypot(across_deg, up_deg))
    turn = np.arctan2(up_deg, across_deg)
    return (np.cos(angle)[:, None] * axis
            + np.sin(angle)[:, None] * (np.cos(turn)[:, None] * across + np.sin(turn)[:, None] * up))


def dome(centre, axis, up, radius, half_across_deg, half_up_deg, detail=4):
    """The part of a sphere within an elliptical cone of angles round an axis."""
    sphere = trimesh.creation.icosphere(subdivisions=detail, radius=1.0)
    a, across, upward = frame(axis, up)
    d = sphere.vertices
    angle = np.degrees(np.arccos(np.clip(d @ a, -1, 1)))
    turn = np.arctan2(d @ upward, d @ across)
    inside = (angle * np.cos(turn) / half_across_deg) ** 2 + (angle * np.sin(turn) / half_up_deg) ** 2 <= 1
    faces = sphere.faces[inside[sphere.faces].all(axis=1)]
    return trimesh.Trimesh(d * radius + centre, faces, process=True)


def facets(centre, axis, up, radius, spacing_deg, half_across_deg, half_up_deg):
    """A dome of hexagonal facets, `spacing_deg` apart as seen from the centre. Returns mesh, count."""
    rows = int(half_up_deg / (spacing_deg * 0.866)) + 1
    columns = int(half_across_deg / spacing_deg) + 1
    grid = np.array([((column + (row % 2) / 2) * spacing_deg, row * spacing_deg * 0.866)
                     for row in range(-rows, rows + 1) for column in range(-columns - 1, columns + 1)])
    grid = grid[(grid[:, 0] / half_across_deg) ** 2 + (grid[:, 1] / half_up_deg) ** 2 <= 1]
    outward = directions(axis, up, grid[:, 0], grid[:, 1])
    reach = 0.5 * radius * np.radians(spacing_deg)  # corner of a hexagon, with a gap to its neighbours
    vertices, faces = [], []
    for normal in outward:
        _, across, upward = frame(normal, up)
        middle = centre + normal * radius
        start = len(vertices)
        vertices.append(middle + normal * reach * 0.35)  # each lens bulges a little
        vertices.extend(middle + reach * (np.cos(t) * across + np.sin(t) * upward)
                        for t in np.arange(6) * np.pi / 3)
        faces.extend((start, start + 1 + i, start + 1 + (i + 1) % 6) for i in range(6))
    return trimesh.Trimesh(np.array(vertices), np.array(faces), process=False), len(grid)


def tube(points, radius):
    return join(trimesh.creation.cylinder(radius=radius, segment=[a, b], sections=12)
                for a, b in zip(points[:-1], points[1:]))


def extent(mesh):
    return float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))


def spot(mesh):
    """A stop at the middle of a part, as big as the part."""
    return mesh.vertices.mean(axis=0), None


# ---------------------------------------------------------------- species

# BodyParts3D concept names for each part of the human model.
HUMAN_PARTS = {
    "eyeball": ["left sclera", "right sclera"],
    "cornea": ["left cornea", "right cornea"],
    "iris": ["left iris", "right iris"],
    "lens": ["left lens", "right lens"],
    "retina": ["optic part of left retina", "optic part of right retina"],
    "optic_nerve": ["left optic nerve", "right optic nerve"],
    "chiasm": ["optic chiasm"],
    "optic_tract": ["left optic tract", "right optic tract"],
    "geniculate": ["left lateral geniculate body", "right lateral geniculate body"],
    "visual_cortex": ["left occipital lobe", "right occipital lobe"],
}
# The rest of the brain is one shell: the folds of the cortex, cerebellum and brainstem.
HUMAN_SHELL_WORDS = ("gyrus", "lobule", "insula")
HUMAN_SHELL = ["cerebellum", "pons", "medulla oblongata", "midbrain"]
HUMAN_OUTSIDE_BRAIN = ["anterior part of right superior temporal gyrus", "anterior part of left superior temporal gyrus",
                       "posterior part of right superior temporal gyrus", "posterior part of left superior temporal gyrus"]


def human():
    owners, names = bp3d_listing()

    def elements(concepts):
        return sorted({e for concept in concepts for e in owners[concept]})

    named = {part: elements(concepts) for part, concepts in HUMAN_PARTS.items()}
    used = {e for group in named.values() for e in group}
    shell = [e for e in elements(["brain"] + HUMAN_OUTSIDE_BRAIN + HUMAN_SHELL) if e not in used
             and (any(word in names[e] for word in HUMAN_SHELL_WORDS) or e in elements(HUMAN_SHELL))]
    right = {concept: elements([concept]) for concept in (
        "right cornea", "right lens", "optic part of right retina", "right optic nerve",
        "right optic tract", "right lateral geniculate body", "right occipital lobe")}
    meshes = bp3d_meshes(sorted(used | set(shell)))
    for element in named["cornea"]:  # the right cornea's file has a stray triangle by the left eye
        meshes[element] = whole_pieces(meshes[element])
    # Each optic nerve comes twice, once whole and once as only its stretch behind the eye: keep the whole one.
    nerves = [max(elements([concept]), key=lambda e: extent(meshes[e])) for concept in HUMAN_PARTS["optic_nerve"]]
    named["optic_nerve"] = nerves
    right["right optic nerve"] = nerves[1:]

    parts = {part: join(meshes[e] for e in group) for part, group in named.items()}
    parts["brain"] = join(meshes[e] for e in shell)
    one = {concept: join(meshes[e] for e in group) for concept, group in right.items()}

    # The route follows the right eye's fibres that stay on the right side.
    forward = np.array([0.0, -1.0, 0.0])  # BodyParts3D: millimetres, +z up, -y forward, the body's right at -x
    cornea = one["right cornea"].vertices.mean(axis=0)
    front = join([one["right cornea"], one["right lens"]])
    back = far_end(one["optic part of right retina"], cornea)
    geniculate = one["right lateral geniculate body"].vertices.mean(axis=0)
    cortex = one["right occipital lobe"].vertices.mean(axis=0)
    stops = {
        "light": (cornea + 45 * forward, 60.0),
        "optics": (front.vertices.mean(axis=0), 30.0),
        "receptors": (back, 30.0),
        "bipolar": (back, 30.0),
        "ganglion": (back, 30.0),
        "nerve": spot(one["right optic nerve"]),
        "chiasm": spot(parts["chiasm"]),
        "tract": spot(one["right optic tract"]),
        "geniculate": (geniculate, 30.0),
        "radiation": ((geniculate + cortex) / 2, 60.0),
        "cortex": spot(one["right occipital lobe"]),
        "beyond": (parts["brain"].bounds.mean(axis=0), extent(parts["brain"]) * 0.8),
    }
    route = [stops[stop][0] for stop in ("light", "optics", "receptors", "nerve", "chiasm", "tract",
                                         "geniculate", "radiation", "cortex")]
    return Model(
        parts=parts, stops=stops, routes=[route],
        axes=np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
        camera=(-0.8, 0.35, 0.5),
        budget={"brain": 46000, "visual_cortex": 9000, "eyeball": 3000, "retina": 3000},
        unit="mm", facts={"shell_regions": len(shell)},
    )


# Allen atlas structure ids for each part of the mouse model.
MOUSE_PARTS = {
    "brain": 997,          # root: the whole brain
    "chiasm": 117,         # och
    "optic_tract": 125,    # opt
    "brachium": 916,       # bsc, brachium of the superior colliculus
    "geniculate": 170,     # LGd, dorsal part of the lateral geniculate complex
    "colliculus": 302,     # SCs, superior colliculus, sensory related
    "visual_cortex": 385,  # VISp, primary visual area
}
# The drawn eye. 3.3 mm is close to the published axial length of an adult
# mouse eye (about 3.4 mm) and the lens fills most of it (about 2 mm). Where
# the eye sits relative to the brain, and which way it points (60 degrees out
# from straight ahead, 20 degrees up), is placed by eye from pictures of the
# skull, not measured.
MOUSE_EYE_UM = 3300.0
MOUSE_LENS_UM = 2100.0
MOUSE_EYE_FROM_CHIASM_UM = (-3500.0, -1500.0, 3800.0)  # forward, up, out to the side


def mouse_eye(centre, axis, up):
    radius = MOUSE_EYE_UM / 2
    ball = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    ball.apply_translation(centre)
    lens = trimesh.creation.icosphere(subdivisions=2, radius=MOUSE_LENS_UM / 2)
    lens.apply_translation(centre + unit(axis) * (radius - MOUSE_LENS_UM / 2 - 250))
    return {"eyeball": ball, "lens": lens,
            "cornea": dome(centre, axis, up, radius * 1.03, 60, 60, detail=3),
            "retina": dome(centre, -unit(axis), up, radius * 0.96, 105, 105, detail=3)}


def mouse():
    # The atlas: micrometres, +x toward the tail, +y down, +z toward the animal's right.
    parts = {part: allen_mesh(structure) for part, structure in MOUSE_PARTS.items()}
    up = np.array([0.0, -1.0, 0.0])
    middle = parts["brain"].bounds.mean(axis=0)[2]
    chiasm = parts["chiasm"]
    eyes = []
    for sign in (-1, 1):  # the left eye, then the right
        out = np.array([0.0, 0.0, float(sign)])
        half = chiasm.vertices[(chiasm.vertices[:, 2] - middle) * sign > 0]
        root = half[half[:, 0] <= np.quantile(half[:, 0], 0.1)].mean(axis=0)  # the front corner of the chiasm
        dx, dy, dz = MOUSE_EYE_FROM_CHIASM_UM
        centre = root + np.array([dx, dy, 0.0]) + dz * out
        axis = unit(np.cos(np.radians(20)) * (np.cos(np.radians(60)) * np.array([-1.0, 0, 0])
                                              + np.sin(np.radians(60)) * out) + np.sin(np.radians(20)) * up)
        eye = mouse_eye(centre, axis, up)
        pole = centre + unit(root - centre) * MOUSE_EYE_UM / 2
        eye["optic_nerve"] = tube([pole, root], 170.0)
        eyes.append((eye, centre, axis, root))
    drawn = ["eyeball", "lens", "cornea", "retina", "optic_nerve"]
    for part in drawn:
        parts[part] = join(eye[part] for eye, *_ in eyes)

    # The route: from the left eye, across at the chiasm, up the right side of the brain.
    eye, centre, axis, root = eyes[0]
    right = {part: side(parts[part], 2, True) for part in
             ("optic_tract", "brachium", "geniculate", "colliculus", "visual_cortex")}
    back = centre - axis * MOUSE_EYE_UM / 2 * 0.9
    front = centre + axis * MOUSE_EYE_UM / 2 * 0.7
    point = {part: mesh.vertices.mean(axis=0) for part, mesh in right.items()}
    stops = {
        "light": (centre + axis * 5000, 7000.0),
        "optics": (front, 4000.0),
        "receptors": (back, 4000.0),
        "bipolar": (back, 4000.0),
        "ganglion": (back, 4000.0),
        "nerve": ((centre + root) / 2, 4500.0),
        "chiasm": spot(chiasm),
        "tract": spot(right["optic_tract"]),
        "colliculus": spot(right["colliculus"]),
        "geniculate": spot(right["geniculate"]),
        "cortex": spot(right["visual_cortex"]),
        "beyond": (parts["brain"].bounds.mean(axis=0), extent(parts["brain"]) * 0.8),
    }
    trunk = [stops["light"][0], front, back, root, chiasm.vertices.mean(axis=0), point["optic_tract"],
             point["geniculate"]]
    return Model(
        parts=parts, stops=stops,
        routes=[trunk + [point["visual_cortex"]], [point["geniculate"], point["brachium"], point["colliculus"]]],
        axes=np.array([[0, 0, -1.0], [0, -1.0, 0], [-1.0, 0, 0]]),
        camera=(0.5, 0.6, 0.65),
        budget={"brain": 40000, "visual_cortex": 5000, "eyeball": 1300},
        unit="um", drawn=drawn,
    )


# Virtual Fly Brain painted domains on the JFRC2 template (VFB_0003xxxx) for each part.
FLY_PARTS = {"medulla": "0624", "lobula": "0622", "lobula_plate": "0612"}
FLY_SHELL = ["0849", "0840"]  # adult cerebral ganglion, adult gnathal ganglion
FLY_TARGET = "0632"           # posterior ventrolateral protocerebrum: where much of the lobula's output lands
# The drawn eye. Facets 5 degrees apart (Land 1997) and 16 micrometres wide
# put them on a sphere of radius 16 / radians(5) = 183 micrometres. The dome's
# outline (taller than wide) is set so that it holds about 750 facets
# (the figure in species/fly.py). The lamina is a sheet under it. Where both
# sit relative to the medulla is placed by eye.
FLY_FACET_UM = 16.0
FLY_SPACING_DEG = 5.0
FLY_EYE_HALF_ANGLES = (63.0, 82.0)   # front to back, top to bottom
FLY_LAMINA_RADIUS_UM = 95.0
FLY_EYE_CENTRE_INSIDE_UM = 55.0      # the dome's centre, in from the medulla's outer face


def fly():
    # JFRC2: micrometres, +x toward the animal's left, +y down, +z toward the back.
    parts = {part: vfb_surface(domain) for part, domain in FLY_PARTS.items()}
    parts["brain"] = join(vfb_surface(domain) for domain in FLY_SHELL)
    up = np.array([0.0, -1.0, 0.0])
    radius = FLY_FACET_UM / np.radians(FLY_SPACING_DEG)
    across, tall = FLY_EYE_HALF_ANGLES
    eyes, laminas, centres, count = [], [], [], 0
    for positive in (False, True):  # the right eye, then the left
        medulla = side(parts["medulla"], 0, positive)
        out = np.array([1.0 if positive else -1.0, 0.0, 0.0])
        face = medulla.vertices[:, 0].max() if positive else medulla.vertices[:, 0].min()
        centre = medulla.vertices.mean(axis=0)
        centre[0] = face - out[0] * FLY_EYE_CENTRE_INSIDE_UM
        lenses, count = facets(centre, out, up, radius, FLY_SPACING_DEG, across, tall)
        eyes.append(join([lenses, dome(centre, out, up, radius * 0.985, across + 3, tall + 3)]))
        laminas.append(dome(centre, out, up, FLY_LAMINA_RADIUS_UM, across * 0.9, tall * 0.9))
        centres.append((centre, out))
    parts["eye"] = join(eyes)
    parts["lamina"] = join(laminas)

    # The route: in through the right eye and across that side's optic lobe.
    centre, out = centres[0]
    right = {part: side(parts[part], 0, False) for part in ("medulla", "lobula", "lobula_plate")}
    target = side(vfb_surface(FLY_TARGET), 0, False).vertices.mean(axis=0)
    surface = centre + out * radius
    point = {part: mesh.vertices.mean(axis=0) for part, mesh in right.items()}
    lamina = centre + out * FLY_LAMINA_RADIUS_UM
    lobes = join([right["lobula"], right["lobula_plate"]])
    stops = {
        "light": (surface + out * 150, 500.0),
        "optics": (surface, 320.0),
        "receptors": (surface - out * 45, 320.0),
        "lamina": (lamina, 260.0),
        "medulla": spot(right["medulla"]),
        "lobula": (lobes.vertices.mean(axis=0), extent(lobes)),
        "brain": (parts["brain"].bounds.mean(axis=0), extent(parts["brain"]) * 0.9),
    }
    trunk = [stops["light"][0], surface, lamina, point["medulla"], point["lobula"], target]
    return Model(
        parts=parts, stops=stops,
        routes=[trunk, [point["medulla"], point["lobula_plate"]]],
        axes=np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
        camera=(-0.5, 0.35, 0.8),
        budget={"brain": 30000, "medulla": 9000, "lobula": 5000, "lobula_plate": 5000, "lamina": 2500},
        unit="um", drawn=["eye", "lamina"],
        facts={"facets_per_eye": count},
    )


SPECIES = {"human": human, "mouse": mouse, "fly": fly}


# ---------------------------------------------------------------- writing

def build(name):
    model = SPECIES[name]()
    corners = np.array([mesh.bounds for mesh in model.parts.values()])
    low, high = corners[:, 0].min(axis=0), corners[:, 1].max(axis=0)
    centre = (low + high) / 2
    scale = RADIUS / (np.linalg.norm(high - low) / 2)

    def to_scene(points):
        return ((np.asarray(points, dtype=float) - centre) * scale) @ model.axes.T

    scene = trimesh.Scene()
    faces = {}
    for part, mesh in model.parts.items():
        mesh = trimesh.Trimesh(to_scene(mesh.vertices), mesh.faces, process=True)
        limit = model.budget.get(part, 4000)
        if part != "eye" and len(mesh.faces) > limit:  # the facets are already as few as they can be
            mesh = mesh.simplify_quadric_decimation(face_count=limit)
        mesh.fix_normals()
        mesh.visual = trimesh.visual.ColorVisuals()
        faces[part] = int(len(mesh.faces))
        scene.add_geometry(mesh, node_name=part, geom_name=part)
    target = MODELS / f"{name}.glb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(scene.export(file_type="glb"))

    def rounded(points):
        return np.round(to_scene(points), 3).tolist()

    stops = {}
    for stop, (point, size) in model.stops.items():
        if size is None:  # as big as the part the stop sits in
            size = min(extent(mesh) for mesh in model.parts.values()
                       if np.all(mesh.bounds[0] <= point + 1e-6) and np.all(point <= mesh.bounds[1] + 1e-6))
        stops[stop] = {"position": rounded(point), "size": round(max(float(size) * scale, 1.2), 3)}
    entry = {
        "stops": stops,
        "routes": [rounded(route) for route in model.routes],
        "camera": np.round(unit(model.camera) * RADIUS * 2.5, 3).tolist(),
        "faces": faces,
        "drawn": sorted(model.drawn),
        "unit": model.unit,
        "units_per_scene_unit": round(1 / scale, 3),
        **model.facts,
    }
    anatomy = json.loads(ANATOMY.read_text(encoding="utf-8")) if ANATOMY.exists() else {}
    anatomy[name] = entry
    ANATOMY.parent.mkdir(parents=True, exist_ok=True)
    ANATOMY.write_text(json.dumps(anatomy, indent=1) + "\n", encoding="utf-8")
    print(f"{target}: {target.stat().st_size / 1e6:.2f} MB, {sum(faces.values())} faces {faces}")
    for stop, where in stops.items():
        print(f"  {stop:11s} {where}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SPECIES:
        raise SystemExit(f"usage: python tools/build_anatomy.py {{{'|'.join(SPECIES)}}}")
    build(sys.argv[1])
