"""Measure reconstruction quality, time and memory on the bundled samples.

Runs one species on every sample across 60 degrees, with real neurons (100 ms
of spikes, seed 0) and with ideal neurons (no noise), and prints a table.

Run from the repository root: python scripts/benchmark.py --size 128
Tracing memory slows the run several times over; add --no-memory to time it.
"""
import argparse
import time
import tracemalloc

import numpy as np

from biovision import io
from biovision.run import run

FOV_DEG = 60.0
WINDOW_MS = 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=128, help="picture size in pixels")
    parser.add_argument("--species", default="human")
    parser.add_argument("--looks", type=int, default=1,
                        help="looks that share the spike window, the eye moved for each")
    parser.add_argument("--no-memory", action="store_true",
                        help="do not trace memory; tracing makes the times several times longer")
    args = parser.parse_args()

    names = io.sample_names()
    if not args.no_memory:
        tracemalloc.start()
    rows = []
    for label, noise in (("real, 100 ms", True), ("ideal", False)):
        psnrs, ssims, times, steps, clipped = [], [], [], [], []
        for name in names:
            start = time.perf_counter()
            result = run(io.load_sample(name), args.species, fov_deg=FOV_DEG,
                         size_px=args.size, window_ms=WINDOW_MS, noise=noise, seed=0,
                         looks=args.looks)
            times.append(time.perf_counter() - start)
            psnrs.append(result.metrics["psnr_db"])
            ssims.append(result.metrics["ssim"])
            solve = result.reconstruction
            steps.append(solve.iterations if solve.converged else float("inf"))
            clipped.append(100.0 * np.mean(result.code.intermediates["rate"] == 0.0))
        rows.append((label, psnrs, ssims, times, steps, clipped))
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    print(f"{args.species}, {args.size} px, {FOV_DEG:g} degrees, "
          f"{int(result.metrics['neurons'])} neurons, {args.looks} look(s)")
    print(f"{'neurons':<14}{'mean PSNR':>10}{'mean SSIM':>11}{'s per run':>11}{'slowest':>9}"
          f"{'most steps':>12}{'most clipped':>14}   PSNR per sample ({' / '.join(names)})")
    for label, psnrs, ssims, times, steps, clipped in rows:
        each = " / ".join(f"{value:.1f}" for value in psnrs)
        print(f"{label:<14}{np.mean(psnrs):>8.2f} dB{np.mean(ssims):>11.3f}"
              f"{np.mean(times):>11.1f}{max(times):>9.1f}{max(steps):>12.0f}"
              f"{max(clipped):>13.3f}%   {each}")
    memory = ("not traced" if args.no_memory
              else f"{peak_mb:.0f} MB, and the times include tracing it")
    print(f"peak memory {memory} (the slowest run is the first, which builds the eye;"
          " steps are the solver's, inf if it did not converge; clipped cells fire at zero)")


if __name__ == "__main__":
    main()
