"""Command line: `biovision list | run | report`."""
import argparse
import sys
import warnings

from . import io
from .core.registry import species
from .report import figures
from .report.export import build_report, write_report
from .run import run


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", help="path to an image; a bundled sample if omitted")
    parser.add_argument("--sample", default="astronaut", help="bundled sample to use")
    parser.add_argument("--size", type=int, default=128, help="working size in pixels")
    parser.add_argument("--fov", type=float, default=60.0, help="field of view in degrees")
    parser.add_argument("--window", type=float, default=100.0, help="spike window in ms")
    parser.add_argument("--no-noise", action="store_true", help="exact spike counts")
    parser.add_argument("--lam", type=float, default=None, help="regularization strength")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--density", type=float, default=1.0,
                        help="receptor density relative to the real eye")
    parser.add_argument("--neuron-density", type=float, default=1.0,
                        help="cortex cell density relative to the real animal")
    parser.add_argument("--looks", type=int, default=1,
                        help="looks that share the spike window, the eye moved a little "
                             "for each (fixational eye movements)")


    parser.add_argument("--photons", type=float, default=None,
                        help="light level: photons a receptor catches per second where the "
                             "picture is white (human cone: sunlight about 1e7, a lit room "
                             "1e5); unlimited light if omitted")


def _settings(args) -> dict:
    return dict(size_px=args.size, fov_deg=args.fov, window_ms=args.window,
                noise=not args.no_noise, lam=args.lam, seed=args.seed,
                density=args.density, neuron_density=args.neuron_density,
                looks=args.looks, photons_per_s=args.photons)


def _image(args):
    return io.load_image(args.image) if args.image else io.load_sample(args.sample)


def serve(host: str, port: int) -> int:
    """Run the web server. Needs the `server` extra."""
    try:
        import uvicorn

        from .server import app
    except ImportError:
        print('error: the server needs extra packages; run: pip install -e ".[server]"',
              file=sys.stderr)
        return 2
    print(f"biovision is running at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biovision", description="Encode an image through a species' visual system "
                                      "and reconstruct it from the neural code.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the registered species")
    serve_parser = commands.add_parser("serve", help="start the web app's server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    run_parser = commands.add_parser("run", help="one species, one figure")
    run_parser.add_argument("--species", required=True)
    run_parser.add_argument("--out", default="biovision_run.png", help="figure to write")
    _add_common(run_parser)
    report_parser = commands.add_parser("report", help="full report for one or all species")
    report_parser.add_argument("--species", default="all", help="a species name, or 'all'")
    report_parser.add_argument("--out", default="results", help="directory to write")
    report_parser.add_argument("--no-sweeps", action="store_true",
                               help="skip the window, lambda and density sweeps (much faster)")
    _add_common(report_parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            print("\n".join(species.names()))
            return 0
        if args.command == "serve":
            return serve(args.host, args.port)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if args.command == "run":
                result = run(_image(args), args.species, **_settings(args))
                figures.pipeline_panel(result).savefig(args.out, dpi=200)
                metrics = result.metrics
                print(f"{args.species}: PSNR {metrics['psnr_db']:.2f} dB, "
                      f"SSIM {metrics['ssim']:.3f}, {int(metrics['receptors'])} receptors, "
                      f"{int(metrics['neurons'])} neurons, "
                      f"{result.reconstruction.iterations} iterations")
                print(f"figure written to {args.out}")
            else:
                names = None if args.species == "all" else [args.species]
                report = build_report(_image(args), names, sweeps=not args.no_sweeps,
                                      **_settings(args))
                out_dir = write_report(report, args.out)
                print(f"report written to {out_dir}")
        for warning in caught:
            print(f"warning: {warning.message}", file=sys.stderr)
        return 0
    except (KeyError, ValueError, FileNotFoundError) as error:
        message = error.args[0] if error.args else str(error)
        print(f"error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
