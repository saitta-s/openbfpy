"""Command-line entry point."""

import argparse

from .simulation import run_simulation


def main():
    parser = argparse.ArgumentParser(description="Run an openBF YAML model using Python")
    parser.add_argument("config")
    parser.add_argument("--savedir")
    parser.add_argument("--out-files", action="store_true")
    parser.add_argument("--save-stats", action="store_true")
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_simulation(args.config, savedir=args.savedir, out_files=args.out_files,
                            save_stats=args.save_stats, write_output=not args.no_output,
                            verbose=not args.quiet)
    if not args.quiet:
        print(f"Stopped: {result.termination_reason}; {result.steps} steps in {result.elapsed_seconds:.3f} s")


if __name__ == "__main__":
    main()
