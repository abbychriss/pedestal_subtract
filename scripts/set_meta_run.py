#!/usr/bin/env python3
"""
set_meta_run — set up (or tear down) an active meta_studies run for pedestal_subtract.

Creating a run writes a fully-defaulted ``analysis_tools.meta_studies`` config
named ``<run_name>.json`` into a directory of your choice, and marks it as the
*active run*. From then on, every pedestal_subtract analysis that saves an
``extension_summary.csv`` automatically appends a dataset entry pointing at that
CSV to the active run's JSON (with empty ``x`` and ``series`` placeholders).

Usage
-----
    # Create <dir>/<run_name>.json and make it the active run
    python scripts/set_meta_run.py <run_name> --directory <path_to_directory>

    # ...defaulting the directory to pedestal_subtract/config
    python scripts/set_meta_run.py <run_name>

    # Attach a note to the run (appended, timestamped, to its note history)
    python scripts/set_meta_run.py <run_name> --notes "first pass, VDD=-20"

    # Restart the 24 h timer on whatever run is currently active
    python scripts/set_meta_run.py --renew

    # Show the currently active run (if any)
    python scripts/set_meta_run.py --status

    # List every run ever recorded (both packages) from the unified context file
    python scripts/set_meta_run.py --list

    # Stop adding runs to any JSON file
    python scripts/set_meta_run.py --unset

Notes
-----
* Re-running with the *same* name re-activates the existing JSON (its collected
  datasets are kept) and resets the 24 h timer. To start fresh, use a new name.
* Running with a *new* name switches all future runs into the new JSON file.
* The active run auto-expires 24 h after it is set.
* nonlinearity_studies has its own independent ``set_meta_run.py`` / active run,
  so a pedestal_subtract run and a nonlinearity_studies run never conflict.
"""

import argparse
import sys
from pathlib import Path

try:
    from pedestal_subtract import meta_run
except ModuleNotFoundError:
    # Allow running from a source checkout where the package isn't installed.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pedestal_subtract import meta_run

# Default location for the run JSON: <repo>/config
DEFAULT_DIRECTORY = Path(__file__).resolve().parent.parent / "config"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Set or unset an active meta_studies run for pedestal_subtract.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_name", nargs="?", default=None,
        help="Name of the run. Used as both the JSON file name and the "
             "meta_studies 'study_name'. Required unless --unset/--status.",
    )
    parser.add_argument(
        "-d", "--directory", type=str, default=str(DEFAULT_DIRECTORY),
        help=f"Directory to create <run_name>.json in "
             f"(default: {DEFAULT_DIRECTORY}).",
    )
    parser.add_argument(
        "--unset", action="store_true",
        help="Deactivate the current run so later analyses are not added to any JSON.",
    )
    parser.add_argument(
        "--renew", action="store_true", dest="restart_timer",
        help="Reset the 24 h expiry timer on the currently active run, without "
             "otherwise changing it (JSON file and datasets are untouched).",
    )
    parser.add_argument(
        "-n", "--notes", type=str, default=None,
        help="A note about this run, appended (timestamped) to its note history "
             "in the unified context file. Re-run with a new --notes to add more.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print the currently active run (if any) and exit.",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_runs",
        help="List every run recorded in the unified context file and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_runs:
        print(meta_run.format_registry())
        return 0

    if args.status:
        print(meta_run.status())
        return 0

    if args.restart_timer:
        state = meta_run.renew_run()
        if state is None:
            print(f"No active {meta_run.PACKAGE_NAME} meta run to restart the timer on.")
        else:
            print(f"Restarted the 24 h timer for {meta_run.PACKAGE_NAME} meta run '{state['run_name']}'.")
            print(f"  file:   {state['path']}")
            print(f"  set at: {state['set_at']}")
        return 0

    if args.unset:
        state = meta_run.unset_run()
        if state is None:
            print(f"No active {meta_run.PACKAGE_NAME} meta run to unset.")
        else:
            print(f"Unset {meta_run.PACKAGE_NAME} meta run '{state.get('run_name')}'. "
                  f"Future analyses will not be added to any JSON file.")
        return 0

    if not args.run_name:
        parser.error("run_name is required unless --unset or --status is given.")

    path, existed = meta_run.create_run(args.run_name, args.directory, note=args.notes)
    verb = "Re-activated existing" if existed else "Created"
    print(f"{verb} meta run '{args.run_name}'.")
    print(f"  file: {path}")
    if args.notes:
        print(f"  note: {args.notes}")
    print(f"All future pedestal_subtract runs will be appended to this file "
          f"(auto-expires in {meta_run.EXPIRY_HOURS} h).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
