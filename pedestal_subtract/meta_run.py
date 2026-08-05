"""
meta_run — set up and track an active *meta_studies run* for pedestal_subtract.

A "meta run" is a single ``analysis_tools.meta_studies``-compatible JSON config
file that collects the per-extension summary tables (``extension_summary.csv``)
produced by successive pedestal_subtract analyses. The workflow is:

    1. ``python scripts/set_meta_run.py <run_name> --directory <dir>`` creates
       ``<dir>/<run_name>.json`` (a fully-defaulted meta_studies config with an
       empty ``datasets`` list) and records it as the *active run*.
    2. Every subsequent pedestal_subtract analysis that writes an
       ``extension_summary.csv`` appends a dataset entry pointing at that CSV to
       the active run's JSON, with empty ``x`` and ``series`` placeholders for
       you to fill in later.
    3. ``python scripts/set_meta_run.py --unset`` clears the active run, so
       later analyses are not added to any JSON file.

The active-run pointer is stored in a small state file living **inside this
package directory** (see ``STATE_PATH``). Because nonlinearity_studies ships its
own independent copy of this module with its own state file, a pedestal_subtract
run and a nonlinearity_studies run can be active simultaneously without
conflicting.

State also auto-expires: if more than ``EXPIRY_HOURS`` (24 h) have passed since
the run was set, it is treated as unset (and the stale state file is removed).
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Which detector package this module belongs to (used only for messages).
PACKAGE_NAME = "pedestal_subtract"

#: Active runs older than this many hours are treated as unset.
EXPIRY_HOURS = 24

#: Where the active-run pointer is stored. Lives inside the package directory so
#: it is found deterministically regardless of the current working directory,
#: and is independent from the nonlinearity_studies copy.
STATE_PATH = Path(__file__).with_name(".meta_run_state.json")

#: Section key for this package within the unified registry.
SECTION = PACKAGE_NAME

#: Unified cross-package "context" registry of every meta run ever created. It
#: lives in the shared parent of both repos (``Privitera_335``) so it is *not*
#: tracked by either git repo, and both pedestal_subtract and nonlinearity_studies
#: write to the same file (into their own section). ``parents[2]`` walks
#: meta_run.py -> package dir -> repo root -> shared parent. Override with the
#: ``META_RUNS_CONTEXT_PATH`` environment variable.
REGISTRY_PATH = Path(
    os.environ.get("META_RUNS_CONTEXT_PATH")
    or (Path(__file__).resolve().parents[2] / "meta_runs_context.json")
)

#: Default ``quantities`` block for a freshly created pedestal_subtract meta run.
#: These are the columns pedestal_subtract writes to extension_summary.csv.
DEFAULT_QUANTITIES = OrderedDict([
    ("gain_adu_per_e", {"ylabel": "Gain (ADU/e-)", "title": "Gain", "fit_line": False}),
    ("noise_adu", {"ylabel": "Noise (ADU)", "title": "Noise (ADU)", "fit_line": False}),
    ("noise_e", {"ylabel": "Noise (e-)", "title": "Noise (e-)", "fit_line": False}),
    ("dark_current_weighted_e_per_pix_day", {"ylabel": "DC (e-/pix/day)",
                                             "title": "Dark current", "fit_line": False}),
])


def _default_template(run_name):
    """A fully-defaulted meta_studies config with an empty ``datasets`` list.

    Field set and default values mirror the canonical meta_studies example
    config; ``study_name``/``suptitle`` are seeded from ``run_name`` and
    ``output_dir`` is left blank for the user to fill in.
    """
    return OrderedDict([
        ("study_name", run_name),
        ("suptitle", run_name.replace("_", " ")),
        ("x_label", ""),
        ("invert_x", False),
        ("x_axis", "value"),
        ("output_dir", ""),
        ("series_label", None),
        ("ext_separate_plot", False),
        ("series_separate_plot", True),
        ("plot_individual", True),
        ("plot_together", True),
        ("plot_mega", False),
        ("plot_overlaid", False),
        ("show_plots", True),
        ("save_output", True),
        ("save_plots", True),
        ("save_csv", True),
        ("save_report", True),
        ("individual_figsize", [9, 7]),
        ("together_subplots_figsize", [16, 9]),
        ("mega_subplots_figsize", [16, 9]),
        ("colors", None),
        ("dpi", 450),
        ("use_average", False),
        ("show_multiple_values", True),
        ("show_error_bars", False),
        ("fit_line", False),
        ("connect_points", True),
        ("error_bar_line", True),
        ("together_subplot_ncols", 3),
        ("series_column", None),
        ("suppress_series_plot", False),
        ("series_line_styles", ["-", "--"]),
        ("series_markers", ["o", "^"]),
        ("uniform_series_style", False),
        ("quantities", OrderedDict(DEFAULT_QUANTITIES)),
        ("datasets", []),
    ])


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _read_state():
    """Return the raw state dict, or ``None`` if there is no state file."""
    if not STATE_PATH.exists():
        return None
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt/unreadable state is treated as "no active run".
        return None


def _write_state(state):
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)


def _clear_state():
    try:
        STATE_PATH.unlink()
    except FileNotFoundError:
        pass


def _is_expired(set_at_iso):
    try:
        set_at = datetime.fromisoformat(set_at_iso)
    except (TypeError, ValueError):
        return True
    return datetime.now() - set_at > timedelta(hours=EXPIRY_HOURS)


# ---------------------------------------------------------------------------
# Unified registry ("context") helpers
# ---------------------------------------------------------------------------

def _read_registry():
    """Return the whole registry (all package sections), or an empty dict."""
    if not REGISTRY_PATH.exists():
        return OrderedDict()
    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except (json.JSONDecodeError, OSError):
        return OrderedDict()


def _write_registry(reg):
    # Best-effort bookkeeping: never let a registry write break an analysis run.
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REGISTRY_PATH.open("w", encoding="utf-8") as f:
            json.dump(reg, f, indent=4)
    except OSError:
        pass


def _new_record(run_name, path, now):
    return OrderedDict([
        ("run_name", run_name),
        ("path", str(path)),
        ("created_at", now),
        ("last_set_at", now),
        ("times_set", 0),
        ("last_unset_at", None),
        ("last_appended_at", None),
        ("dataset_count", 0),
        ("notes", []),
    ])


def _register_created(run_name, path, note=None):
    """Record (or refresh) a run in this package's registry section.

    Creates the record on first use, updates ``last_set_at``/``times_set`` on
    every (re)activation, and appends ``note`` (timestamped) when supplied.
    """
    reg = _read_registry()
    section = reg.setdefault(SECTION, OrderedDict())
    now = datetime.now().isoformat(timespec="seconds")
    rec = section.get(run_name) or _new_record(run_name, path, now)
    section[run_name] = rec
    rec["path"] = str(path)
    rec["last_set_at"] = now
    rec["times_set"] = rec.get("times_set", 0) + 1
    if note:
        rec.setdefault("notes", []).append(
            OrderedDict([("at", now), ("note", note)]))
    _write_registry(reg)


def _register_appended(run_name, path, dataset_count):
    """Update ``last_appended_at``/``dataset_count`` for a run in the registry."""
    reg = _read_registry()
    section = reg.setdefault(SECTION, OrderedDict())
    now = datetime.now().isoformat(timespec="seconds")
    rec = section.get(run_name) or _new_record(run_name, path, now)
    section[run_name] = rec
    rec["last_appended_at"] = now
    rec["dataset_count"] = dataset_count
    _write_registry(reg)


def _register_unset(run_name):
    reg = _read_registry()
    section = reg.get(SECTION)
    if not section or run_name not in section:
        return
    section[run_name]["last_unset_at"] = datetime.now().isoformat(timespec="seconds")
    _write_registry(reg)


def registry():
    """Return the full unified registry (both package sections)."""
    return _read_registry()


def format_registry():
    """A human-readable listing of every recorded run, newest first per section."""
    reg = _read_registry()
    if not reg:
        return f"No meta runs recorded yet.\n  (registry: {REGISTRY_PATH})"
    lines = [f"Meta runs context: {REGISTRY_PATH}"]
    for section, runs in reg.items():
        lines.append(f"\n[{section}]")
        if not runs:
            lines.append("  (none)")
            continue
        for name, rec in sorted(runs.items(),
                                key=lambda kv: kv[1].get("last_set_at", ""),
                                reverse=True):
            lines.append(f"  {name}")
            lines.append(f"      path:      {rec.get('path')}")
            lines.append(f"      created:   {rec.get('created_at')}")
            lines.append(f"      last set:  {rec.get('last_set_at')}  "
                         f"(set x{rec.get('times_set')})")
            lines.append(f"      datasets:  {rec.get('dataset_count', 0)}   "
                         f"last append: {rec.get('last_appended_at')}")
            if rec.get("last_unset_at"):
                lines.append(f"      last unset: {rec.get('last_unset_at')}")
            for n in (rec.get("notes") or []):
                lines.append(f"      note [{n.get('at')}]: {n.get('note')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def active_run():
    """Return the active run's state dict, or ``None``.

    Returns ``None`` when there is no active run, or when the active run has
    expired (older than ``EXPIRY_HOURS``); in the expired case the stale state
    file is removed as a side effect. The returned dict has keys ``run_name``,
    ``path`` and ``set_at``.
    """
    state = _read_state()
    if not state:
        return None
    if _is_expired(state.get("set_at")):
        _clear_state()
        return None
    # If the target JSON has since been deleted, the run is effectively gone.
    if not Path(state.get("path", "")).exists():
        return None
    return state


def create_run(run_name, directory, note=None):
    """Create ``<directory>/<run_name>.json`` (if absent) and make it active.

    An existing JSON of the same name is preserved (its collected ``datasets``
    are kept); the run is simply re-activated and its 24 h timer reset. When
    ``note`` is given it is appended (timestamped) to the run's note history in
    the unified registry. Returns a tuple ``(path, existed)`` where ``existed``
    is whether the JSON already existed.
    """
    if not run_name or not str(run_name).strip():
        raise ValueError("run_name must be a non-empty string")

    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}.json"

    existed = path.exists()
    if not existed:
        with path.open("w", encoding="utf-8") as f:
            json.dump(_default_template(run_name), f, indent=4)

    _write_state({
        "run_name": run_name,
        "path": str(path.resolve()),
        "set_at": datetime.now().isoformat(timespec="seconds"),
    })
    _register_created(run_name, path.resolve(), note=note)
    return path.resolve(), existed


def unset_run():
    """Deactivate the current run. Returns the deactivated state dict or ``None``."""
    state = _read_state()
    _clear_state()
    if state and state.get("run_name"):
        _register_unset(state["run_name"])
    return state


def renew_run():
    """Reset the 24 h expiry timer on the currently active run.

    The run's JSON file is left untouched — only the active-run timestamp is
    refreshed (both in the local state file and the unified registry). Returns
    the refreshed state dict, or ``None`` if there is no active run to renew.
    """
    state = active_run()
    if state is None:
        return None

    new_state = {
        "run_name": state["run_name"],
        "path": state["path"],
        "set_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_state(new_state)
    _register_created(state["run_name"], Path(state["path"]))
    return new_state


# Return codes from append_dataset, so callers can print a useful message.
APPENDED = "appended"
DUPLICATE = "duplicate"
NO_ACTIVE_RUN = "no_active_run"


def append_dataset(table_path, x="", series="", exclude_ext=None):
    """Append a dataset entry for ``table_path`` to the active run's JSON.

    The entry is ``{"x": x, "exclude_ext": exclude_ext or [], "series": series,
    "table": <absolute path>}``. If the same table path is already present the
    call is a no-op (de-duplication), so re-running an analysis will not add the
    same CSV twice. Returns one of ``APPENDED``, ``DUPLICATE`` or
    ``NO_ACTIVE_RUN``.
    """
    state = active_run()
    if state is None:
        return NO_ACTIVE_RUN

    json_path = Path(state["path"])
    table_abs = str(Path(table_path).expanduser().resolve())

    with json_path.open("r", encoding="utf-8") as f:
        config = json.load(f, object_pairs_hook=OrderedDict)

    datasets = config.setdefault("datasets", [])
    for entry in datasets:
        existing = entry.get("table")
        if existing and str(Path(existing)) == str(Path(table_abs)):
            return DUPLICATE

    datasets.append(OrderedDict([
        ("x", x),
        ("exclude_ext", list(exclude_ext) if exclude_ext else []),
        ("series", series),
        ("table", table_abs),
    ]))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    _register_appended(state["run_name"], json_path, len(datasets))
    return APPENDED


def status():
    """Return a human-readable one-line description of the current active run."""
    state = active_run()
    if state is None:
        return f"No active {PACKAGE_NAME} meta run."
    set_at = datetime.fromisoformat(state["set_at"])
    expires = set_at + timedelta(hours=EXPIRY_HOURS)
    remaining = expires - datetime.now()
    hours_left = max(0, remaining.total_seconds()) / 3600
    return (f"Active {PACKAGE_NAME} meta run: '{state['run_name']}'\n"
            f"  file:    {state['path']}\n"
            f"  set at:  {state['set_at']}\n"
            f"  expires: {expires.isoformat(timespec='seconds')} "
            f"({hours_left:.1f} h left)")
