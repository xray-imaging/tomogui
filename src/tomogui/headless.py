"""Headless core for tomogui.

Pure-Python module (no PyQt5) exposing everything the GUI does with the
reconstruction backend — path resolution, COR persistence, AI-COR flag
building, tomocupy subprocess launchers, batch orchestration, tomolog
upload — as reusable functions/classes.

Intended callers:
  * ``tomogui.cli`` (the shipped CLI)
  * external agents scripting reconstructions in Python
  * the GUI itself once we finish porting its logic here

Conventions match the GUI so a session can be driven from either side:
  * per-file params live in ``{data}/recon_params.json``
  * chosen CORs live in ``{data}/rot_cen.json``
  * try output lives in ``{data}_rec/try_center/{proj}/*.tiff`` plus
    ``center_of_rotation.txt`` (tomocupy AI writes this)
  * full output lives in ``{data}_rec/{proj}_rec.h5`` (h5* save formats)
    or ``{data}_rec/{proj}_rec/*.tiff`` (legacy tiff save format)
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Keep this import light — h5py is needed for shape reads and status checks.
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# HDF5 lock env: NFS-hosted data returns EAGAIN on flock, so disable HDF5's
# own file locking process-wide before anything imports h5py through us.
# ---------------------------------------------------------------------------
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Configuration bundle for a tomogui headless session.

    ``data_folder``  — directory containing the source ``.h5`` projection files.
    ``model_path``   — DINOv2 weights for tomocupy's AI COR finder. Optional;
                       AI operations return an error if unset.
    ``recon_way``    — ``recon`` (chunked z) or ``recon_steps`` (chunked z+θ).
    ``ai_search_method`` — ``fine`` (default) or ``full`` (multi-stage bin refine).
    ``gpu``          — GPU index for CUDA_VISIBLE_DEVICES on launched subprocesses.
    ``extra_args``   — list of extra ``--flag value`` pairs appended to every
                       tomocupy command (Reconstruction/Rings/Phase/etc. tab
                       overrides). Deduped by the last-wins rule at dispatch.
    """
    data_folder: str
    model_path: str | None = None
    recon_way: str = "recon"
    ai_search_method: str = "fine"
    gpu: int | None = None
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.data_folder = str(self.data_folder)
        if self.model_path:
            self.model_path = str(self.model_path)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def proj_name_of(proj_file: str) -> str:
    return os.path.splitext(os.path.basename(proj_file))[0]


def try_dir_of(data_folder: str, proj_file: str) -> str:
    return os.path.join(f"{data_folder}_rec", "try_center",
                        proj_name_of(proj_file))


def full_h5_of(data_folder: str, proj_file: str) -> str:
    return os.path.join(f"{data_folder}_rec",
                        f"{proj_name_of(proj_file)}_rec.h5")


def full_tiff_dir_of(data_folder: str, proj_file: str) -> str:
    return os.path.join(f"{data_folder}_rec",
                        f"{proj_name_of(proj_file)}_rec")


def resolve_full_recon(data_folder: str, proj_file: str) -> dict:
    """Return ``{'kind': 'h5'|'tiff'|None, ...}`` for the reconstruction of
    ``proj_file``. Same contract as ``MainWindow._resolve_full_recon``."""
    result: dict = {'kind': None, 'h5_path': None, 'tiff_files': [],
                    'n_slices': 0, 'range': None}
    h5_path = full_h5_of(data_folder, proj_file)
    if os.path.isfile(h5_path):
        try:
            with h5py.File(h5_path, 'r') as fh:
                n = int(fh['/exchange/data'].shape[0])
            result.update(kind='h5', h5_path=h5_path,
                          n_slices=n, range=(0, max(0, n - 1)))
            return result
        except (OSError, KeyError):
            pass
    tiff_dir = full_tiff_dir_of(data_folder, proj_file)
    if os.path.isdir(tiff_dir):
        tiffs = sorted(glob.glob(os.path.join(tiff_dir, "*.tiff")))
        if tiffs:
            try:
                n1 = int(Path(tiffs[0]).stem.split("_")[-1])
                n2 = int(Path(tiffs[-1]).stem.split("_")[-1])
            except (ValueError, IndexError):
                n1, n2 = 0, len(tiffs) - 1
            result.update(kind='tiff', tiff_files=tiffs,
                          n_slices=len(tiffs), range=(n1, n2))
    return result


def list_h5(data_folder: str, pattern: str = "*.h5") -> list[dict]:
    """Enumerate H5 projection files with reconstruction status."""
    if not data_folder or not os.path.isdir(data_folder):
        return []
    out = []
    for path in sorted(glob.glob(os.path.join(data_folder, pattern))):
        info = resolve_full_recon(data_folder, path)
        try_d = try_dir_of(data_folder, path)
        has_try = os.path.isdir(try_d) and bool(
            glob.glob(os.path.join(try_d, "*.tiff")))
        out.append({
            "path": path,
            "name": os.path.basename(path),
            "size": os.path.getsize(path),
            "has_try": has_try,
            "has_full": info['kind'] is not None,
            "full_kind": info['kind'],
            "full_slices": info['n_slices'],
        })
    return out


# ---------------------------------------------------------------------------
# COR I/O (rot_cen.json)
# ---------------------------------------------------------------------------

def cor_json_path(data_folder: str) -> str:
    return os.path.join(data_folder, "rot_cen.json")


def load_cor_data(data_folder: str) -> dict:
    """Read the ``rot_cen.json`` map (proj_file → COR-as-string). Missing or
    unreadable file returns {}."""
    p = cor_json_path(data_folder)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, json.JSONDecodeError):
        return {}


def save_cor_data(data_folder: str, cor_data: dict) -> None:
    p = cor_json_path(data_folder)
    with open(p, "w") as f:
        json.dump(cor_data, f, indent=2)


def set_cor(data_folder: str, proj_file: str, cor: float) -> None:
    """Set the COR for one file in ``rot_cen.json`` (persisted immediately)."""
    data = load_cor_data(data_folder)
    data[proj_file] = f"{float(cor):.2f}"
    save_cor_data(data_folder, data)


def get_cor(data_folder: str, proj_file: str) -> float | None:
    data = load_cor_data(data_folder)
    val = data.get(proj_file)
    if val is None:
        return None
    try:
        if isinstance(val, list):
            val = val[0] if val else None
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Try-dir helpers (AI COR result file)
# ---------------------------------------------------------------------------

def read_ai_cor(data_folder: str, proj_file: str) -> float | None:
    """Parse the last value from ``try_center/{proj}/center_of_rotation.txt``
    (tomocupy's AI writes this). Returns None if absent/unparseable."""
    p = os.path.join(try_dir_of(data_folder, proj_file),
                     "center_of_rotation.txt")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None
        return float(lines[-1].split()[-1])
    except (OSError, ValueError):
        return None


def clear_stale_ai_cor(data_folder: str, proj_file: str) -> None:
    """Delete ``center_of_rotation.txt`` for a proj before a new AI run —
    tomocupy appends, so leftover values from a previous run would confuse
    the last-line parse in :func:`read_ai_cor`."""
    p = os.path.join(try_dir_of(data_folder, proj_file),
                     "center_of_rotation.txt")
    if os.path.isfile(p):
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Middle-of-width fallback seed
# ---------------------------------------------------------------------------

def middle_of_width(proj_file: str) -> float | None:
    """Return image_width / 2 (i.e. /exchange/data.shape[2] / 2) from
    ``proj_file``. Used as the AI-COR seed when no COR is available in either
    the row or top-bar. Returns None if the file can't be opened."""
    if not proj_file or not os.path.isfile(proj_file):
        return None
    try:
        with h5py.File(proj_file, 'r') as fh:
            shape = fh['/exchange/data'].shape
    except (OSError, KeyError):
        return None
    if len(shape) < 3:
        return None
    return float(shape[2]) / 2.0


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _strip_flag(cmd: list[str], flag: str) -> list[str]:
    out = []
    skip = False
    for a in cmd:
        if skip:
            skip = False
            continue
        if a == flag:
            skip = True
            continue
        out.append(a)
    return out


def build_ai_cor_args(model_path: str, ai_search_method: str = "fine") -> list[str]:
    """CLI flags that enable tomocupy's built-in AI COR finder.

    Both ``--infer-model-path`` and ``--bin-infer-model-path`` are passed —
    the ``full`` method's multi-stage refinement uses the latter, but its
    final ``run_rec`` call still reads the former. See
    ``MainWindow._ai_cor_args`` in the GUI for the same rationale.
    """
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"AI model path invalid: {model_path!r}")
    return [
        "--rotation-axis-method", "ai",
        "--ai-search-method", ai_search_method,
        "--infer-model-path", model_path,
        "--bin-infer-model-path", model_path,
    ]


def apply_ai_cor(cmd: list[str], model_path: str,
                 ai_search_method: str = "fine") -> list[str]:
    """Strip stale AI flags from ``cmd`` and re-append them last so argparse's
    last-wins tie-break selects the AI values."""
    ai = build_ai_cor_args(model_path, ai_search_method)
    for f in ("--rotation-axis-method", "--ai-search-method",
              "--infer-model-path", "--bin-infer-model-path"):
        cmd = _strip_flag(cmd, f)
    return cmd + ai


def _base_cmd(proj_file: str, kind: str, session: Session) -> list[str]:
    return [
        "tomocupy", session.recon_way,
        "--reconstruction-type", kind,
        "--file-name", proj_file,
    ]


def build_try_cmd(proj_file: str, session: Session, cor: float | None = None,
                  auto: bool = False) -> list[str]:
    cmd = _base_cmd(proj_file, "try", session)
    if auto or cor is None:
        cmd += ["--rotation-axis-auto", "auto"]
    else:
        cmd += ["--rotation-axis-auto", "manual",
                "--rotation-axis", f"{float(cor):.6g}"]
    cmd += list(session.extra_args)
    return cmd


def build_full_cmd(proj_file: str, cor: float, session: Session) -> list[str]:
    cmd = _base_cmd(proj_file, "full", session)
    cmd += ["--rotation-axis-auto", "manual",
            "--rotation-axis", f"{float(cor):.6g}"]
    cmd += list(session.extra_args)
    return cmd


def build_ai_try_cmd(proj_file: str, session: Session,
                     seed: float | None = None) -> list[str]:
    """Build a tomocupy try+AI command.

    ``seed`` (if provided) is passed as ``--rotation-axis`` so tomocupy uses
    it as the centre of the search range. When None, the caller should have
    computed a seed already (row / top-bar / mid-width) — this builder does
    NOT fall back to middle_of_width itself, so callers stay in control.
    """
    if not session.model_path:
        raise ValueError("session.model_path must be set for AI COR")
    cmd = _base_cmd(proj_file, "try", session)
    cmd += ["--rotation-axis-auto", "auto"]
    if seed is not None:
        cmd += ["--rotation-axis", f"{float(seed):.6g}"]
    cmd += list(session.extra_args)
    return apply_ai_cor(cmd, session.model_path, session.ai_search_method)


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------

def _env_for(session: Session) -> dict:
    env = os.environ.copy()
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if session.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(session.gpu)
    return env


def run_cmd(cmd: list[str], session: Session, *,
            capture: bool = False, echo: bool = True) -> subprocess.CompletedProcess:
    """Run a command with the right env for tomocupy.

    ``capture=False`` (default): stdout/stderr forwarded live to the parent's
    terminal — right for interactive CLI use and for agents that read a
    log file.

    ``capture=True``: stdout/stderr captured into the returned object's
    ``stdout`` / ``stderr`` attrs. Right for agents that want to parse
    the reconstruction log programmatically.

    Prepends ``env HDF5_USE_FILE_LOCKING=FALSE`` at the shell level too so
    the setting survives every possible env-inheritance quirk.
    """
    prefixed = list(cmd)
    if prefixed and prefixed[0] != "env":
        prefixed = ["env", "HDF5_USE_FILE_LOCKING=FALSE"] + prefixed
    if echo and not capture:
        print("$", " ".join(prefixed), flush=True)
    return subprocess.run(
        prefixed,
        env=_env_for(session),
        capture_output=capture,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------

def run_try(proj_file: str, session: Session, *,
            cor: float | None = None,
            auto: bool = False,
            capture: bool = False) -> int:
    """Try reconstruction. Returns the tomocupy exit code."""
    cmd = build_try_cmd(proj_file, session, cor=cor, auto=auto)
    return run_cmd(cmd, session, capture=capture).returncode


def run_full(proj_file: str, cor: float, session: Session, *,
             capture: bool = False) -> int:
    cmd = build_full_cmd(proj_file, cor, session)
    return run_cmd(cmd, session, capture=capture).returncode


def run_ai_cor(proj_file: str, session: Session, *,
               seed: float | None = None,
               capture: bool = False) -> tuple[int, float | None]:
    """Try + AI COR search in a single tomocupy call. Returns
    ``(exit_code, ai_cor_or_None)``. On success also persists the COR into
    ``rot_cen.json`` and returns the value read from
    ``center_of_rotation.txt`` (tomocupy's AI writes it).
    """
    if seed is None:
        seed = get_cor(session.data_folder, proj_file)
    if seed is None:
        seed = middle_of_width(proj_file)
    clear_stale_ai_cor(session.data_folder, proj_file)
    cmd = build_ai_try_cmd(proj_file, session, seed=seed)
    rc = run_cmd(cmd, session, capture=capture).returncode
    if rc != 0:
        return rc, None
    cor_val = read_ai_cor(session.data_folder, proj_file)
    if cor_val is not None:
        set_cor(session.data_folder, proj_file, cor_val)
    return rc, cor_val


def run_ai_full(proj_file: str, session: Session, *,
                seed: float | None = None,
                capture: bool = False) -> tuple[int, float | None]:
    """AI COR search followed by a full reconstruction using that COR.

    Returns ``(final_exit_code, cor_used_or_None)``. Exit code is the Full
    recon's; if AI failed, returns AI's exit code and Full is skipped.
    """
    rc, cor_val = run_ai_cor(proj_file, session, seed=seed, capture=capture)
    if rc != 0 or cor_val is None:
        return rc, cor_val
    rc_full = run_full(proj_file, cor_val, session, capture=capture)
    return rc_full, cor_val


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    file: str
    phase: str                 # 'try' | 'ai' | 'full' | 'ai-full' | 'tomolog'
    exit_code: int
    cor: float | None = None
    error: str | None = None


def run_batch(files: Iterable[str], session: Session, *,
              phases: Iterable[str] = ("ai", "full"),
              capture: bool = False,
              skip_manual_ai: bool = False,
              tomolog_kwargs: dict | None = None) -> list[BatchResult]:
    """Sequentially run the requested phases on each file.

    ``phases`` — subset of ``{"try", "ai", "full", "tomolog"}`` in the order
    they should run per file.
      * ``try``  — plain try recon (no AI). Needs a per-file COR or ``auto``.
      * ``ai``   — try + tomocupy AI COR. Writes rot_cen.json.
      * ``full`` — full recon using the current COR (from rot_cen.json, or
                   AI just wrote it).
      * ``tomolog`` — tomolog upload; pass ``tomolog_kwargs`` for beamline etc.

    ``skip_manual_ai`` — when True, if a file already has a COR in
    ``rot_cen.json`` at batch start it skips the AI phase for that file
    (matches the "manual override" idea; default is False so AI runs on
    every file and refines around the existing COR).
    """
    phases = list(phases)
    tomolog_kwargs = tomolog_kwargs or {}
    manual_at_start = set()
    if skip_manual_ai:
        preload = load_cor_data(session.data_folder)
        for f in files:
            try:
                if float(preload.get(f, "")):
                    manual_at_start.add(f)
            except (ValueError, TypeError):
                pass
    results: list[BatchResult] = []
    for f in files:
        for phase in phases:
            if phase == "try":
                cor = get_cor(session.data_folder, f)
                rc = run_try(f, session, cor=cor, capture=capture)
                results.append(BatchResult(f, "try", rc, cor=cor))
                if rc != 0:
                    break
            elif phase == "ai":
                if f in manual_at_start:
                    results.append(BatchResult(
                        f, "ai", 0, cor=get_cor(session.data_folder, f),
                        error="skipped (manual COR override)"))
                    continue
                rc, cor_val = run_ai_cor(f, session, capture=capture)
                results.append(BatchResult(f, "ai", rc, cor=cor_val))
                if rc != 0 or cor_val is None:
                    break
            elif phase == "full":
                cor = get_cor(session.data_folder, f)
                if cor is None:
                    results.append(BatchResult(
                        f, "full", -1,
                        error="no COR available for full recon"))
                    break
                rc = run_full(f, cor, session, capture=capture)
                results.append(BatchResult(f, "full", rc, cor=cor))
                if rc != 0:
                    break
            elif phase == "tomolog":
                rc = run_tomolog(f, session, **tomolog_kwargs)
                results.append(BatchResult(f, "tomolog", rc))
                if rc != 0:
                    break
            else:
                results.append(BatchResult(
                    f, phase, -1, error=f"unknown phase: {phase}"))
                break
    return results


# ---------------------------------------------------------------------------
# Tomolog upload
# ---------------------------------------------------------------------------

def auto_contrast(proj_file: str, session: Session,
                  lo_pct: float = 5.0, hi_pct: float = 95.0
                  ) -> tuple[str, str] | tuple[None, None]:
    """Compute (vmin, vmax) as formatted strings from the lo/hi percentile
    of a representative slice. Prefers full H5, falls back to full TIFFs,
    then try TIFFs. Same behaviour as the GUI's ``_auto_contrast_for_file``.
    """
    info = resolve_full_recon(session.data_folder, proj_file)
    try:
        if info['kind'] == 'h5':
            with h5py.File(info['h5_path'], 'r') as fh:
                dset = fh['/exchange/data']
                arr = np.asarray(dset[dset.shape[0] // 2], dtype=np.float32)
        elif info['kind'] == 'tiff':
            from PIL import Image
            mid = info['tiff_files'][len(info['tiff_files']) // 2]
            arr = np.array(Image.open(mid)).astype(np.float32)
        else:
            from PIL import Image
            tdir = try_dir_of(session.data_folder, proj_file)
            tiffs = sorted(glob.glob(os.path.join(tdir, "*.tiff")))
            if not tiffs:
                return (None, None)
            arr = np.array(Image.open(tiffs[len(tiffs) // 2])).astype(np.float32)
        lo = float(np.percentile(arr, lo_pct))
        hi = float(np.percentile(arr, hi_pct))
        if hi <= lo:
            return (None, None)
        return (f"{lo:.6g}", f"{hi:.6g}")
    except Exception:
        return (None, None)


def run_tomolog(proj_file: str, session: Session, *,
                beamline: str = "32-id",
                cloud: str = "imgur",
                url: str = "",
                idx: str = "-1",
                idy: str = "-1",
                idz: str = "-1",
                note: str | None = None,
                vmin: str | None = None,
                vmax: str | None = None,
                auto_contrast_pct: tuple[float, float] | None = None,
                extra_params: str = "",
                capture: bool = False) -> int:
    """Run ``tomolog run`` for one dataset. If ``vmin/vmax`` are None and
    ``auto_contrast_pct`` is set (e.g. ``(5, 95)``), compute per-file
    contrast from the reconstruction; otherwise let tomolog pick defaults.
    """
    if (vmin is None or vmax is None) and auto_contrast_pct is not None:
        avmin, avmax = auto_contrast(proj_file, session, *auto_contrast_pct)
        if vmin is None:
            vmin = avmin
        if vmax is None:
            vmax = avmax
    cmd = [
        "tomolog", "run",
        "--beamline", beamline,
        "--file-name", proj_file,
        "--cloud", cloud,
        "--presentation-url", url,
        "--idx", str(idx),
        "--idy", str(idy),
        "--idz", str(idz),
    ]
    if note:
        cmd += ["--note", f'"{note}"']
    if vmin:
        cmd += ["--min", vmin]
    if vmax:
        cmd += ["--max", vmax]
    if extra_params:
        cmd += extra_params.split()
    return run_cmd(cmd, session, capture=capture).returncode


# ---------------------------------------------------------------------------
# Convenience for external agents
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Volume abstraction — uniform read for H5 and TIFF-stack reconstructions.
# ---------------------------------------------------------------------------

class Volume:
    """Uniform slice-reader over a reconstruction.

    ``Volume`` hides the H5 / TIFF / try-center distinction from callers.
    Use as an iterator (``for arr in vol``) or by index (``vol[k]``); call
    ``vol.close()`` when done (H5 backend holds a file handle). Supports
    ``with`` statements.

    Attributes:
        kind       — 'h5' | 'tiff' | 'try'
        n_slices   — number of frames
        shape      — (n_slices, height, width) tuple
        source     — path of the file/dir the volume was opened from
    """

    def __init__(self, kind: str, source: str, *, n_slices: int,
                 shape: tuple, _h5=None, _tiffs=None):
        self.kind = kind
        self.source = source
        self.n_slices = int(n_slices)
        self.shape = shape
        self._h5 = _h5
        self._tiffs = _tiffs

    @classmethod
    def open(cls, target: str, *, data_folder: str | None = None) -> "Volume":
        """Open a reconstruction volume.

        ``target`` can be:
          * a source projection H5 (``.../scan0001.h5``) — resolves the full
            recon via :func:`resolve_full_recon`; if none exists, falls back
            to the try-center TIFF stack for that scan.
          * a full-recon H5 (``.../scan0001_rec.h5``) — opened directly.
          * a directory of TIFF slices (``.../scan0001_rec``) — opened as
            a stack.
          * an explicit ``center_of_rotation``-style try_center dir.

        ``data_folder`` is only needed when resolving via a source projection
        file whose parent isn't the data folder.
        """
        target = os.path.abspath(target)
        # Direct H5 file
        if os.path.isfile(target) and target.endswith(".h5"):
            # Distinguish "source projection" (has /exchange/data with N proj)
            # from "full recon" (also /exchange/data but interpreted as z,y,x)
            # by looking at siblings: source H5 lives next to the data folder;
            # a *_rec.h5 or *_rec_parts/ pattern → reconstruction file.
            if target.endswith("_rec.h5") or os.path.isdir(target[:-3] + "_parts"):
                return cls._open_h5(target)
            df = data_folder or os.path.dirname(target)
            info = resolve_full_recon(df, target)
            if info["kind"] == "h5":
                return cls._open_h5(info["h5_path"])
            if info["kind"] == "tiff":
                return cls._open_tiff_list(info["tiff_files"],
                                           source=full_tiff_dir_of(df, target))
            # Fall back to try_center for this file
            tdir = try_dir_of(df, target)
            tiffs = sorted(glob.glob(os.path.join(tdir, "*.tiff")))
            if tiffs:
                return cls._open_tiff_list(tiffs, source=tdir, kind="try")
            raise FileNotFoundError(
                f"no reconstruction (h5 or tiff) or try_center TIFFs for "
                f"{target}")
        # Directory of TIFFs
        if os.path.isdir(target):
            tiffs = sorted(glob.glob(os.path.join(target, "*.tiff")))
            if not tiffs:
                raise FileNotFoundError(f"no .tiff files under {target}")
            kind = "try" if os.path.basename(os.path.dirname(target)) == "try_center" else "tiff"
            return cls._open_tiff_list(tiffs, source=target, kind=kind)
        raise FileNotFoundError(target)

    @classmethod
    def _open_h5(cls, path: str) -> "Volume":
        try:
            fh = h5py.File(path, "r", locking=False)
        except (TypeError, ValueError):
            fh = h5py.File(path, "r")
        dset = fh["/exchange/data"]
        return cls("h5", path, n_slices=int(dset.shape[0]),
                   shape=tuple(int(x) for x in dset.shape), _h5=fh)

    @classmethod
    def _open_tiff_list(cls, tiffs: list[str], *, source: str,
                        kind: str = "tiff") -> "Volume":
        from PIL import Image
        with Image.open(tiffs[0]) as im:
            arr = np.array(im)
        h, w = arr.shape[-2:]
        return cls(kind, source, n_slices=len(tiffs),
                   shape=(len(tiffs), h, w), _tiffs=tiffs)

    # ---- slice access -----------------------------------------------------

    def __len__(self) -> int:
        return self.n_slices

    def __getitem__(self, idx: int) -> np.ndarray:
        idx = self._normalize_index(idx)
        if self.kind == "h5":
            return np.asarray(self._h5["/exchange/data"][idx, :, :])
        from PIL import Image
        with Image.open(self._tiffs[idx]) as im:
            arr = np.array(im)
        return arr[..., 0] if arr.ndim == 3 else arr

    def _normalize_index(self, idx: int) -> int:
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"slice index must be int, got {type(idx).__name__}")
        n = self.n_slices
        if idx < 0:
            idx += n
        if not (0 <= idx < n):
            raise IndexError(f"slice {idx} out of range 0..{n - 1}")
        return int(idx)

    def __iter__(self):
        for i in range(self.n_slices):
            yield self[i]

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except OSError:
                pass
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- convenience helpers ---------------------------------------------

    def middle_slice(self) -> np.ndarray:
        return self[self.n_slices // 2]

    def stats(self, idx: int | None = None) -> dict:
        """Basic stats for one slice, or a size-limited sample of the volume.
        ``idx=None`` samples up to 32 evenly-spaced slices to keep memory
        bounded even on huge volumes."""
        if idx is not None:
            arr = self[idx].astype(np.float32, copy=False)
        else:
            step = max(1, self.n_slices // 32)
            picks = list(range(0, self.n_slices, step))[:32]
            arr = np.stack([self[i].astype(np.float32, copy=False)
                            for i in picks], axis=0)
        return {
            "shape": tuple(arr.shape),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "p1": float(np.percentile(arr, 1)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


# ---------------------------------------------------------------------------
# Slice rendering (headless PNG export)
# ---------------------------------------------------------------------------

def render_slice(arr: np.ndarray, *, vmin: float | None = None,
                 vmax: float | None = None, cmap: str = "gray",
                 pct: tuple[float, float] | None = (5.0, 95.0)) -> np.ndarray:
    """Turn a raw slice into an 8-bit RGB image ready for PNG writing.

    When ``vmin`` / ``vmax`` are None, uses ``pct`` (default 5/95 percentile)
    to autoscale. ``cmap`` is any matplotlib colormap name; falls back to
    grayscale if matplotlib isn't importable.
    """
    arr = arr.astype(np.float32, copy=False)
    lo = float(vmin) if vmin is not None else float(np.percentile(arr, pct[0]))
    hi = float(vmax) if vmax is not None else float(np.percentile(arr, pct[1]))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    try:
        import matplotlib.cm as mcm
        table = (mcm.get_cmap(cmap)(norm) * 255.0).astype(np.uint8)
        return table[..., :3]
    except Exception:
        g = (norm * 255.0).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)


def save_slice_png(arr: np.ndarray, out_path: str, **render_kwargs) -> None:
    from PIL import Image
    rgb = render_slice(arr, **render_kwargs)
    Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")


def status(session: Session, pattern: str = "*.h5") -> list[dict]:
    """Return a JSON-serialisable status list for every H5 in the session's
    data folder — reconstruction state + current COR value if any."""
    cor_data = load_cor_data(session.data_folder)
    rows = list_h5(session.data_folder, pattern=pattern)
    for r in rows:
        val = cor_data.get(r["path"])
        try:
            if isinstance(val, list):
                val = val[0] if val else None
            r["cor"] = float(val) if val is not None else None
        except (ValueError, TypeError):
            r["cor"] = None
    return rows
