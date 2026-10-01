"""Command-line interface for tomogui's headless core.

Usage examples::

    # Enumerate H5 files with reconstruction status (JSON output)
    tomogui-cli status /data/session --json

    # Single-file try recon at a fixed COR on GPU 1
    tomogui-cli try /data/session/scan0001.h5 --cor 1024.5 --gpu 1

    # Try + AI COR search + full recon (writes rot_cen.json + full H5)
    tomogui-cli ai-full /data/session/scan0001.h5 \\
        --model /path/to/epoch_10.pth --gpu 0

    # Batch pipeline across every H5 in a folder, AI then full, on GPU 2
    tomogui-cli batch /data/session --phases ai,full \\
        --model /path/to/epoch_10.pth --gpu 2

    # Read / set / list the persisted CORs (rot_cen.json)
    tomogui-cli cor get /data/session/scan0001.h5
    tomogui-cli cor set /data/session/scan0001.h5 1024.5
    tomogui-cli cor list /data/session

    # Tomolog upload with per-file 5–95 % percentile auto-contrast
    tomogui-cli tomolog /data/session/scan0001.h5 --auto-contrast

Every subcommand exits non-zero on failure so an external agent can wire
the CLI directly into a workflow. ``--json`` on the read-style commands
(``status``, ``cor list``, ``cor get``) prints machine-parseable output.

Anything the CLI doesn't cover directly can be reached by importing
``tomogui.headless`` in Python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import headless as H


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(args, model_required: bool = False) -> H.Session:
    data_folder = getattr(args, "data_folder", None) or _infer_data_folder(
        getattr(args, "file", None))
    if not data_folder:
        _die("could not infer data folder — pass --data-folder or a file "
             "inside one")
    model = getattr(args, "model", None)
    if model_required and not (model and os.path.isfile(model)):
        _die(f"AI operation requires --model pointing at valid weights "
             f"(got {model!r})")
    return H.Session(
        data_folder=data_folder,
        model_path=model,
        recon_way=getattr(args, "recon_way", "recon"),
        ai_search_method=getattr(args, "ai_search_method", "fine"),
        gpu=getattr(args, "gpu", None),
        extra_args=_parse_extras(getattr(args, "extra", None)),
    )


def _infer_data_folder(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.dirname(os.path.abspath(path))


def _parse_extras(raw: str | None) -> list[str]:
    """Extras are passed as a single ``--extra`` string, shell-split. Use for
    tomocupy overrides not otherwise exposed, e.g.
    ``--extra "--dezinger 5 --nsino 0.5"``."""
    if not raw:
        return []
    import shlex
    return shlex.split(raw)


def _die(msg: str, code: int = 2) -> None:
    print(f"tomogui-cli: error: {msg}", file=sys.stderr)
    sys.exit(code)


def _emit(obj, as_json: bool):
    if as_json:
        json.dump(obj, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        if isinstance(obj, list):
            for row in obj:
                print(_fmt_row(row))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                print(f"{k}: {v}")
        else:
            print(obj)


def _fmt_row(row: dict) -> str:
    if isinstance(row, dict) and "name" in row:
        cor = row.get("cor")
        cor_s = f"{cor:.2f}" if isinstance(cor, (int, float)) else "-"
        try_s = "try" if row.get("has_try") else "   "
        if row.get("has_full"):
            full_s = f"full({row.get('full_kind')},{row.get('full_slices')})"
        else:
            full_s = "        "
        return f"{row['name']:<60} cor={cor_s:<10} {try_s}  {full_s}"
    return repr(row)


def _files_from_folder(folder: str, pattern: str) -> list[str]:
    import glob
    return sorted(glob.glob(os.path.join(folder, pattern)))


def _files_from_args(args) -> list[str]:
    """Collect file paths from ``--file`` (repeated), ``--files-from`` (a txt
    file with one path per line), and the positional ``targets``."""
    files: list[str] = []
    files += list(getattr(args, "file", None) or [])
    src = getattr(args, "files_from", None)
    if src:
        with open(src) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    files.append(line)
    files += list(getattr(args, "targets", None) or [])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in files:
        p = os.path.abspath(p)
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_status(args) -> int:
    if not args.data_folder:
        _die("status requires --data-folder")
    session = H.Session(data_folder=args.data_folder)
    rows = H.status(session, pattern=args.pattern)
    _emit(rows, as_json=args.json)
    return 0


def cmd_try(args) -> int:
    session = _make_session(args)
    rc = H.run_try(args.file, session,
                   cor=args.cor,
                   auto=args.auto,
                   capture=args.quiet)
    return rc


def cmd_full(args) -> int:
    session = _make_session(args)
    cor = args.cor
    if cor is None:
        cor = H.get_cor(session.data_folder, args.file)
    if cor is None:
        _die(f"no --cor given and no COR in rot_cen.json for {args.file}")
    rc = H.run_full(args.file, cor, session, capture=args.quiet)
    return rc


def cmd_ai_cor(args) -> int:
    session = _make_session(args, model_required=True)
    rc, cor = H.run_ai_cor(args.file, session,
                           seed=args.seed, capture=args.quiet)
    if args.json:
        json.dump({"file": args.file, "exit_code": rc, "ai_cor": cor},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif rc == 0 and cor is not None:
        print(f"AI COR = {cor:.2f}   (saved to rot_cen.json)")
    else:
        print(f"AI COR search failed (exit {rc})")
    return rc


def cmd_ai_full(args) -> int:
    session = _make_session(args, model_required=True)
    rc, cor = H.run_ai_full(args.file, session,
                            seed=args.seed, capture=args.quiet)
    if args.json:
        json.dump({"file": args.file, "exit_code": rc, "cor_used": cor},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if cor is not None:
            print(f"COR = {cor:.2f} → full recon exit {rc}")
        else:
            print(f"AI COR search failed (exit {rc}); full skipped")
    return rc


def cmd_batch(args) -> int:
    if args.data_folder:
        files = _files_from_folder(args.data_folder, args.pattern)
    else:
        files = _files_from_args(args)
    if not files:
        _die("batch: no files selected (use --data-folder, --file, "
             "--files-from, or positional paths)")
    session = _make_session(args, model_required="ai" in args.phases)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    tomolog_kwargs = {}
    if "tomolog" in phases:
        tomolog_kwargs = dict(
            beamline=args.tomolog_beamline,
            cloud=args.tomolog_cloud,
            url=args.tomolog_url or "",
            note=args.tomolog_note,
            auto_contrast_pct=((5.0, 95.0) if args.tomolog_auto_contrast
                                else None),
        )
    results = H.run_batch(files, session,
                          phases=phases,
                          capture=args.quiet,
                          skip_manual_ai=args.skip_manual_ai,
                          tomolog_kwargs=tomolog_kwargs)
    if args.json:
        json.dump([r.__dict__ for r in results], sys.stdout,
                  indent=2, default=str)
        sys.stdout.write("\n")
    else:
        for r in results:
            tag = "OK " if r.exit_code == 0 else "FAIL"
            cor = f"cor={r.cor:.2f}" if r.cor is not None else "cor=-"
            err = f" — {r.error}" if r.error else ""
            print(f"[{tag}] {os.path.basename(r.file)} {r.phase:<7} "
                  f"exit={r.exit_code} {cor}{err}")
    # Non-zero if any phase failed
    return 0 if all(r.exit_code == 0 for r in results) else 1


def cmd_cor_get(args) -> int:
    val = H.get_cor(_infer_data_folder(args.file), args.file)
    if val is None:
        _die(f"no COR recorded for {args.file}", code=1)
    if args.json:
        json.dump({"file": args.file, "cor": val}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"{val:.6g}")
    return 0


def cmd_cor_set(args) -> int:
    data_folder = _infer_data_folder(args.file)
    H.set_cor(data_folder, args.file, args.cor)
    print(f"{args.file}: COR set to {args.cor}", file=sys.stderr)
    return 0


def cmd_cor_list(args) -> int:
    df = args.data_folder
    if not df:
        _die("cor list requires --data-folder")
    data = H.load_cor_data(df)
    if args.json:
        _emit(data, as_json=True)
    else:
        for k, v in data.items():
            print(f"{k}: {v}")
    return 0


def _parse_index_spec(spec: str, n: int) -> list[int]:
    """Turn a slice spec string into a sorted, deduped, in-range index list.

    Accepts:
      * ``A``            → single index
      * ``A,B,C``        → explicit list
      * ``A:B``          → half-open range [A, B)
      * ``A:B:S``        → range with step S
      * ``every:N``      → every Nth slice, offset 0
      * ``mid``          → single mid-slice
      * ``all``          → all slices
    Negative indices count from the end.
    """
    def _norm(i):
        i = int(i)
        if i < 0:
            i += n
        return i

    out: list[int] = []
    spec = spec.strip()
    if spec in ("", "mid"):
        return [n // 2]
    if spec == "all":
        return list(range(n))
    if spec.startswith("every:"):
        step = max(1, int(spec.split(":", 1)[1]))
        return list(range(0, n, step))
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = part.split(":")
            a = _norm(bits[0]) if bits[0] else 0
            b = _norm(bits[1]) if len(bits) > 1 and bits[1] else n
            s = int(bits[2]) if len(bits) > 2 and bits[2] else 1
            out.extend(range(a, b, s))
        else:
            out.append(_norm(part))
    # sort + dedupe + clip
    return sorted({i for i in out if 0 <= i < n})


def cmd_view(args) -> int:
    """Extract slices from any reconstruction (H5 or TIFF) as PNGs, print
    stats, or open an interactive viewer."""
    df = getattr(args, "data_folder", None) or _infer_data_folder(args.target) \
        if os.path.isfile(args.target) else None
    try:
        vol = H.Volume.open(args.target, data_folder=df)
    except FileNotFoundError as exc:
        _die(str(exc), code=1)
    try:
        # Info / stats
        if args.info:
            info = {
                "source": vol.source, "kind": vol.kind,
                "n_slices": vol.n_slices, "shape": list(vol.shape),
            }
            if args.stats:
                info["stats"] = vol.stats(
                    idx=(args.slice if args.slice is not None else None))
            _emit(info, as_json=args.json)
            return 0

        # Interactive Qt slider window
        if args.interactive:
            return _run_interactive_viewer(vol, args)

        # Slice extraction / export
        indices = _parse_index_spec(args.slices or ("mid" if args.slice is None
                                                    else str(args.slice)), vol.n_slices)
        if not indices:
            _die("no slices selected — check --slices / --slice", code=1)

        render_kwargs = dict(
            vmin=args.vmin, vmax=args.vmax, cmap=args.cmap,
            pct=(args.pct_lo, args.pct_hi),
        )

        # Single-slice → single file (or stdout if --out is '-')
        out = args.out
        if len(indices) == 1 and (out is None or not os.path.isdir(out or "")
                                  and (out is None or not out.endswith("/"))):
            arr = vol[indices[0]]
            if out is None:
                # Default: print stats to stderr, PNG to stdout as raw bytes
                _die("--out is required (path or '-' for stdout)", code=2)
            if out == "-":
                # Write PNG bytes to stdout
                from io import BytesIO
                from PIL import Image
                rgb = H.render_slice(arr, **render_kwargs)
                buf = BytesIO()
                Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
                sys.stdout.buffer.write(buf.getvalue())
                return 0
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".",
                        exist_ok=True)
            H.save_slice_png(arr, out, **render_kwargs)
            if not args.quiet:
                print(f"wrote {out} (slice {indices[0]})", file=sys.stderr)
            return 0

        # Multi-slice → directory
        if out is None:
            _die("--out DIR is required when extracting multiple slices",
                 code=2)
        os.makedirs(out, exist_ok=True)
        pad = len(str(vol.n_slices - 1))
        for i in indices:
            fp = os.path.join(out, f"slice_{i:0{pad}d}.png")
            H.save_slice_png(vol[i], fp, **render_kwargs)
        if not args.quiet:
            print(f"wrote {len(indices)} slice(s) to {out}", file=sys.stderr)
        return 0
    finally:
        vol.close()


def _run_interactive_viewer(vol: "H.Volume", args) -> int:
    """Launch the real tomogui GUI and drop the user straight into the
    viewer for this volume. We deliberately reuse the shipped viewer widget
    (VisPy / PyQtGraph, contrast box, cmap picker, slice slider, ROI, save)
    rather than reimplementing a lesser copy in the CLI.
    """
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError as e:
        _die(f"interactive viewer requires PyQt5: {e}", code=1)
    from .gui import TomoGUI

    # Figure out the data folder and highlighted scan the GUI needs to point
    # at. For H5-full-recon or TIFF-full-recon volumes we still need the
    # source projection H5 so the row highlight and View Full trigger work.
    if vol.kind == "h5" and vol.source.endswith("_rec.h5"):
        proj_stem = os.path.basename(vol.source)[:-len("_rec.h5")]
        data_folder = os.path.dirname(os.path.dirname(vol.source)).rstrip("/")
        # strip trailing _rec if data_folder was …_rec
        if data_folder.endswith("_rec"):
            data_folder = data_folder[:-4]
        proj_file = os.path.join(data_folder, f"{proj_stem}.h5")
        view_mode = "full"
    elif vol.kind == "tiff":
        # source is …/{proj}_rec
        parent = os.path.dirname(vol.source)
        proj_stem = os.path.basename(vol.source)[:-len("_rec")] \
            if vol.source.endswith("_rec") else os.path.basename(vol.source)
        data_folder = parent[:-4] if parent.endswith("_rec") else parent
        proj_file = os.path.join(data_folder, f"{proj_stem}.h5")
        view_mode = "full"
    elif vol.kind == "try":
        # source is …/try_center/{proj}
        proj_stem = os.path.basename(vol.source)
        data_folder = os.path.dirname(os.path.dirname(vol.source))
        if data_folder.endswith("_rec"):
            data_folder = data_folder[:-4]
        proj_file = os.path.join(data_folder, f"{proj_stem}.h5")
        view_mode = "try"
    else:
        # Source projection H5 given directly; assume caller wants Full recon.
        data_folder = os.path.dirname(os.path.abspath(vol.source))
        proj_file = vol.source
        view_mode = "full"

    # Close our own H5 handle before the GUI opens the same file.
    vol.close()

    app = QApplication.instance() or QApplication(sys.argv)
    gui = TomoGUI()
    gui.show()
    # Point the GUI at the folder + row, then trigger the built-in viewer.
    try:
        gui.data_path.setText(data_folder)
        # Populate the table (the GUI has a helper triggered by set/browse).
        if hasattr(gui, "read_h5_files"):
            gui.read_h5_files()
        # Highlight the requested scan
        for row, fi in enumerate(getattr(gui, "batch_file_main_list", [])):
            if fi.get("path") == proj_file:
                gui.highlight_scan = proj_file
                gui.highlight_row = row
                break
        if view_mode == "full":
            gui.view_full_reconstruction()
        else:
            gui.view_try_reconstruction()
    except Exception as exc:
        print(f"tomogui-cli: could not auto-drive the GUI ({exc}); the "
              f"window is still open — pick the file manually.",
              file=sys.stderr)
    return app.exec_()


def cmd_tomolog(args) -> int:
    session = _make_session(args)
    rc = H.run_tomolog(
        args.file, session,
        beamline=args.beamline,
        cloud=args.cloud,
        url=args.url or "",
        idx=str(args.idx), idy=str(args.idy), idz=str(args.idz),
        note=args.note,
        vmin=args.vmin, vmax=args.vmax,
        auto_contrast_pct=((5.0, 95.0) if args.auto_contrast else None),
        extra_params=args.extra_params or "",
        capture=args.quiet,
    )
    return rc


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _add_common_recon_args(p, with_data_folder: bool = True):
    if with_data_folder:
        p.add_argument("--data-folder",
                       help="Explicit data folder (default: parent of --file)")
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA_VISIBLE_DEVICES for the tomocupy child")
    p.add_argument("--recon-way", choices=["recon", "recon_steps"],
                   default="recon",
                   help="tomocupy dispatch (default: recon)")
    p.add_argument("--extra", default=None,
                   help='Extra tomocupy args, shell-quoted, e.g. '
                        '"--dezinger 5 --nsino 0.5"')
    p.add_argument("--quiet", action="store_true",
                   help="Capture subprocess output instead of forwarding")


def _add_ai_args(p):
    p.add_argument("--model",
                   help="Path to DINOv2 model weights (.pth). Required for "
                        "AI operations")
    p.add_argument("--ai-search-method", choices=["fine", "full"],
                   default="fine",
                   help="tomocupy --ai-search-method (default: fine)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tomogui-cli",
        description="Headless driver for tomogui / tomocupy reconstructions.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # status
    s = sub.add_parser("status", help="List H5 files with reconstruction status")
    s.add_argument("data_folder", help="Directory of .h5 projection files")
    s.add_argument("--pattern", default="*.h5")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    # try
    s = sub.add_parser("try", help="Try reconstruction on one file")
    s.add_argument("file", help="Path to .h5")
    group = s.add_mutually_exclusive_group()
    group.add_argument("--cor", type=float, help="Manual COR")
    group.add_argument("--auto", action="store_true",
                       help="Let tomocupy pick COR (--rotation-axis-auto auto)")
    _add_common_recon_args(s)
    s.set_defaults(func=cmd_try)

    # full
    s = sub.add_parser("full", help="Full reconstruction on one file")
    s.add_argument("file")
    s.add_argument("--cor", type=float,
                   help="COR to use (defaults to rot_cen.json entry)")
    _add_common_recon_args(s)
    s.set_defaults(func=cmd_full)

    # ai-cor
    s = sub.add_parser("ai-cor",
                       help="Try + AI COR search; persists COR to rot_cen.json")
    s.add_argument("file")
    s.add_argument("--seed", type=float,
                   help="Starting COR seed. Defaults to rot_cen.json entry, "
                        "then to image_width / 2 from the source H5")
    _add_common_recon_args(s)
    _add_ai_args(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ai_cor)

    # ai-full
    s = sub.add_parser("ai-full",
                       help="ai-cor followed by a full recon at the AI COR")
    s.add_argument("file")
    s.add_argument("--seed", type=float)
    _add_common_recon_args(s)
    _add_ai_args(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ai_full)

    # batch
    s = sub.add_parser("batch", help="Pipeline over many files")
    src = s.add_mutually_exclusive_group()
    src.add_argument("--data-folder",
                     help="Folder whose --pattern matches select files")
    src.add_argument("--file", action="append",
                     help="Explicit file (repeatable)")
    src.add_argument("--files-from",
                     help="Text file with one path per line")
    s.add_argument("targets", nargs="*", help="Additional file paths")
    s.add_argument("--pattern", default="*.h5")
    s.add_argument("--phases", default="ai,full",
                   help="Comma-separated: try, ai, full, tomolog "
                        "(default: ai,full)")
    s.add_argument("--skip-manual-ai", action="store_true",
                   help="Skip AI phase for files that already have a COR "
                        "in rot_cen.json (treat existing COR as override)")
    _add_common_recon_args(s, with_data_folder=False)
    _add_ai_args(s)
    # tomolog-specific args (only used if 'tomolog' is in --phases)
    s.add_argument("--tomolog-beamline", default="32-id")
    s.add_argument("--tomolog-cloud", default="imgur",
                   choices=["imgur", "globus", "aps"])
    s.add_argument("--tomolog-url", default=None)
    s.add_argument("--tomolog-note", default=None)
    s.add_argument("--tomolog-auto-contrast", action="store_true",
                   help="Compute 5-95%% percentile per file (else tomolog "
                        "picks defaults)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_batch)

    # cor
    cor_p = sub.add_parser("cor", help="Manage persisted CORs (rot_cen.json)")
    cor_sub = cor_p.add_subparsers(dest="cor_command", required=True)

    cg = cor_sub.add_parser("get", help="Print the COR for one file")
    cg.add_argument("file")
    cg.add_argument("--json", action="store_true")
    cg.set_defaults(func=cmd_cor_get)

    cs = cor_sub.add_parser("set", help="Set/overwrite the COR for one file")
    cs.add_argument("file")
    cs.add_argument("cor", type=float)
    cs.set_defaults(func=cmd_cor_set)

    cl = cor_sub.add_parser("list", help="Dump rot_cen.json for a folder")
    cl.add_argument("--data-folder", required=True)
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=cmd_cor_list)

    # view
    s = sub.add_parser(
        "view",
        help="Show slices of a reconstruction (H5 or TIFF stack). Extract "
             "PNGs headlessly, print stats, or launch the shipped GUI viewer.",
    )
    s.add_argument("target",
                   help="Path to a source projection .h5 (resolves the full "
                        "recon), a full-recon .h5, a TIFF-stack dir, or a "
                        "try_center dir.")
    src = s.add_mutually_exclusive_group()
    src.add_argument("--slice", type=int,
                     help="Single slice index (negative counts from end)")
    src.add_argument("--slices", default=None,
                     help='Slice selector: "A", "A,B,C", "A:B", "A:B:S", '
                          '"every:N", "mid" (default), or "all"')
    s.add_argument("--out", default=None,
                   help="Output PNG path for a single slice, or output "
                        "directory for multiple slices. Use '-' to write "
                        "PNG bytes to stdout (single slice only).")
    s.add_argument("--vmin", type=float, default=None)
    s.add_argument("--vmax", type=float, default=None)
    s.add_argument("--pct-lo", type=float, default=5.0,
                   help="Lower percentile for autoscale (default 5)")
    s.add_argument("--pct-hi", type=float, default=95.0,
                   help="Upper percentile for autoscale (default 95)")
    s.add_argument("--cmap", default="gray",
                   help="Matplotlib colormap (default: gray)")
    s.add_argument("--info", action="store_true",
                   help="Print volume metadata (kind, shape, n_slices) "
                        "instead of extracting")
    s.add_argument("--stats", action="store_true",
                   help="With --info, add min/max/mean/percentiles")
    s.add_argument("--interactive", action="store_true",
                   help="Launch the full tomogui GUI focused on this file "
                        "(reuses the shipped viewer — VisPy/PyQtGraph, "
                        "contrast/cmap controls, ROI, save PNG).")
    s.add_argument("--data-folder", default=None,
                   help="Explicit data folder (only needed when TARGET is a "
                        "source .h5 whose parent isn't the data folder)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_view)

    # tomolog
    s = sub.add_parser("tomolog", help="Upload a reconstruction to tomolog")
    s.add_argument("file")
    s.add_argument("--beamline", default="32-id")
    s.add_argument("--cloud", default="imgur",
                   choices=["imgur", "globus", "aps"])
    s.add_argument("--url", default=None)
    s.add_argument("--idx", type=float, default=-1)
    s.add_argument("--idy", type=float, default=-1)
    s.add_argument("--idz", type=float, default=-1)
    s.add_argument("--note", default=None)
    s.add_argument("--vmin", default=None)
    s.add_argument("--vmax", default=None)
    s.add_argument("--auto-contrast", action="store_true",
                   help="Fill --vmin/--vmax with 5-95%% percentile from "
                        "the reconstruction")
    s.add_argument("--extra-params", default=None,
                   help='Passed through to tomolog, e.g. "--public True"')
    _add_common_recon_args(s)  # for --data-folder / --gpu (env only)
    s.set_defaults(func=cmd_tomolog)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rc = args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"tomogui-cli: {exc}", file=sys.stderr)
        return 3
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
