"""
Tkinter GUI for the Rb beam profiler (run-format data).

Three tabs:
  1. Run Analysis  — pick the data/output folders, adjust fit settings, run the
                     full pipeline with a live log.
  2. Image Browser — click through every stage image the pipeline saved, filter
                     to one probe position, pick a profile point to jump to the
                     side fits behind it, and force a side's accept/reject
                     verdict when the automatic tests got it wrong.
  3. Results       — the final numbers from results.json, formatted.

Only the standard library + PIL (pillow) are needed on top of the analysis
dependencies; the pipeline runs on a worker thread so the window stays
responsive, with log messages marshalled back through a queue.
"""

from __future__ import annotations

import csv
import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..config import AnalysisConfig
from ..side_overrides import load_overrides, save_overrides

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = ImageTk = None

# Pseudo-stages in the browser's stage list, alongside the real output folders.
PICKER_STAGE = "(profile picker)"
FINAL_STAGE = "(final summaries)"
ALL_POSITIONS = "all positions"


def _empty_offsets():
    """An empty (0, 2) offsets array for clearing a scatter highlight."""
    import numpy as np
    return np.empty((0, 2))


def _add_entry(parent, row, label, var, width=12, tip=""):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
    ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="w", padx=4, pady=2)
    if tip:
        ttk.Label(parent, text=tip, foreground="gray").grid(row=row, column=2, sticky="w", padx=4)


def _add_combo(parent, row, label, var, values, tip=""):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
    ttk.Combobox(parent, textvariable=var, values=values, width=12, state="readonly").grid(
        row=row, column=1, sticky="w", padx=4, pady=2)
    if tip:
        ttk.Label(parent, text=tip, foreground="gray").grid(row=row, column=2, sticky="w", padx=4)


class BeamProfilerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rb Beam Profiler")
        self.geometry("1150x780")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._return_to_browser = False   # a run kicked off from the Image Browser

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.tab_run, self.tab_browse, self.tab_results = ttk.Frame(nb), ttk.Frame(nb), ttk.Frame(nb)
        nb.add(self.tab_run, text="  Run Analysis  ")
        nb.add(self.tab_browse, text="  Image Browser  ")
        nb.add(self.tab_results, text="  Results  ")
        self.notebook = nb

        self._build_run_tab()
        self._build_browser_tab()
        self._build_results_tab()
        self.after(150, self._poll_log)

    # ── Tab 1: Run Analysis ────────────────────────────────────────────────────
    def _build_run_tab(self):
        f = self.tab_run
        paths = ttk.LabelFrame(f, text="Data selection")
        paths.pack(fill="x", padx=8, pady=6)

        self.var_data = tk.StringVar()
        self.var_out = tk.StringVar()
        self.var_label = tk.StringVar()

        def picker_row(row, label, var, tip=""):
            ttk.Label(paths, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(paths, textvariable=var, width=80).grid(row=row, column=1, padx=4)
            ttk.Button(paths, text="Browse...",
                       command=lambda: var.set(filedialog.askdirectory() or var.get())
                       ).grid(row=row, column=2, padx=4)
            if tip:
                ttk.Label(paths, text=tip, foreground="gray").grid(row=row, column=3, sticky="w", padx=4)

        picker_row(0, "Data folder (point_*.csv)", self.var_data,
                   tip="acquisition-runner folder of run-format CSVs")
        picker_row(1, "Output folder", self.var_out, tip="all plots + results written here")
        ttk.Label(paths, text="Dataset label").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(paths, textvariable=self.var_label, width=30).grid(row=2, column=1, sticky="w", padx=4)

        mid = ttk.Frame(f)
        mid.pack(fill="x", padx=8, pady=4)

        fit_box = ttk.LabelFrame(mid, text="Fitting")
        fit_box.grid(row=0, column=0, sticky="nsew", padx=4)
        self.var_corner = tk.StringVar(value="0.06")
        self.var_offset = tk.BooleanVar(value=True)
        self.var_dark = tk.StringVar(value="0.0")
        self.var_snr = tk.StringVar(value="3.0")
        self.var_daic = tk.StringVar(value="6.0")
        self.var_gammamax = tk.StringVar(value="2.5")
        self.var_runaway = tk.StringVar(value="0.97")
        self.var_minsides = tk.StringVar(value="1")
        self.var_rejmode = tk.StringVar(value="zero")
        self.var_error = tk.StringVar(value="STD")
        self.var_useover = tk.BooleanVar(value=True)
        r = 0
        _add_entry(fit_box, r, "Corner fraction", self.var_corner,
                   tip="how much of each edge anchors the background line"); r += 1
        ttk.Checkbutton(fit_box, text="Allow offset in peak fit", variable=self.var_offset
                        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        _add_entry(fit_box, r, "Dark offset [V]", self.var_dark,
                   tip="photodiode dark level subtracted from CH2"); r += 1
        _add_entry(fit_box, r, "Min SNR", self.var_snr, tip="reject side fits below this"); r += 1
        _add_entry(fit_box, r, "Min dAIC", self.var_daic,
                   tip="peak must beat noise-only by this"); r += 1
        # The two runaway knobs: where the linewidth ceiling sits, and how close
        # to it a fit has to crawl before it counts as having run away.
        _add_entry(fit_box, r, "Max linewidth (xref)", self.var_gammamax,
                   tip="upper HWHM bound, x the reference linewidth"); r += 1
        _add_entry(fit_box, r, "Runaway sensitivity", self.var_runaway,
                   tip="reject at HWHM >= this x the max (1.0 = only exactly pinned, 0 = off)"); r += 1
        _add_entry(fit_box, r, "Min good sides", self.var_minsides,
                   tip="fewer -> position flagged (red)"); r += 1
        _add_combo(fit_box, r, "Rejected sides", self.var_rejmode, ["zero", "exclude", "keep"],
                   "zero = count a failed side as 0 and average it in"); r += 1
        _add_combo(fit_box, r, "Point error", self.var_error, ["STD", "SEM", "upper_bound"],
                   "STD = side-to-side scatter (thesis choice)"); r += 1
        ttk.Checkbutton(fit_box, text="Honour manual side verdicts", variable=self.var_useover
                        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Label(fit_box, text="set in the Image Browser; uncheck for a purely automatic run",
                  foreground="gray").grid(row=r, column=2, sticky="w", padx=4)

        geo_box = ttk.LabelFrame(mid, text="Geometry / physics")
        geo_box.grid(row=0, column=1, sticky="nsew", padx=4)
        self.var_zprobe = tk.StringVar(value="2.46")
        self.var_zsub = tk.StringVar(value="19.685")
        self.var_nozzle = tk.StringVar(value="0.1")
        self.var_aperture = tk.StringVar(value="0.5")
        self.var_temp = tk.StringVar(value="230")
        self.var_arflow = tk.StringVar(value="")
        _add_entry(geo_box, 0, "Nozzle->probe [in]", self.var_zprobe)
        _add_entry(geo_box, 1, "Nozzle->substrate [in]", self.var_zsub)
        _add_entry(geo_box, 2, "Nozzle diameter [in]", self.var_nozzle)
        _add_entry(geo_box, 3, "Aperture radius [cm]", self.var_aperture)
        _add_entry(geo_box, 4, "Oven temp [C]", self.var_temp)
        _add_entry(geo_box, 5, "Ar flow [SCCM]", self.var_arflow, tip="blank = skip dopant density")

        model_box = ttk.LabelFrame(mid, text="Profile models")
        model_box.grid(row=0, column=2, sticky="nsew", padx=4)
        self.var_gauss = tk.BooleanVar(value=True)
        self.var_lor = tk.BooleanVar(value=True)
        self.var_pv = tk.BooleanVar(value=False)
        self.var_poffset = tk.BooleanVar(value=False)
        ttk.Checkbutton(model_box, text="Gaussian", variable=self.var_gauss).pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(model_box, text="Lorentzian", variable=self.var_lor).pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(model_box, text="Pseudo-Voigt", variable=self.var_pv).pack(anchor="w", padx=6, pady=2)
        ttk.Label(model_box, text="(pseudo-Voigt's eta tells you\nGaussian vs Lorentzian directly)",
                  foreground="gray").pack(anchor="w", padx=6)
        ttk.Checkbutton(model_box, text="Offset in profile fit",
                        variable=self.var_poffset).pack(anchor="w", padx=6, pady=6)

        btns = ttk.Frame(f)
        btns.pack(fill="x", padx=8, pady=4)
        self.btn_run = ttk.Button(btns, text="▶  Run Full Pipeline", command=self._on_run)
        self.btn_run.pack(side="left", padx=4)
        ttk.Button(btns, text="Save Settings...", command=self._save_config).pack(side="left", padx=4)
        ttk.Button(btns, text="Load Settings...", command=self._load_config).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(btns, mode="indeterminate", length=220)
        self.progress.pack(side="right", padx=8)

        log_frame = ttk.LabelFrame(f, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=6)
        self.log = tk.Text(log_frame, height=14, state="disabled", font=("Consolas", 9), wrap="none")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

    def _collect_config(self) -> AnalysisConfig:
        cfg = AnalysisConfig()
        cfg.corner_cluster_fraction = float(self.var_corner.get())
        cfg.fit_peak_offset = self.var_offset.get()
        cfg.dark_offset_V = float(self.var_dark.get() or 0.0)
        cfg.snr_min = float(self.var_snr.get())
        cfg.delta_aic_min = float(self.var_daic.get())
        cfg.gamma_rel_bounds = (cfg.gamma_rel_bounds[0], float(self.var_gammamax.get()))
        cfg.linewidth_runaway_fraction = float(self.var_runaway.get())
        cfg.use_manual_overrides = self.var_useover.get()
        cfg.min_good_sides = int(self.var_minsides.get())
        cfg.rejected_mode = self.var_rejmode.get()
        cfg.error_mode = self.var_error.get()
        cfg.z_probe_in = float(self.var_zprobe.get())
        cfg.z_substrate_in = float(self.var_zsub.get())
        cfg.nozzle_diameter_in = float(self.var_nozzle.get())
        cfg.aperture_radius_cm = float(self.var_aperture.get())
        cfg.oven_temp_C = float(self.var_temp.get())
        cfg.ar_flow_sccm = float(self.var_arflow.get()) if self.var_arflow.get().strip() else None
        models = []
        if self.var_gauss.get():
            models.append("gaussian")
        if self.var_lor.get():
            models.append("lorentzian")
        if self.var_pv.get():
            models.append("pseudo_voigt")
        cfg.profile_models = models or ["gaussian"]
        cfg.profile_offset = self.var_poffset.get()
        return cfg

    def _apply_config(self, cfg: AnalysisConfig):
        self.var_corner.set(str(cfg.corner_cluster_fraction))
        self.var_offset.set(cfg.fit_peak_offset)
        self.var_dark.set(str(cfg.dark_offset_V))
        self.var_snr.set(str(cfg.snr_min))
        self.var_daic.set(str(cfg.delta_aic_min))
        self.var_gammamax.set(str(cfg.gamma_rel_bounds[1]))
        self.var_runaway.set(str(cfg.linewidth_runaway_fraction))
        self.var_useover.set(cfg.use_manual_overrides)
        self.var_minsides.set(str(cfg.min_good_sides))
        self.var_rejmode.set(cfg.rejected_mode)
        self.var_error.set(cfg.error_mode)
        self.var_zprobe.set(str(cfg.z_probe_in))
        self.var_zsub.set(str(cfg.z_substrate_in))
        self.var_nozzle.set(str(cfg.nozzle_diameter_in))
        self.var_aperture.set(str(cfg.aperture_radius_cm))
        self.var_temp.set(str(cfg.oven_temp_C))
        self.var_arflow.set("" if cfg.ar_flow_sccm is None else str(cfg.ar_flow_sccm))
        self.var_gauss.set("gaussian" in cfg.profile_models)
        self.var_lor.set("lorentzian" in cfg.profile_models)
        self.var_pv.set("pseudo_voigt" in cfg.profile_models)
        self.var_poffset.set(cfg.profile_offset)

    def _save_config(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            self._collect_config().save(p)
            self._log_line(f"Settings saved to {p}")

    def _load_config(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if p:
            self._apply_config(AnalysisConfig.load(p))
            self._log_line(f"Settings loaded from {p}")

    def _on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        data, out = self.var_data.get().strip(), self.var_out.get().strip()
        if not data or not out:
            messagebox.showerror("Missing input", "Select both a data folder and an output folder.")
            return
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Bad setting", f"Could not parse a setting: {exc}")
            return
        label = self.var_label.get().strip() or Path(data).name

        self.btn_run.configure(state="disabled")
        self.progress.start(12)
        self._log_line("=" * 70)

        def work():
            from ..pipeline import run_full_pipeline
            try:
                run_full_pipeline(data, out, cfg, progress=self.log_queue.put, label=label)
                self.log_queue.put("__DONE__")
            except Exception as exc:  # noqa: BLE001
                self.log_queue.put(f"ERROR: {exc}")
                self.log_queue.put("__FAIL__")

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg in ("__DONE__", "__FAIL__"):
                    self.progress.stop()
                    self.btn_run.configure(state="normal")
                    if msg == "__DONE__":
                        self._log_line("Pipeline finished.")
                        self._refresh_browser()
                        self._refresh_results()
                        # A run launched from the Image Browser (e.g. to apply a
                        # manual verdict) returns you there, not to Results, so
                        # you can see the verdict took effect.
                        if self._return_to_browser:
                            self.notebook.select(self.tab_browse)
                        else:
                            self.notebook.select(self.tab_results)
                    self._return_to_browser = False
                else:
                    self._log_line(msg)
        except queue.Empty:
            pass
        self.after(150, self._poll_log)

    def _log_line(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ── Tab 2: Image Browser ───────────────────────────────────────────────────
    # Three things live here beyond "look at the PNGs":
    #   * a POSITION FILTER, so the 40-odd side images of a 13-point run can be
    #     narrowed to the one probe position you are questioning;
    #   * the PROFILE PICKER, a live plot of the profile where clicking a point
    #     takes you straight to the side fits that produced it;
    #   * the VERDICT BAR, which forces a side to accept/reject when the
    #     automatic tests got it wrong (written to manual_overrides.json and
    #     applied on the next run — see beamprofiler/side_overrides.py).
    def _build_browser_tab(self):
        f = self.tab_browse
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Stage:").pack(side="left")
        self.var_stage = tk.StringVar()
        self.combo_stage = ttk.Combobox(top, textvariable=self.var_stage, width=26, state="readonly")
        self.combo_stage.pack(side="left", padx=6)
        self.combo_stage.bind("<<ComboboxSelected>>", lambda e: self._load_stage_images())
        ttk.Button(top, text="Refresh", command=self._refresh_browser).pack(side="left", padx=4)
        ttk.Button(top, text="◀ Prev", command=lambda: self._step_image(-1)).pack(side="left", padx=(20, 2))
        ttk.Button(top, text="Next ▶", command=lambda: self._step_image(1)).pack(side="left", padx=2)
        ttk.Label(top, text="Position:").pack(side="left", padx=(24, 2))
        self.var_posfilter = tk.StringVar(value=ALL_POSITIONS)
        self.combo_pos = ttk.Combobox(top, textvariable=self.var_posfilter, width=18, state="readonly")
        self.combo_pos.pack(side="left", padx=4)
        self.combo_pos.bind("<<ComboboxSelected>>", lambda e: self._load_stage_images())

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        self.listbox = tk.Listbox(body, width=40, font=("Consolas", 9), exportselection=False)
        self.listbox.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._show_selected())
        self.listbox.bind("<Double-Button-1>", lambda e: self._on_list_double_click())

        self.display = ttk.Frame(body)
        self.display.pack(side="left", fill="both", expand=True)
        self.img_label = ttk.Label(self.display, anchor="center",
                                   text="Run the pipeline (or Refresh) to browse images.")
        self.img_label.pack(fill="both", expand=True)
        self.img_label.bind("<Configure>", lambda e: self._show_selected(redraw_only=True))

        self._build_verdict_bar(f)

        self._current_photo = None
        self._stage_files: list[Path] = []
        self._picker_widget = None          # FigureCanvasTkAgg widget, built lazily
        self._picker_canvas = None
        self._picker_fig = None
        self._picker_ax = None
        self._picker_marker = None
        self._picker_shown = False
        self._picker_points = None          # (positions, amplitudes) in data coords
        self._profile_rows: list[dict] = []  # absorption_profile.csv
        self._side_rows: dict[str, dict] = {}   # image filename -> side_details.csv row
        self._point_stems: dict[float, list[str]] = {}   # position -> file stems
        self._overrides: dict[str, str] = {}

    def _build_verdict_bar(self, parent):
        bar = ttk.LabelFrame(parent, text="Side verdict (02_side_fits)")
        bar.pack(fill="x", padx=8, pady=(0, 8))
        self.lbl_verdict = ttk.Label(
            bar, text="Select a side image to accept or reject it by hand.",
            font=("Consolas", 9))
        self.lbl_verdict.grid(row=0, column=0, columnspan=5, sticky="w", padx=6, pady=(4, 2))
        self.btn_accept = ttk.Button(bar, text="Force ACCEPT",
                                     command=lambda: self._set_override("accept"))
        self.btn_accept.grid(row=1, column=0, padx=6, pady=4)
        self.btn_reject = ttk.Button(bar, text="Force REJECT",
                                     command=lambda: self._set_override("reject"))
        self.btn_reject.grid(row=1, column=1, padx=6, pady=4)
        self.btn_auto = ttk.Button(bar, text="Use automatic",
                                   command=lambda: self._set_override(None))
        self.btn_auto.grid(row=1, column=2, padx=6, pady=4)
        self.lbl_overrides = ttk.Label(bar, text="", foreground="#762a83")
        self.lbl_overrides.grid(row=1, column=3, sticky="w", padx=16)
        ttk.Button(bar, text="Re-run pipeline (applies verdicts)",
                   command=self._rerun_from_browser).grid(row=1, column=4, padx=6, sticky="e")
        bar.columnconfigure(3, weight=1)
        self._set_verdict_buttons(False)

    def _set_verdict_buttons(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in (self.btn_accept, self.btn_reject, self.btn_auto):
            b.configure(state=state)

    # ── loading the browsable state ───────────────────────────────────────────
    def _refresh_browser(self):
        out = self.var_out.get().strip()
        stages: list[str] = []
        keep_stage, keep_pos = self.var_stage.get(), self.var_posfilter.get()
        if out and Path(out).is_dir():
            root = Path(out)
            self._read_run_tables(root)
            self._overrides = load_overrides(root)
            for sub in sorted(root.rglob("*")):
                if sub.is_dir() and any(sub.glob("*.png")):
                    stages.append(str(sub.relative_to(root)))
            if any(root.glob("*.png")):
                stages.insert(0, FINAL_STAGE)
            if self._profile_rows:
                stages.insert(0, PICKER_STAGE)
        self.combo_stage["values"] = stages
        positions = [ALL_POSITIONS] + [f"{r['position']:g} thou" for r in self._profile_rows]
        self.combo_pos["values"] = positions
        if keep_pos not in positions:
            self.var_posfilter.set(ALL_POSITIONS)
        if stages:
            self.var_stage.set(keep_stage if keep_stage in stages else stages[0])
            self._load_stage_images()
        self._update_override_count()

    def _read_run_tables(self, root: Path):
        """Read the two CSVs the browser needs: the profile and the side details.

        Read with the csv module rather than pandas — they are small, and the GUI
        should still open if the analysis dependencies are half-installed.
        """
        self._profile_rows, self._side_rows, self._point_stems = [], {}, {}
        prof = root / "03_profile" / "absorption_profile.csv"
        if prof.exists():
            with open(prof, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        self._profile_rows.append({
                            "position": float(row["position"]),
                            "amplitude": float(row["amplitude"]),
                            "amplitude_err": float(row["amplitude_err"]),
                            "flagged": str(row.get("flagged", "")).lower() == "true",
                            "n_good_sides": int(float(row.get("n_good_sides", 0))),
                            "n_sides": int(float(row.get("n_sides", 0))),
                            "n_overridden": int(float(row.get("n_overridden", 0) or 0)),
                        })
                    except (ValueError, KeyError):
                        continue
            self._profile_rows.sort(key=lambda r: r["position"])

        details = root / "03_profile" / "side_details.csv"
        if details.exists():
            with open(details, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    stem = row.get("side", "")
                    # 'image' was added later; older result folders lack it.
                    name = row.get("image") or (stem.replace(":", "_") + ".png")
                    self._side_rows[name] = row
                    try:
                        pos = float(row["position"])
                    except (ValueError, KeyError):
                        continue
                    point = stem.split(":")[0]
                    stems = self._point_stems.setdefault(pos, [])
                    if point not in stems:
                        stems.append(point)

    def _filter_position(self) -> float | None:
        """The position currently selected in the filter combo, if any."""
        text = self.var_posfilter.get()
        if not text or text == ALL_POSITIONS:
            return None
        try:
            return float(text.split()[0])
        except ValueError:
            return None

    def _load_stage_images(self):
        out = Path(self.var_out.get().strip() or ".")
        stage = self.var_stage.get()
        self.listbox.delete(0, "end")

        if stage == PICKER_STAGE:
            self._stage_files = []
            for r in self._profile_rows:
                mark = "!" if r["flagged"] else " "
                self.listbox.insert(
                    "end", f"{mark}{r['position']:>7g} thou  A={r['amplitude']:.4f}%  "
                           f"{r['n_good_sides']}/{r['n_sides']}")
            self._show_picker()
            return

        folder = out if stage == FINAL_STAGE else out / stage
        files = sorted(folder.glob("*.png"))
        pos = self._filter_position()
        if pos is not None and stage in ("01_raw_traces", "02_side_fits"):
            stems = self._point_stems.get(pos, [])
            files = [p for p in files
                     if any(p.stem == s or p.stem.startswith(s + "_") for s in stems)]
        self._stage_files = files
        for p in files:
            self.listbox.insert("end", self._list_label(p))
        if files:
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(0)
        self._show_selected()

    def _list_label(self, path: Path) -> str:
        """File name, prefixed with its verdict when it is a side image."""
        row = self._side_rows.get(path.name)
        if row is None:
            return path.name
        if self._overrides.get(row.get("side", "")):
            mark = "M"          # verdict set by hand, not yet re-run
        elif str(row.get("accepted", "")).lower() == "true":
            mark = "+"
        else:
            mark = "-"
        return f"{mark} {path.name}"

    def _step_image(self, delta: int):
        n = self.listbox.size()
        if not n:
            return
        sel = self.listbox.curselection()
        i = max(0, min(n - 1, (sel[0] if sel else 0) + delta))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(i)
        self.listbox.see(i)
        self._show_selected()

    def _on_list_double_click(self):
        """In the picker, double-clicking a position jumps to its side fits."""
        if self.var_stage.get() != PICKER_STAGE:
            return
        sel = self.listbox.curselection()
        if sel and sel[0] < len(self._profile_rows):
            self._goto_position_sides(self._profile_rows[sel[0]]["position"])

    # ── display: either a saved PNG or the interactive picker ─────────────────
    def _show_selected(self, redraw_only: bool = False):
        if self.var_stage.get() == PICKER_STAGE:
            self._highlight_picker_selection()
            self._update_verdict_bar(None)
            return
        self._show_picker(hide=True)
        sel = self.listbox.curselection()
        if not self._stage_files or not sel:
            self._update_verdict_bar(None)
            return
        path = self._stage_files[sel[0]]
        self._update_verdict_bar(path)
        if Image is None:
            self.img_label.configure(text=f"{path.name}\n(install pillow to view images here)")
            return
        try:
            img = Image.open(path)
        except OSError:
            return
        w = max(self.img_label.winfo_width(), 50)
        h = max(self.img_label.winfo_height(), 50)
        scale = min(w / img.width, h / img.height, 1.0)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        self._current_photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self._current_photo, text="")

    def _show_picker(self, hide: bool = False):
        """Draw (or hide) the clickable profile.

        Built with a bare Figure + FigureCanvasTkAgg rather than pyplot: the
        pipeline's own figures are made on a worker thread through the Agg
        backend, and the two must not share pyplot's figure manager.
        """
        if hide:
            if self._picker_shown:
                if self._picker_widget is not None:
                    self._picker_widget.pack_forget()
                self.img_label.pack(fill="both", expand=True)
                self._picker_shown = False
            return

        if not self._picker_shown:
            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                from matplotlib.figure import Figure
            except ImportError:
                self.img_label.configure(
                    image="", text="matplotlib is needed for the profile picker.")
                return
            if self._picker_widget is None:
                self._picker_fig = Figure(figsize=(6.5, 4.6), dpi=100)
                self._picker_canvas = FigureCanvasTkAgg(self._picker_fig, master=self.display)
                self._picker_canvas.mpl_connect("button_press_event", self._on_picker_click)
                self._picker_widget = self._picker_canvas.get_tk_widget()
            self.img_label.pack_forget()
            self._picker_widget.pack(fill="both", expand=True)
            self._picker_shown = True
        self._draw_picker()

    def _draw_picker(self):
        fig = self._picker_fig
        fig.clear()
        ax = fig.add_subplot(111)
        rows = self._profile_rows
        if not rows:
            ax.text(0.5, 0.5, "No absorption_profile.csv yet — run the pipeline.",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            self._picker_canvas.draw_idle()
            return

        x = [r["position"] for r in rows]
        y = [r["amplitude"] for r in rows]
        e = [r["amplitude_err"] for r in rows]
        colors = ["#d7301f" if r["flagged"] else "#2c7fb8" for r in rows]
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor="#9ecae1", capsize=3, zorder=1)
        ax.scatter(x, y, c=colors, s=45, zorder=3)
        # Ring the positions that already carry a hand-set verdict.
        marked = [(r["position"], r["amplitude"]) for r in rows if r["n_overridden"]]
        if marked:
            ax.scatter([p for p, _ in marked], [a for _, a in marked], s=150,
                       facecolors="none", edgecolors="#762a83", linewidths=1.8, zorder=2,
                       label="has manual verdict")
            ax.legend(fontsize=8, loc="best")
        self._picker_marker = ax.scatter([], [], s=220, facecolors="none",
                                         edgecolors="#31a354", linewidths=2.2, zorder=4)
        ax.set_xlabel("Stage position [thou]")
        ax.set_ylabel("Absorption [%]")
        ax.set_title("Click a point to open its side fits  (red = noise-dominated)",
                     fontsize=9, color="gray")
        ax.grid(ls="--", alpha=0.4)
        fig.tight_layout()
        self._picker_points = (x, y)
        self._picker_ax = ax
        self._highlight_picker_selection()

    def _highlight_picker_selection(self):
        """Ring the position highlighted in the list, so list and plot agree."""
        if self._picker_widget is None or self._picker_points is None:
            return
        sel = self.listbox.curselection()
        x, y = self._picker_points
        if sel and sel[0] < len(x):
            self._picker_marker.set_offsets([[x[sel[0]], y[sel[0]]]])
        else:
            self._picker_marker.set_offsets(_empty_offsets())
        self._picker_canvas.draw_idle()

    def _on_picker_click(self, event):
        """Nearest-point pick, in pixels so both axes count equally."""
        if event.inaxes is None or not self._picker_points:
            return
        x, y = self._picker_points
        best_i, best_d = None, 1e9
        for i, (xi, yi) in enumerate(zip(x, y)):
            px, py = self._picker_ax.transData.transform((xi, yi))
            d = ((px - event.x) ** 2 + (py - event.y) ** 2) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        if best_i is None or best_d > 40:      # 40 px — a miss, not a pick
            return
        self._goto_position_sides(self._profile_rows[best_i]["position"])

    def _goto_position_sides(self, position: float):
        """Filter to one probe position and show its 02_side_fits images."""
        self.var_posfilter.set(f"{position:g} thou")
        stages = list(self.combo_stage["values"])
        if "02_side_fits" in stages:
            self.var_stage.set("02_side_fits")
        self._load_stage_images()

    # ── manual accept/reject ──────────────────────────────────────────────────
    def _selected_side_stem(self) -> str | None:
        """The side label ('point_0004:s3') of the selected image, if it is one."""
        sel = self.listbox.curselection()
        if not sel or not self._stage_files or sel[0] >= len(self._stage_files):
            return None
        row = self._side_rows.get(self._stage_files[sel[0]].name)
        return row.get("side") if row else None

    def _update_verdict_bar(self, path: Path | None):
        stem = self._selected_side_stem() if path is not None else None
        if stem is None:
            self.lbl_verdict.configure(
                text="Select a side image (02_side_fits) to accept or reject it by hand.")
            self._set_verdict_buttons(False)
            return
        row = self._side_rows.get(path.name, {})
        auto = str(row.get("auto_accepted", row.get("accepted", ""))).lower() == "true"
        current = self._overrides.get(stem)
        # What the automatic tests said, then what will actually be used. The
        # two are shown separately on purpose: an override should never make it
        # look as though the fit passed on its own.
        line = (f"{stem}   auto: {'ACCEPT' if auto else 'REJECT'}"
                f"   SNR {float(row.get('snr', 0) or 0):.1f}"
                f"   dAIC {float(row.get('delta_aic', 0) or 0):.0f}"
                f"   HWHM {float(row.get('gamma_us', 0) or 0):.0f} us"
                f" ({float(row.get('gamma_pinned_frac', 0) or 0):.2f} of bound)")
        reasons = row.get("reject_reasons", "")
        if reasons:
            line += f"\n   {reasons}"
        line += ("\n   -> manual verdict: NONE (automatic)" if current is None
                 else f"\n   -> manual verdict: {current.upper()} (applied on the next run)")
        self.lbl_verdict.configure(text=line)
        self._set_verdict_buttons(True)

    def _set_override(self, verdict: str | None):
        stem = self._selected_side_stem()
        if stem is None:
            return
        out = self.var_out.get().strip()
        if not out:
            messagebox.showerror("No output folder", "Set the output folder first.")
            return
        if verdict is None:
            self._overrides.pop(stem, None)
        else:
            self._overrides[stem] = verdict
        save_overrides(out, self._overrides)
        self._log_line(f"Manual verdict: {stem} -> "
                       f"{verdict.upper() if verdict else 'automatic'} "
                       f"(saved; re-run the pipeline to apply)")
        sel = self.listbox.curselection()
        if sel and sel[0] < len(self._stage_files):
            i = sel[0]
            self.listbox.delete(i)
            self.listbox.insert(i, self._list_label(self._stage_files[i]))
            self.listbox.selection_set(i)
        self._update_override_count()
        self._show_selected()

    def _update_override_count(self):
        n = len(self._overrides)
        self.lbl_overrides.configure(
            text="" if not n else f"{n} manual verdict(s) pending — re-run to apply")

    def _rerun_from_browser(self):
        if not self.var_useover.get():
            messagebox.showwarning(
                "Manual verdicts off",
                "'Honour manual side verdicts' is unchecked on the Run Analysis tab, "
                "so this run will ignore them.")
        self._return_to_browser = True
        self.notebook.select(self.tab_run)
        self._on_run()

    # ── Tab 3: Results ─────────────────────────────────────────────────────────
    def _build_results_tab(self):
        f = self.tab_results
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Refresh", command=self._refresh_results).pack(side="left")
        ttk.Button(top, text="Open Output Folder", command=self._open_output).pack(side="left", padx=6)
        self.results_text = tk.Text(f, font=("Consolas", 10), state="disabled", wrap="none")
        self.results_text.pack(fill="both", expand=True, padx=8, pady=6)

    def _open_output(self):
        out = self.var_out.get().strip()
        if out and Path(out).is_dir():
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(Path(out))])
            else:
                subprocess.Popen(["xdg-open", str(Path(out))])

    def _refresh_results(self):
        out = self.var_out.get().strip()
        path = Path(out) / "results.json" if out else None
        text = "No results.json found yet — run the pipeline first."
        if path and path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
            lines = [f"Dataset: {r.get('label', '')}", ""]
            if r.get("zone1"):
                lines.append("--- PROFILE (Method A) " + "-" * 42)
                for model, d in r["zone1"].items():
                    lines.append(f"  {model}:")
                    lines.append(f"    FWHM          = {d['fwhm_mm']:.2f} +- {d['fwhm_mm_err']:.2f} mm")
                    lines.append(f"    theta_1/2     = {d['theta_half_deg']:.3f} +- {d['theta_half_deg_err']:.3f} deg")
                    lines.append(f"    capture eff.  = {d['capture_eff']*100:.1f} +- {d['capture_eff_err']*100:.1f} %")
                    lines.append(f"    reduced chi2  = {d['chi2_reduced']:.2f}   R^2 = {d['r_squared']:.3f}")
                    if d.get("eta") is not None:
                        lines.append(f"    eta (0=G,1=L) = {d['eta']:.2f}")
                lines.append("")
            if r.get("flux"):
                lines.append("--- FLUX " + "-" * 55)
                for model, d in r["flux"].items():
                    lines.append(f"  {model}:")
                    lines.append(f"    peak absorp.  = {d['absorption_pct']:.3f} %")
                    lines.append(f"    column dens.  = {d['n_col_cm2']:.3e} cm^-2")
                    lines.append(f"    Rb flux       = {d['rb_flux_atoms_s']:.3e} atoms/s")
                    if d.get("dopant_density_cm3") is not None:
                        lines.append(f"    dopant dens.  = {d['dopant_density_cm3']:.3e} cm^-3")
                lines.append("")
            text = "\n".join(lines)
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")


def main():
    BeamProfilerApp().mainloop()


if __name__ == "__main__":
    main()
