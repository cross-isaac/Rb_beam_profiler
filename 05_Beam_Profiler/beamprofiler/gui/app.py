"""
Tkinter GUI for the Rb beam profiler.

Three tabs:
  1. Run Analysis  — pick the data/flux/output folders, adjust fit settings,
                     run the full pipeline with a live log.
  2. Image Browser — click through every stage image the pipeline saved
                     (raw traces, per-side fits, profiles, flux, summaries).
  3. Results       — the final numbers from results.json, nicely formatted.

Only the standard library + PIL (pillow) are needed on top of the analysis
dependencies; the pipeline itself runs on a worker thread so the window stays
responsive, with log messages marshalled back through a queue.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..config import AnalysisConfig

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = ImageTk = None


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers for building labelled setting rows
# ──────────────────────────────────────────────────────────────────────────────
def _add_entry(parent, row, label, var, width=12, tip=""):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
    e = ttk.Entry(parent, textvariable=var, width=width)
    e.grid(row=row, column=1, sticky="w", padx=4, pady=2)
    if tip:
        ttk.Label(parent, text=tip, foreground="gray").grid(
            row=row, column=2, sticky="w", padx=4)
    return e


def _add_combo(parent, row, label, var, values, tip=""):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
    c = ttk.Combobox(parent, textvariable=var, values=values, width=10, state="readonly")
    c.grid(row=row, column=1, sticky="w", padx=4, pady=2)
    if tip:
        ttk.Label(parent, text=tip, foreground="gray").grid(
            row=row, column=2, sticky="w", padx=4)
    return c


class BeamProfilerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rb Beam Profiler")
        self.geometry("1150x780")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.tab_run = ttk.Frame(nb)
        self.tab_browse = ttk.Frame(nb)
        self.tab_results = ttk.Frame(nb)
        nb.add(self.tab_run, text="  Run Analysis  ")
        nb.add(self.tab_browse, text="  Image Browser  ")
        nb.add(self.tab_results, text="  Results  ")
        self.notebook = nb

        self._build_run_tab()
        self._build_browser_tab()
        self._build_results_tab()
        self.after(150, self._poll_log)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 1 — Run Analysis
    # ──────────────────────────────────────────────────────────────────────────
    def _build_run_tab(self):
        f = self.tab_run

        # ── folder pickers ────────────────────────────────────────────────────
        paths = ttk.LabelFrame(f, text="Data selection")
        paths.pack(fill="x", padx=8, pady=6)

        self.var_data = tk.StringVar()
        self.var_flux = tk.StringVar()
        self.var_zone2 = tk.StringVar()
        self.var_out = tk.StringVar()
        self.var_label = tk.StringVar()

        def picker_row(row, label, var, is_dir=True, optional=False, tip=""):
            text = label + (" (optional)" if optional else "")
            ttk.Label(paths, text=text).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(paths, textvariable=var, width=80).grid(row=row, column=1, padx=4)

            def browse():
                p = (filedialog.askdirectory() if is_dir
                     else filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")]))
                if p:
                    var.set(p)
            ttk.Button(paths, text="Browse...", command=browse).grid(row=row, column=2, padx=4)
            if tip:
                ttk.Label(paths, text=tip, foreground="gray").grid(
                    row=row, column=3, sticky="w", padx=4)

        picker_row(0, "Data folder (position CSVs)", self.var_data,
                   tip="run folder (point_0001.csv ...) or legacy 000.csv ... 900.csv")
        picker_row(1, "Flux trace file or folder", self.var_flux, is_dir=False, optional=True,
                   tip="legacy calibration trace — run-format data computes flux from the profile")
        picker_row(2, "Zone 2 folder (Method B)", self.var_zone2, optional=True,
                   tip="second-distance scan for two-zone divergence")
        picker_row(3, "Output folder", self.var_out,
                   tip="all plots + results are written here")
        ttk.Label(paths, text="Dataset label").grid(row=4, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(paths, textvariable=self.var_label, width=30).grid(
            row=4, column=1, sticky="w", padx=4)

        # ── settings ──────────────────────────────────────────────────────────
        mid = ttk.Frame(f)
        mid.pack(fill="x", padx=8, pady=4)

        fit_box = ttk.LabelFrame(mid, text="Fitting")
        fit_box.grid(row=0, column=0, sticky="nsew", padx=4)
        self.var_degree = tk.StringVar(value="auto")
        self.var_offset = tk.BooleanVar(value=True)
        self.var_error = tk.StringVar(value="STD")
        self.var_seg = tk.StringVar(value="sync")
        self.var_snr = tk.StringVar(value="3.0")
        self.var_daic = tk.StringVar(value="6.0")
        self.var_minsides = tk.StringVar(value="2")
        self.var_rejmode = tk.StringVar(value="zero")
        self.var_autotrim = tk.BooleanVar(value=True)
        self.var_bgmethod = tk.StringVar(value="corners")
        self.var_absmode = tk.StringVar(value="auto")
        self.var_dark = tk.StringVar(value="0.0")
        _add_combo(fit_box, 0, "Ramp poly degree", self.var_degree, ["auto", "1", "2"],
                   "auto: 1 for raw run data, 2 for legacy amplified data")
        r = 1
        cb = ttk.Checkbutton(fit_box, text="Allow offset in peak fit", variable=self.var_offset)
        cb.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        cb2 = ttk.Checkbutton(fit_box, text="Auto-trim curved ramp ends", variable=self.var_autotrim)
        cb2.grid(row=r, column=0, columnspan=2, sticky="w", padx=4, pady=2); r += 1
        _add_combo(fit_box, r, "Background fit", self.var_bgmethod,
                   ["corners", "wings", "masked"],
                   "corners = line through turn-point clusters (recommended)"); r += 1
        _add_combo(fit_box, r, "Absorption scale", self.var_absmode,
                   ["auto", "absolute", "relative"],
                   "auto = absolute (CH3/CH2) when a DC channel exists"); r += 1
        _add_entry(fit_box, r, "Dark offset [V]", self.var_dark,
                   tip="photodiode dark level subtracted from CH2"); r += 1
        _add_combo(fit_box, r, "Point error", self.var_error, ["STD", "SEM"],
                   "STD = side-to-side scatter (thesis choice)"); r += 1
        _add_combo(fit_box, r, "Segmentation", self.var_seg, ["sync", "auto"],
                   "sync uses the CH4 square wave for the sweep period"); r += 1
        _add_entry(fit_box, r, "Min SNR", self.var_snr, tip="reject peaks below this"); r += 1
        _add_entry(fit_box, r, "Min dAIC", self.var_daic,
                   tip="peak model must beat noise-only by this"); r += 1
        _add_entry(fit_box, r, "Min good sides", self.var_minsides,
                   tip="position rejected below this"); r += 1
        _add_combo(fit_box, r, "Rejected points", self.var_rejmode,
                   ["zero", "exclude", "keep"],
                   "zero = amplitude 0 with upper-bound error (recommended)")

        geo_box = ttk.LabelFrame(mid, text="Geometry / physics")
        geo_box.grid(row=0, column=1, sticky="nsew", padx=4)
        self.var_zprobe = tk.StringVar(value="2.46")
        self.var_zprobe2 = tk.StringVar(value="12.3")
        self.var_zsub = tk.StringVar(value="19.685")
        self.var_nozzle = tk.StringVar(value="0.1")
        self.var_aperture = tk.StringVar(value="0.5")
        self.var_temp = tk.StringVar(value="230")
        self.var_arflow = tk.StringVar(value="")
        self.var_fluxpos = tk.StringVar(value="")
        _add_entry(geo_box, 0, "Nozzle->probe [in]", self.var_zprobe, tip="2.46 near / 12.3 far")
        _add_entry(geo_box, 1, "Nozzle->zone 2 [in]", self.var_zprobe2)
        _add_entry(geo_box, 2, "Nozzle->substrate [in]", self.var_zsub)
        _add_entry(geo_box, 3, "Nozzle diameter [in]", self.var_nozzle)
        _add_entry(geo_box, 4, "Aperture radius [cm]", self.var_aperture)
        _add_entry(geo_box, 5, "Oven temp [C]", self.var_temp)
        _add_entry(geo_box, 6, "Ar flow [SCCM]", self.var_arflow,
                   tip="blank = skip dopant density")
        _add_entry(geo_box, 7, "Flux ref. position", self.var_fluxpos,
                   tip="blank = parse from flux filename")

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

        # ── buttons + log ─────────────────────────────────────────────────────
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
        self.log = tk.Text(log_frame, height=14, state="disabled",
                           font=("Consolas", 9), wrap="none")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

    # ── config <-> widgets ────────────────────────────────────────────────────
    def _collect_config(self) -> AnalysisConfig:
        cfg = AnalysisConfig()
        deg = self.var_degree.get()
        cfg.ramp_degree = deg if deg == "auto" else int(deg)
        cfg.fit_peak_offset = self.var_offset.get()
        cfg.auto_trim = self.var_autotrim.get()
        cfg.background_method = self.var_bgmethod.get()
        cfg.absorption_mode = self.var_absmode.get()
        cfg.dark_offset_V = float(self.var_dark.get() or 0.0)
        cfg.error_mode = self.var_error.get()
        cfg.segmentation = self.var_seg.get()
        cfg.snr_min = float(self.var_snr.get())
        cfg.delta_aic_min = float(self.var_daic.get())
        cfg.min_good_sides = int(self.var_minsides.get())
        cfg.rejected_mode = self.var_rejmode.get()
        cfg.z_probe_in = float(self.var_zprobe.get())
        cfg.z_probe2_in = float(self.var_zprobe2.get())
        cfg.z_substrate_in = float(self.var_zsub.get())
        cfg.nozzle_diameter_in = float(self.var_nozzle.get())
        cfg.aperture_radius_cm = float(self.var_aperture.get())
        cfg.oven_temp_C = float(self.var_temp.get())
        cfg.ar_flow_sccm = float(self.var_arflow.get()) if self.var_arflow.get().strip() else None
        cfg.flux_reference_position = (float(self.var_fluxpos.get())
                                       if self.var_fluxpos.get().strip() else None)
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
        self.var_degree.set(str(cfg.ramp_degree)
                            if str(cfg.ramp_degree) in ("auto", "1", "2") else "auto")
        self.var_offset.set(cfg.fit_peak_offset)
        self.var_autotrim.set(cfg.auto_trim)
        self.var_bgmethod.set(cfg.background_method
                              if cfg.background_method in ("corners", "wings", "masked")
                              else "corners")
        self.var_absmode.set(cfg.absorption_mode)
        self.var_dark.set(str(cfg.dark_offset_V))
        self.var_error.set(cfg.error_mode)
        self.var_seg.set(cfg.segmentation if cfg.segmentation in ("sync", "auto") else "sync")
        self.var_snr.set(str(cfg.snr_min))
        self.var_daic.set(str(cfg.delta_aic_min))
        self.var_minsides.set(str(cfg.min_good_sides))
        self.var_rejmode.set(cfg.rejected_mode)
        self.var_zprobe.set(str(cfg.z_probe_in))
        self.var_zprobe2.set(str(cfg.z_probe2_in))
        self.var_zsub.set(str(cfg.z_substrate_in))
        self.var_nozzle.set(str(cfg.nozzle_diameter_in))
        self.var_aperture.set(str(cfg.aperture_radius_cm))
        self.var_temp.set(str(cfg.oven_temp_C))
        self.var_arflow.set("" if cfg.ar_flow_sccm is None else str(cfg.ar_flow_sccm))
        self.var_fluxpos.set("" if cfg.flux_reference_position is None
                             else str(cfg.flux_reference_position))
        self.var_gauss.set("gaussian" in cfg.profile_models)
        self.var_lor.set("lorentzian" in cfg.profile_models)
        self.var_pv.set("pseudo_voigt" in cfg.profile_models)
        self.var_poffset.set(cfg.profile_offset)

    def _save_config(self):
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         filetypes=[("JSON", "*.json")])
        if p:
            self._collect_config().save(p)
            self._log_line(f"Settings saved to {p}")

    def _load_config(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if p:
            self._apply_config(AnalysisConfig.load(p))
            self._log_line(f"Settings loaded from {p}")

    # ── running ───────────────────────────────────────────────────────────────
    def _on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        data, out = self.var_data.get().strip(), self.var_out.get().strip()
        if not data or not out:
            messagebox.showerror("Missing input",
                                 "Please select both a data folder and an output folder.")
            return
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Bad setting", f"Could not parse a setting: {exc}")
            return

        flux = self.var_flux.get().strip() or None
        zone2 = self.var_zone2.get().strip() or None
        label = self.var_label.get().strip() or Path(data).name

        self.btn_run.configure(state="disabled")
        self.progress.start(12)
        self._log_line("=" * 70)

        def work():
            from ..pipeline import run_full_pipeline
            try:
                run_full_pipeline(data, out, cfg, flux_source=flux,
                                  zone2_folder=zone2,
                                  progress=self.log_queue.put, label=label)
                self.log_queue.put("__DONE__")
            except Exception as exc:  # noqa: BLE001 — surface any failure in the log
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
                        self.notebook.select(self.tab_results)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 2 — Image Browser
    # ──────────────────────────────────────────────────────────────────────────
    def _build_browser_tab(self):
        f = self.tab_browse
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Stage:").pack(side="left")
        self.var_stage = tk.StringVar()
        self.combo_stage = ttk.Combobox(top, textvariable=self.var_stage, width=40,
                                        state="readonly")
        self.combo_stage.pack(side="left", padx=6)
        self.combo_stage.bind("<<ComboboxSelected>>", lambda e: self._load_stage_images())
        ttk.Button(top, text="Refresh", command=self._refresh_browser).pack(side="left", padx=4)
        ttk.Button(top, text="◀ Prev", command=lambda: self._step_image(-1)).pack(
            side="left", padx=(30, 2))
        ttk.Button(top, text="Next ▶", command=lambda: self._step_image(1)).pack(
            side="left", padx=2)

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        self.listbox = tk.Listbox(body, width=38, font=("Consolas", 9))
        self.listbox.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._show_selected_image())
        self.img_label = ttk.Label(body, anchor="center",
                                   text="Run the pipeline (or Refresh) to browse images.")
        self.img_label.pack(side="left", fill="both", expand=True)
        self.img_label.bind("<Configure>", lambda e: self._show_selected_image(redraw_only=True))
        self._current_photo = None
        self._stage_files: list[Path] = []

    def _refresh_browser(self):
        out = self.var_out.get().strip()
        stages = []
        if out and Path(out).is_dir():
            root = Path(out)
            # stage subfolders (including zone2/...) that contain PNGs
            for sub in sorted(root.rglob("*")):
                if sub.is_dir() and any(sub.glob("*.png")):
                    stages.append(str(sub.relative_to(root)))
            if any(root.glob("*.png")):
                stages.insert(0, "(final summaries)")
        self.combo_stage["values"] = stages
        if stages:
            self.var_stage.set(stages[0])
            self._load_stage_images()

    def _load_stage_images(self):
        out = Path(self.var_out.get().strip())
        stage = self.var_stage.get()
        folder = out if stage == "(final summaries)" else out / stage
        self._stage_files = sorted(folder.glob("*.png"))
        self.listbox.delete(0, "end")
        for p in self._stage_files:
            self.listbox.insert("end", p.name)
        if self._stage_files:
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(0)
            self._show_selected_image()

    def _step_image(self, delta: int):
        if not self._stage_files:
            return
        sel = self.listbox.curselection()
        i = (sel[0] if sel else 0) + delta
        i = max(0, min(len(self._stage_files) - 1, i))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(i)
        self.listbox.see(i)
        self._show_selected_image()

    def _show_selected_image(self, redraw_only: bool = False):
        if Image is None or not self._stage_files:
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        path = self._stage_files[sel[0]]
        try:
            img = Image.open(path)
        except OSError:
            return
        # Fit to the label while preserving aspect ratio.
        w = max(self.img_label.winfo_width(), 50)
        h = max(self.img_label.winfo_height(), 50)
        scale = min(w / img.width, h / img.height, 1.0)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        self._current_photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self._current_photo, text="")

    # ──────────────────────────────────────────────────────────────────────────
    # Tab 3 — Results
    # ──────────────────────────────────────────────────────────────────────────
    def _build_results_tab(self):
        f = self.tab_results
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Refresh", command=self._refresh_results).pack(side="left")
        ttk.Button(top, text="Open Output Folder",
                   command=self._open_output).pack(side="left", padx=6)
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
            for zone_key, zone_name in (("zone1", "ZONE 1"), ("zone2", "ZONE 2")):
                if r.get(zone_key):
                    lines.append(f"--- {zone_name} (Method A) " + "-" * 40)
                    for model, d in r[zone_key].items():
                        lines.append(f"  {model}:")
                        lines.append(f"    FWHM          = {d['fwhm_mm']:.2f} +- {d['fwhm_mm_err']:.2f} mm")
                        lines.append(f"    theta_1/2     = {d['theta_half_deg']:.3f} +- {d['theta_half_deg_err']:.3f} deg")
                        if "capture_eff" in d:
                            lines.append(f"    capture eff.  = {d['capture_eff']*100:.1f} +- {d['capture_eff_err']*100:.1f} %")
                        lines.append(f"    reduced chi2  = {d['chi2_reduced']:.2f}")
                        if d.get("eta") is not None:
                            lines.append(f"    eta (0=G,1=L) = {d['eta']:.2f}")
                    lines.append("")
            if r.get("method_b"):
                lines.append("--- METHOD B (two-zone differential) " + "-" * 26)
                for model, d in r["method_b"].items():
                    lines.append(f"  {model}: theta_1/2 = {d['theta_half_deg']:.3f} "
                                 f"+- {d['theta_half_deg_err']:.3f} deg")
                lines.append("")
            if r.get("flux"):
                lines.append("--- FLUX " + "-" * 55)
                for model, d in r["flux"].items():
                    lines.append(f"  {model}:")
                    lines.append(f"    absorption    = {d['absorption_pct']:.3f} % at position {d['reference_position']}")
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
    app = BeamProfilerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
