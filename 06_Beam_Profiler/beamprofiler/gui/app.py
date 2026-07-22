"""
Tkinter GUI for the Rb beam profiler (run-format data).

Three tabs:
  1. Run Analysis  — pick the data/output folders, adjust fit settings, run the
                     full pipeline with a live log.
  2. Image Browser — click through every stage image the pipeline saved.
  3. Results       — the final numbers from results.json, formatted.

Only the standard library + PIL (pillow) are needed on top of the analysis
dependencies; the pipeline runs on a worker thread so the window stays
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
        self.var_minsides = tk.StringVar(value="1")
        self.var_rejmode = tk.StringVar(value="zero")
        self.var_error = tk.StringVar(value="STD")
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
        _add_entry(fit_box, r, "Min good sides", self.var_minsides,
                   tip="fewer -> position flagged (red)"); r += 1
        _add_combo(fit_box, r, "Rejected sides", self.var_rejmode, ["zero", "exclude", "keep"],
                   "zero = count a failed side as 0 and average it in"); r += 1
        _add_combo(fit_box, r, "Point error", self.var_error, ["STD", "SEM", "upper_bound"],
                   "STD = side-to-side scatter (thesis choice)")

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

    # ── Tab 2: Image Browser ───────────────────────────────────────────────────
    def _build_browser_tab(self):
        f = self.tab_browse
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Stage:").pack(side="left")
        self.var_stage = tk.StringVar()
        self.combo_stage = ttk.Combobox(top, textvariable=self.var_stage, width=40, state="readonly")
        self.combo_stage.pack(side="left", padx=6)
        self.combo_stage.bind("<<ComboboxSelected>>", lambda e: self._load_stage_images())
        ttk.Button(top, text="Refresh", command=self._refresh_browser).pack(side="left", padx=4)
        ttk.Button(top, text="◀ Prev", command=lambda: self._step_image(-1)).pack(side="left", padx=(30, 2))
        ttk.Button(top, text="Next ▶", command=lambda: self._step_image(1)).pack(side="left", padx=2)

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
        i = max(0, min(len(self._stage_files) - 1, (sel[0] if sel else 0) + delta))
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
        try:
            img = Image.open(self._stage_files[sel[0]])
        except OSError:
            return
        w = max(self.img_label.winfo_width(), 50)
        h = max(self.img_label.winfo_height(), 50)
        scale = min(w / img.width, h / img.height, 1.0)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        self._current_photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self._current_photo, text="")

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
