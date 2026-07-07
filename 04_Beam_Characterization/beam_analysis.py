import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import norm

# General Functions
def plot_beam_summary(div_results, flux_results, capture_results, dopant_density_cm3=None,
                      fit_type="lorentzian", title=None,
                      save_dir=None, dataset_label=None):
    """
    Combined summary plot: beam profile (Lorentzian or Gaussian fit) on the left,
    results table on the right.

    Parameters
    ----------
    div_results      : dict — from divergence_fitting_lor or divergence_fitting_gaus
    flux_results     : dict — from flux_from_divergence_lor or flux_from_divergence_gaus
    capture_results  : dict — from capture_efficiency_lor or capture_efficiency_gauss
    dopant_density_cm3 : float or None — from dopant_density(); omit row if None
    fit_type         : "lorentzian" or "gaussian"
    title            : str or None — suptitle override
    """
    unit_conv = 39.3701  # thou/mm

    popt   = div_results['popt']
    x_full = div_results['x_full']
    y_full = div_results['y_full']
    y_err  = div_results['y_err']
    x0     = popt[1]
    rescale = flux_results['rescale_factor']

    # ── build fit curve ───────────────────────────────────────────────────────
    #x_fit_thou = np.linspace(x_full.min(), x_full.max(), 2000)
    x_fit_thou = np.linspace(-50,1050,2000)

    if fit_type == "lorentzian":
        def profile(x, A, x0, gamma):
            return A / (1 + ((x - x0) / gamma)**2)
    else:
        def profile(x, A, x0, sigma):
            return A * np.exp(-(x - x0)**2 / (2.0 * sigma**2))

    y_fit_scaled = rescale * profile(x_fit_thou, *popt[:3])
    x_plot       = (x_fit_thou - x0) / unit_conv        # mm, centred
    x_data       = (x_full    - x0) / unit_conv

    # ── build table rows ──────────────────────────────────────────────────────
    table_rows = [
        [r"Fit type",          fit_type.capitalize()],
        [r"Reduced $\chi^2$",  f"{div_results['chi2_reduced']:.3f}"],
        [r"$R^2$",             f"{div_results['r_squared']:.4f}"],
        [r"FWHM [mm]",         f"{div_results['FWHM']:.2f} $\pm$ {div_results['FWHM_err']:.2f}"],
        [r"$\theta_{1/2}$ (FWHM) [deg]",
                               f"{div_results['divhalfangle_FWHM']:.3f} $\pm$ {div_results['divhalfangle_FWHM_err']:.3f}"],
        [r"Rb flux [atoms/s]", f"{flux_results['Rb_flux_atoms_s']:.2e}"],
        [r"Capture eff. [%]",  f"{capture_results['P']*100:.1f} $\pm$ {capture_results['P_err']*100:.1f}"],
        [r"Dist Nozzle [cm]",   f"{div_results['distance_nozzle']*2.54:.3f}"],
    ]
    if dopant_density_cm3 is not None:
        table_rows.append(
            [r"$\rho_\mathrm{Rb}$ [cm$^{-3}$]", f"{dopant_density_cm3:.2e}"]
        )

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(10, 5),
                                   gridspec_kw={'width_ratios': [2, 1]})
    if title:
        fig.suptitle(title, fontsize=9, color='gray')

    # profile panel
    ax.errorbar(x_data, rescale * y_full, yerr=rescale * y_err,
                fmt='x', markersize=4, capsize=2, linestyle='None',
                color='#2c7fb8', label='Data')
    ax.plot(x_plot, y_fit_scaled,
            color='#2c7fb8', linewidth=1.5, alpha=0.85,
            label=f'{fit_type.capitalize()} fit')

    ax.set_xlabel(r'$\Delta y$ [mm]', fontsize=14)
    ax.set_ylabel(r'Rb Column Density [cm$^{-2}$]', fontsize=14)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True, labelsize=11)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)

    ax.legend(fontsize=11)

    # fixed x ticks at -10, -5, 0, 5, 10 mm
    ax.set_xticks([-10, -5, 0, 5, 10])

    # y axis in units of 1e6, fixed range 0 → 1.5e6
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val/1e6:.1f}"))
    ax.set_ylabel(r'Rb Column Density [$\times10^6$ cm$^{-2}$]', fontsize=14)
    ax.set_ylim(0, 1.5e6)


    # table panel
    ax_t.axis('off')
    tbl = ax_t.table(
        cellText=table_rows,
        colLabels=["Parameter", "Value"],
        loc='center', cellLoc='left'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.1)

    # style header row
    for j in range(2):
        tbl[0, j].set_facecolor('#41b6c4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
  

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # build a descriptive filename from available info
        parts = []
        if dataset_label:
            parts.append(dataset_label)
        parts.append(fit_type)
        parts.append(f"chi2_{div_results['chi2_reduced']:.2f}".replace(".", "p"))
        parts.append(f"theta_{div_results['divhalfangle_FWHM']:.3f}deg".replace(".", "p"))
        fname = "beam_summary_" + "_".join(parts)

        fig.savefig(os.path.join(save_dir, f"{fname}.png"),
                    dpi=150, bbox_inches='tight')
        print(f"Saved → {os.path.join(save_dir, fname)}.png")

    plt.show()



def dopant_density(Ar_flow_sccm, Rb_flux, Ar_density_crystal = 1768):

    # ── Constants ─────────────────────────────────────────────────────────────
    Na    = 6.02214076e23          # mol⁻¹ Avogadro Constant
    R     = 8.3145                 # J/(mol·K)
    Z     = 0.99906                # compressibility factor for Ar at STP
    P     = 101325.0               # Pa (1 atm)
    M_Ar  = 39.948e-3              # kg/mol
    T_k   = 273.15                 # K (standard conditions)

    Ar_flow_scms = Ar_flow_sccm * 1e-6/60 # cm^3/min -> m^3/s

    # mass flow rate [kg/s] = volumetric * rho = volumetric * P*M / (Z*R*T)
    mass_flow = Ar_flow_scms * (P * M_Ar) / (Z * R * T_k)
    mass_flow_g_hr = mass_flow * 1000 * 3600  # kg/s → g/hr

    # atom flux [atoms/s]
    Ar_flux = mass_flow / M_Ar * Na 

    # ── Dopant Density ──────────────────────────────────────────────────────────
    Rb_per_Ar = Rb_flux / Ar_flux
    Ar_molar_density_crystal = Ar_density_crystal / M_Ar # mol/m^3

    # get num Ar atoms in cubic cm
    Ar_number_per_cm = Ar_molar_density_crystal * Na * 1e-6 # atoms/cm^3
    # get num Rb atoms per cubic cm Ar crystal
    Rb_density = Rb_per_Ar * Ar_number_per_cm # atoms/cm^3

    return Ar_flux, Rb_density, mass_flow_g_hr


def integrate_beam_profile(div_results, fit_type="gaussian", x_range=(-np.inf, np.inf)):
    """
    Integrate the fitted beam profile over x_range.
    
    Parameters
    ----------
    div_results : dict — output from divergence_fitting_gaus or divergence_fitting_lor
    fit_type    : str  — "gaussian" or "lorentzian"
    x_range     : tuple — (x_min, x_max) in thou, default infinite
    """
    popt = div_results['popt']

    if fit_type == "gaussian":
        def fit_func(x):
            A, x0, sigma = popt[0], popt[1], popt[2]
            return A * np.exp(-(x - x0)**2 / (2.0 * sigma**2))
    elif fit_type == "lorentzian":
        def fit_func(x):
            A, x0, gamma = popt[0], popt[1], popt[2]
            return A / (1 + ((x - x0) / gamma)**2)

    area, area_err = quad(fit_func, x_range[0], x_range[1])

    return area, area_err

# ──────────────────────────────────────────────────────────────────────────────# ──────────────────────────────────────────────────────────────────────────────
# Gaussian Versions
def divergence_fitting_gaus(file_folder, file_name, offset = False, error="STD", plot = True, distancenozzle = 2.46):
    # ── INPUTS ────────────────────────────────────────────────────────────────────
    filepath3      = os.path.join(file_folder, file_name)
    nozzleheight   = 0.150             # nozzle height in, 
    nozzleheight   = 0.1               # nozzle diameter in, this seems to be correct....
    # distancenozzle in in
    # near = 2.46
    # far = 12.3
    unit_conv      = 39.3701              # unit conversion factor applied to x-axis
    # ──────────────────────────────────────────────────────────────────────────────

    # load results CSV
    df = pd.read_csv(filepath3)

    # compute mean and std error across sides for plotting
    x_full = df['height']
    scales = df[['Scale1', 'Scale2', 'Scale3', 'Scale4']].to_numpy()
    y_full = np.mean(scales, axis=1)
    if error == "STD":
        y_err  = np.std(scales, axis=1, ddof=1)
    elif error == "SEM":
        y_err  = np.std(scales, axis=1, ddof=1) / np.sqrt(scales.shape[1])

    # fit gaussian
    if offset:
        p0_gaussian    = [0.05, 400, 100, 0.0]  # initial guess: [amplitude, center, sigma, offset]
        def gaussian(x, A, x0, sigma, offset):
            return A * np.exp(- (x - x0)**2 / (2.0 * sigma**2)) + offset
    else:          
        p0_gaussian    = [0.05, 400, 100]  # initial guess: [amplitude, center, sigma, offset]
        def gaussian(x, A, x0, sigma):
            return A * np.exp(- (x - x0)**2 / (2.0 * sigma**2))

    y_err_fit = np.where(y_err > 0, y_err, np.finfo(float).eps) #prevent divide be zero error, sometimes happens

    popt, pcov = curve_fit(
        gaussian, x_full, y_full,
        p0=p0_gaussian,
        sigma=y_err_fit,
        absolute_sigma=True
    )

    # ── Extract fit parameters and widths ─────────────────────────────────────────
    sigma     = popt[2]
    sigma_err = np.sqrt(pcov[2, 2])
    dx        = popt[1]

    FWHM      = 2 * np.sqrt(2 * np.log(2)) * sigma
    width_1e  = 2 * np.sqrt(2) * sigma

    FWHM_err     = 2 * np.sqrt(2 * np.log(2)) * sigma_err
    width_1e_err = 2 * np.sqrt(2) * sigma_err

    # ── Reduced chi-squared & R² ──────────────────────────────────────────────────
    fit = gaussian(x_full, *popt)
    residuals = (y_full - fit)
    dof = len(y_full) - len(popt)
    chi2_reduced = np.sum((residuals/y_err)**2)/dof

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_full - np.mean(y_full)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # ── Divergence half-angle — 1/e width ─────────────────────────────────────────
    beamgrowth = width_1e / 1000 - nozzleheight # in

    du_dw      = 1.0 / (2 * 1000 * distancenozzle)
    u          = (beamgrowth / 2) / distancenozzle
    dtheta_dw  = (1.0 / (1.0 + u**2)) * du_dw

    divhalfangle_1e     = math.atan((beamgrowth / 2) / distancenozzle)
    divhalfangle_1e_err = abs(dtheta_dw) * width_1e_err

    # ── Divergence half-angle — FWHM ──────────────────────────────────────────────
    beamgrowth = FWHM / 1000 - nozzleheight

    du_dw      = 1.0 / (2000 * distancenozzle)
    u          = (beamgrowth / 2) / distancenozzle
    dtheta_dw  = (1.0 / (1.0 + u**2)) * du_dw

    divhalfangle_FWHM     = math.atan((beamgrowth / 2) / distancenozzle)
    divhalfangle_FWHM_err = abs(dtheta_dw) * FWHM_err

    # convert thou in to mm
    FWHM = FWHM/unit_conv
    FWHM_err = FWHM_err/unit_conv
    width_1e = width_1e/unit_conv
    width_1e_err = width_1e_err/unit_conv

    if plot:
        # ── Plot + results table ───────────────────────────────────────────────────────
        x_fit = np.linspace(0, 1000, 2000)
        y_fit = gaussian(x_fit, *popt)

        fig, (ax_plot, ax_table) = plt.subplots(1, 2, figsize=(10, 5),
                                                gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(file_name, fontsize=9, color='gray')

        ax_plot.errorbar((x_full - popt[1]) / unit_conv, y_full, yerr=y_err,
                        fmt='x', markersize=3, linestyle='None', capsize=2, c="blue",label='Data')
        ax_plot.plot((x_fit - popt[1]) / unit_conv, y_fit, c="blue", alpha=0.8, linewidth=1.5,label="Gaussian fit")
        ax_plot.set_xlabel(r'$\Delta y$ [mm]',fontsize=16)
        ax_plot.set_ylabel("Relative Absorption [Arb. Units]",fontsize=16)
        ax_plot.tick_params(axis='both', labelsize=12)
        ax_plot.legend()

        # table of key results
        table_data = [
            ["Reduced χ²",        f"{chi2_reduced:.4f}"],
            ["R²",                f"{r_squared:.4f}"],
            ["FWHM (mm)",              f"{FWHM:.2f} ± {FWHM_err:.2f}"],
            ["θ½ (FWHM) [deg]",   f"{divhalfangle_FWHM*180/math.pi:.3f} ± {divhalfangle_FWHM_err*180/math.pi:.3f}"],
            ["1/e width (mm)",         f"{width_1e:.2f} ± {width_1e_err:.2f}"],
            ["θ½ (1/e) [deg]",    f"{divhalfangle_1e*180/math.pi:.3f} ± {divhalfangle_1e_err*180/math.pi:.3f}"],
        ]

        settings_data = [
            ["Offset", str(offset)],
            ["Error",  error],
        ]

        ax_table.axis('off')
        tbl = ax_table.table(cellText=table_data, colLabels=["Parameter", "Value"],
                            loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2)

        tbl2 = ax_table.table(cellText=settings_data, colLabels=["Setting", "Value"],
                            loc='lower center', cellLoc='left')
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        tbl2.scale(1, 1.5)

        plt.tight_layout()
        plt.show()
    
    # put results in dict
    results = {
        "FWHM":               FWHM,
        "FWHM_err":           FWHM_err,
        "width_1e":           width_1e,
        "width_1e_err":       width_1e_err,
        "divhalfangle_FWHM":  divhalfangle_FWHM * 180 / math.pi,
        "divhalfangle_FWHM_err": divhalfangle_FWHM_err * 180 / math.pi,
        "divhalfangle_1e":    divhalfangle_1e * 180 / math.pi,
        "divhalfangle_1e_err": divhalfangle_1e_err * 180 / math.pi,
        "chi2_reduced":       chi2_reduced,
        "r_squared":          r_squared,
        "x_full":             x_full.to_numpy(),
        "y_full":             y_full,
        "y_err":              y_err,
        "popt":               popt,
        "pcov":               pcov,
        "distance_nozzle":    distancenozzle,
    }
    
    return results





# ──────────────────────────────────────────────────────────────────────────────# ──────────────────────────────────────────────────────────────────────────────
def flux_from_divergence_gaus(flux_file, div_results, center_x_thou, T_oven_C=230, plot=True):
    """
    Compute total Rb beam flux....

    """
    # ── Constants ─────────────────────────────────────────────────────────────
    sigma_eff = 1.246e-9                         # cm² (isotropic pol, 85Rb D2 F3→F'4)
    fF3       = 7 / 12                           # thermal population fraction in F=3
    PRb85     = 0.7217                           # 85Rb isotopic abundance
    kB        = 1.380649e-23                     # J/K
    m_rb      = 85.4678 * 1.66053906660e-27      # kg
    unit_conv = 39.3701                          # thou-in per mm
    
    T_K = 273.15 + T_oven_C

    # ── 1. Column density from absolute absorption ────────────────────────────
    df_flux     = pd.read_csv(flux_file)
    absorptions = np.concatenate([df_flux[f'Scale{i}'].to_numpy() for i in range(1, 5)])
    abs_mean    = absorptions.mean()

    T0    = 1 - abs_mean / 100 
    OD0   = -math.log(T0)
    nRb85 = OD0 / (sigma_eff * fF3)       # cm⁻²
    nRb   = nRb85 / PRb85                 # cm⁻²

    # ── 2. Rescale divergence Gaussian to column density ──────────────────────
    y_full = div_results['y_full']
    y_err  = div_results['y_err']
    popt   = div_results['popt']
    x0     = popt[1]

    x_full  = div_results['x_full']
    closest_idx = np.argmin(np.abs(x_full - center_x_thou))
    rescale = nRb / y_full[closest_idx]

    def gaussian(x, A, x0, sigma):
        return A * np.exp(-(x - x0)**2 / (2.0 * sigma**2))

    x_fit_thou = np.linspace(0, 1000, 2000)
    y_fit      = rescale * gaussian(x_fit_thou, *popt[:3])

    # ── 3. Integrate column density across beam ───────────────────────────────
    x_fit_cm        = np.linspace(0, 2.54, 2000)
    N_cross_section = np.trapezoid(y_fit, x_fit_cm)   # cm⁻¹

    # ── 4. Mean speed → flux ──────────────────────────────────────────────────
    v_mean = math.sqrt(8 * kB * T_K / (math.pi * m_rb)) * 100   # cm/s
    Rb_flux = N_cross_section * v_mean                            # atoms/s

    # ── 5. Volume density for reference ───────────────────────────────────────
    interaction_len_cm = div_results['width_1e'] / 10   # mm → cm
    number_density_cm  = nRb / interaction_len_cm        # cm⁻³

    # ── plot ──────────────────────────────────────────────────────────────────
    if plot:
        x_fit_thou = np.linspace(0, 1000, 2000)
        y_fit_scaled = rescale * gaussian(x_fit_thou, *popt[:3])

        fig, (ax_plot, ax_table) = plt.subplots(1, 2, figsize=(10, 5),
                                                 gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(flux_file, fontsize=9, color='gray')

        x_full = div_results['x_full']
        ax_plot.errorbar((x_full - x0) / unit_conv, rescale * y_full, yerr=rescale * y_err,
                         fmt='x', markersize=3, linestyle='None', capsize=2, c="blue", label='Data')
        ax_plot.plot((x_fit_thou - x0) / unit_conv, y_fit_scaled, c="blue", alpha=0.8,
                     linewidth=1.5, label='Gaussian fit')
        ax_plot.set_xlabel(r'$\Delta y$ [mm]', fontsize=16)
        ax_plot.set_ylabel(r'Rb Density [cm$^{-2}$]', fontsize=16)
        ax_plot.tick_params(axis='both', labelsize=12)
        ax_plot.legend()

        table_data = [
            ["Flux [atoms/s]",       f"{Rb_flux:.1e}"],
            ["n [cm⁻³]",             f"{number_density_cm:.1e}"],
            #["θ½ (FWHM) [deg]",      f"{div_results['divhalfangle_FWHM']:.2f} ± {div_results['divhalfangle_FWHM_err']:.2f}"],
            ["θ½ (1/e) [deg]",       f"{div_results['divhalfangle_1e']:.2f} ± {div_results['divhalfangle_1e_err']:.2f}"],
            #["FWHM [mm]",            f"{div_results['FWHM']:.2f} ± {div_results['FWHM_err']:.2f}"],
            #["1/e width [mm]",       f"{div_results['width_1e']:.2f} ± {div_results['width_1e_err']:.2f}"],
        ]

        settings_data = [
            ["T oven [°C]", str(T_oven_C)],
        ]

        ax_table.axis('off')
        tbl = ax_table.table(cellText=table_data, colLabels=["Parameter", "Value"],
                             loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2)

        tbl2 = ax_table.table(cellText=settings_data, colLabels=["Setting", "Value"],
                              loc='lower center', cellLoc='left')
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        tbl2.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

    results = {
        "abs_mean_pct":        abs_mean,
        "OD":                  OD0,
        "nRb_column_cm2":      nRb,
        "N_cross_section_cm":  N_cross_section,
        "v_mean_cm_s":         v_mean,
        "Rb_flux_atoms_s":     Rb_flux,
        "number_density_cm3":  number_density_cm,
        "number_density_m3":   number_density_cm * 1e6,
        "rescale_factor":      rescale,
    }

    return results



# ──────────────────────────────────────────────────────────────────────────────# ──────────────────────────────────────────────────────────────────────────────
def capture_efficiency_gauss(
    div_deg,
    div_deg_err,
    z_thou,
    z_thou_err,
    distance_nozzle_to_z1=2.46,
    distance_nozzle_to_gr=19.685,
    aperture_radius_cm=0.5,
    plot=True,
):
    """
    Compute beam capture efficiency at the growth substrate (Gaussian, FWHM-based).

    Parameters
    ----------
    div_deg              : float — FWHM half-angle divergence [degrees]
    div_deg_err          : float — uncertainty in div_deg [degrees]
    z_thou               : float — FWHM full beam width at zone 1 [thou]
    z_thou_err           : float — uncertainty in z_thou [thou]
    distance_nozzle_to_z1: float — nozzle→measurement point [in] (default 2.46)
    distance_nozzle_to_gr: float — nozzle→growth substrate [in] (default 19.685)
    aperture_radius_cm   : float — capture half-width [cm] (default 0.5)
    plot                 : bool  — show diagnostic plot
    """
    # ── Constants ─────────────────────────────────────────────────────────────
    thou_to_cm = 2.54 / 1000.0
    in_to_cm   = 2.54

    # ── Geometry ──────────────────────────────────────────────────────────────
    D_in  = distance_nozzle_to_gr - distance_nozzle_to_z1
    D_cm  = D_in * in_to_cm

    theta = math.radians(div_deg)

    # ── Beam FWHM at growth substrate ─────────────────────────────────────────
    z_cm   = z_thou * thou_to_cm
    g_FWHM = z_cm + 2 * D_cm * math.tan(theta)               # full FWHM at substrate [cm]
    sigma  = g_FWHM / (2 * math.sqrt(2 * math.log(2)))        # Gaussian sigma [cm]

    # ── Propagate uncertainty ─────────────────────────────────────────────────
    dg_dz       = thou_to_cm
    dg_ddiv_deg = 2 * D_cm / math.cos(theta)**2 * (math.pi / 180.0)

    dsigma_dz   = dg_dz       / (2 * math.sqrt(2 * math.log(2)))
    dsigma_ddiv = dg_ddiv_deg / (2 * math.sqrt(2 * math.log(2)))

    sigma_err  = math.sqrt((dsigma_dz * z_thou_err)**2 + (dsigma_ddiv * div_deg_err)**2)
    g_FWHM_err = sigma_err * (2 * math.sqrt(2 * math.log(2)))

    # ── Capture probability P(|X| ≤ aperture_radius_cm) ──────────────────────
    a = aperture_radius_cm
    P = norm.cdf(a, 0, sigma) - norm.cdf(-a, 0, sigma)

    # dP/dσ (analytic)
    t         = a / sigma
    phi       = math.exp(-0.5 * t**2) / math.sqrt(2 * math.pi)
    dP_dsigma = -2 * a * phi / sigma**2
    P_err     = abs(dP_dsigma) * sigma_err

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        x = np.linspace(-4 * sigma, 4 * sigma, 1000)
        f = norm.pdf(x, 0, sigma)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, f, color='blue', linewidth=1.5, label='Beam profile')
        ax.axvspan(-a, a, alpha=0.2, color='green', label=f'Aperture ±{a} cm  (P = {P*100:.1f} ± {P_err*100:.1f}%)')
        ax.set_xlabel('Position [cm]', fontsize=14)
        ax.set_ylabel('Probability Density', fontsize=14)
        ax.tick_params(axis='both', labelsize=11)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {
        "g_FWHM":     g_FWHM,
        "g_FWHM_err": g_FWHM_err,
        "sigma":      sigma,
        "sigma_err":  sigma_err,
        "P":          P,
        "P_err":      P_err,
    }



# ──────────────────────────────────────────────────────────────────────────────# ──────────────────────────────────────────────────────────────────────────────
# Lorentzian Versions
def divergence_fitting_lor(file_folder, file_name, offset = False, error="STD", plot = True, distancenozzle = 2.46):
    # ── INPUTS ────────────────────────────────────────────────────────────────────
    filepath3      = os.path.join(file_folder, file_name)
    nozzleheight   = 0.150             # nozzle height in, 
    nozzleheight   = 0.1               # nozzle diameter in, this seems to be correct....
    # distancenozzle in in
    # near = 2.46
    # far = 12.3
    unit_conv      = 39.3701              # unit conversion factor applied to x-axis
    # ──────────────────────────────────────────────────────────────────────────────

    # load results CSV
    df = pd.read_csv(filepath3)

    x_full = df['height']
    scales = df[['Scale1', 'Scale2', 'Scale3', 'Scale4']].to_numpy()
    y_full = np.mean(scales, axis=1)
    if error == "STD":
        y_err  = np.std(scales, axis=1, ddof=1)
    elif error == "SEM":
        y_err  = np.std(scales, axis=1, ddof=1) / np.sqrt(scales.shape[1])

    if offset:
        p0_lorentzian  = [0.05, 400, 100, 0.0]
        def lorentzian(x, A, x0, gamma, offset):
            return A / (1 + ((x - x0) / gamma)**2) + offset
    else:
        p0_lorentzian  = [0.05, 400, 100]
        def lorentzian(x, A, x0, gamma):
            return A / (1 + ((x - x0) / gamma)**2)

    y_err_fit = np.where(y_err > 0, y_err, np.finfo(float).eps)

    popt, pcov = curve_fit(
        lorentzian, x_full, y_full,
        p0=p0_lorentzian,
        sigma=y_err_fit,
        absolute_sigma=True
    )

    # ── Extract fit parameters and widths ─────────────────────────────────────────
    gamma     = popt[2]
    gamma_err = np.sqrt(pcov[2, 2])
    dx        = popt[1]

    FWHM      = 2 * gamma
    FWHM_err  = 2 * gamma_err

    # ── Reduced chi-squared & R² ──────────────────────────────────────────────────
    fit = lorentzian(x_full, *popt)
    residuals = (y_full - fit)
    dof = len(y_full) - len(popt)
    chi2_reduced = np.sum((residuals/y_err)**2)/dof

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_full - np.mean(y_full)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # ── Divergence half-angle — FWHM ──────────────────────────────────────────────
    beamgrowth = FWHM / 1000 - nozzleheight

    du_dw      = 1.0 / (2000 * distancenozzle)
    u          = (beamgrowth / 2) / distancenozzle
    dtheta_dw  = (1.0 / (1.0 + u**2)) * du_dw

    divhalfangle_FWHM     = math.atan((beamgrowth / 2) / distancenozzle)
    divhalfangle_FWHM_err = abs(dtheta_dw) * FWHM_err

    # convert thou to mm
    FWHM     = FWHM / unit_conv
    FWHM_err = FWHM_err / unit_conv

    if plot:
        x_fit = np.linspace(0, 1000, 2000)
        y_fit = lorentzian(x_fit, *popt)

        fig, (ax_plot, ax_table) = plt.subplots(1, 2, figsize=(10, 5),
                                                gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(file_name, fontsize=9, color='gray')

        ax_plot.errorbar((x_full - popt[1]) / unit_conv, y_full, yerr=y_err,
                        fmt='x', markersize=3, linestyle='None', capsize=2, c="blue", label='Data')
        ax_plot.plot((x_fit - popt[1]) / unit_conv, y_fit, c="blue", alpha=0.8, linewidth=1.5, label="Lorentzian fit")
        ax_plot.set_xlabel(r'$\Delta y$ [mm]', fontsize=16)
        ax_plot.set_ylabel("Relative Absorption [Arb. Units]", fontsize=16)
        ax_plot.tick_params(axis='both', labelsize=12)
        ax_plot.legend()

        table_data = [
            ["Reduced χ²",       f"{chi2_reduced:.4f}"],
            ["R²",               f"{r_squared:.4f}"],
            ["FWHM (mm)",        f"{FWHM:.2f} ± {FWHM_err:.2f}"],
            ["θ½ (FWHM) [deg]",  f"{divhalfangle_FWHM*180/math.pi:.3f} ± {divhalfangle_FWHM_err*180/math.pi:.3f}"],
        ]

        settings_data = [
            ["Offset", str(offset)],
            ["Error",  error],
        ]

        ax_table.axis('off')
        tbl = ax_table.table(cellText=table_data, colLabels=["Parameter", "Value"],
                            loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2)

        tbl2 = ax_table.table(cellText=settings_data, colLabels=["Setting", "Value"],
                            loc='lower center', cellLoc='left')
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        tbl2.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

    results = {
        "FWHM":                  FWHM,
        "FWHM_err":              FWHM_err,
        "divhalfangle_FWHM":     divhalfangle_FWHM * 180 / math.pi,
        "divhalfangle_FWHM_err": divhalfangle_FWHM_err * 180 / math.pi,
        "chi2_reduced":          chi2_reduced,
        "r_squared":             r_squared,
        "x_full":                x_full.to_numpy(),
        "y_full":                y_full,
        "y_err":                 y_err,
        "popt":                  popt,
        "pcov":                  pcov,
        "distance_nozzle":       distancenozzle,
    }

    return results


def flux_from_divergence_lor(flux_file, div_results, center_x_thou, T_oven_C=230, plot=True):
    """
    Compute total Rb beam flux (Lorentzian beam profile).
    """
    # ── Constants ─────────────────────────────────────────────────────────────
    sigma_eff = 1.246e-9                         # cm² (isotropic pol, 85Rb D2 F3→F'4)
    fF3       = 7 / 12                           # thermal population fraction in F=3
    PRb85     = 0.7217                           # 85Rb isotopic abundance
    kB        = 1.380649e-23                     # J/K
    m_rb      = 85.4678 * 1.66053906660e-27      # kg
    unit_conv = 39.3701                          # thou-in per mm
    
    T_K = 273.15 + T_oven_C

    # ── 1. Column density from absolute absorption ────────────────────────────
    df_flux     = pd.read_csv(flux_file)
    absorptions = np.concatenate([df_flux[f'Scale{i}'].to_numpy() for i in range(1, 5)])
    abs_mean    = absorptions.mean()

    T0    = 1 - abs_mean / 100
    OD0   = -math.log(T0)
    nRb85 = OD0 / (sigma_eff * fF3)       # cm⁻²
    nRb   = nRb85 / PRb85                 # cm⁻²

    # ── 2. Rescale divergence Lorentzian to column density ────────────────────
    y_full = div_results['y_full']
    y_err  = div_results['y_err']
    popt   = div_results['popt']
    x0     = popt[1]

    x_full      = div_results['x_full']
    closest_idx = np.argmin(np.abs(x_full - center_x_thou))
    rescale     = nRb / y_full[closest_idx]

    def lorentzian(x, A, x0, gamma):
        return A / (1 + ((x - x0) / gamma)**2)

    x_fit_thou = np.linspace(0, 1000, 2000)
    y_fit      = rescale * lorentzian(x_fit_thou, *popt[:3])

    # ── 3. Integrate column density across beam ───────────────────────────────
    x_fit_cm        = np.linspace(0, 2.54, 2000)
    N_cross_section = np.trapezoid(y_fit, x_fit_cm)   # cm⁻¹

    # ── 4. Mean speed → flux ──────────────────────────────────────────────────
    v_mean  = math.sqrt(8 * kB * T_K / (math.pi * m_rb)) * 100   # cm/s
    Rb_flux = N_cross_section * v_mean                             # atoms/s

    # ── 5. Volume density for reference ───────────────────────────────────────
    interaction_len_cm = div_results['FWHM'] / 10   # mm → cm
    number_density_cm  = nRb / interaction_len_cm    # cm⁻³

    # ── plot ──────────────────────────────────────────────────────────────────
    if plot:
        x_fit_thou   = np.linspace(0, 1000, 2000)
        y_fit_scaled = rescale * lorentzian(x_fit_thou, *popt[:3])

        fig, (ax_plot, ax_table) = plt.subplots(1, 2, figsize=(10, 5),
                                                 gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(flux_file, fontsize=9, color='gray')

        ax_plot.errorbar((x_full - x0) / unit_conv, rescale * y_full, yerr=rescale * y_err,
                         fmt='x', markersize=3, linestyle='None', capsize=2, c="blue", label='Data')
        ax_plot.plot((x_fit_thou - x0) / unit_conv, y_fit_scaled, c="blue", alpha=0.8,
                     linewidth=1.5, label='Lorentzian fit')
        ax_plot.set_xlabel(r'$\Delta y$ [mm]', fontsize=16)
        ax_plot.set_ylabel(r'Rb Density [cm$^{-2}$]', fontsize=16)
        ax_plot.tick_params(axis='both', labelsize=12)
        ax_plot.legend()

        table_data = [
            ["Flux [atoms/s]",   f"{Rb_flux:.1e}"],
            ["n [cm⁻³]",         f"{number_density_cm:.1e}"],
            ["θ½ (FWHM) [deg]",  f"{div_results['divhalfangle_FWHM']:.2f} ± {div_results['divhalfangle_FWHM_err']:.2f}"],
            ["FWHM [mm]",        f"{div_results['FWHM']:.2f} ± {div_results['FWHM_err']:.2f}"],
        ]

        settings_data = [
            ["T oven [°C]", str(T_oven_C)],
        ]

        ax_table.axis('off')
        tbl = ax_table.table(cellText=table_data, colLabels=["Parameter", "Value"],
                             loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2)

        tbl2 = ax_table.table(cellText=settings_data, colLabels=["Setting", "Value"],
                              loc='lower center', cellLoc='left')
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        tbl2.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

    results = {
        "abs_mean_pct":        abs_mean,
        "OD":                  OD0,
        "nRb_column_cm2":      nRb,
        "N_cross_section_cm":  N_cross_section,
        "v_mean_cm_s":         v_mean,
        "Rb_flux_atoms_s":     Rb_flux,
        "number_density_cm3":  number_density_cm,
        "number_density_m3":   number_density_cm * 1e6,
        "rescale_factor":      rescale,
    }

    return results

def capture_efficiency_lor(
    div_deg,
    div_deg_err,
    z_thou,
    z_thou_err,
    distance_nozzle_to_z1=2.46,
    distance_nozzle_to_gr=19.685,
    aperture_radius_cm=0.5,
    plot=True,
):
    """
    Compute beam capture efficiency at the growth substrate (Lorentzian, FWHM-based).

    Parameters
    ----------
    div_deg              : float — FWHM half-angle divergence [degrees]
    div_deg_err          : float — uncertainty in div_deg [degrees]
    z_thou               : float — FWHM full beam width at zone 1 [thou]
    z_thou_err           : float — uncertainty in z_thou [thou]
    distance_nozzle_to_z1: float — nozzle→measurement point [in] (default 2.46)
    distance_nozzle_to_gr: float — nozzle→growth substrate [in] (default 19.685)
    aperture_radius_cm   : float — capture half-width [cm] (default 0.5)
    plot                 : bool  — show diagnostic plot
    """
    # ── Constants ─────────────────────────────────────────────────────────────
    thou_to_cm = 2.54 / 1000.0
    in_to_cm   = 2.54

    # ── Geometry ──────────────────────────────────────────────────────────────
    D_in  = distance_nozzle_to_gr - distance_nozzle_to_z1
    D_cm  = D_in * in_to_cm

    theta = math.radians(div_deg)

    # ── Beam FWHM at growth substrate ─────────────────────────────────────────
    z_cm   = z_thou * thou_to_cm
    g_FWHM = z_cm + 2 * D_cm * math.tan(theta)   # full FWHM at substrate [cm]
    gamma  = g_FWHM / 2.0                         # Lorentzian half-width γ [cm]

    # ── Propagate uncertainty ─────────────────────────────────────────────────
    dg_dz       = thou_to_cm
    dg_ddiv_deg = 2 * D_cm / math.cos(theta)**2 * (math.pi / 180.0)

    dgamma_dz   = dg_dz       / 2.0
    dgamma_ddiv = dg_ddiv_deg / 2.0

    gamma_err  = math.sqrt((dgamma_dz * z_thou_err)**2 + (dgamma_ddiv * div_deg_err)**2)
    g_FWHM_err = gamma_err * 2.0

    # ── Capture probability P(|X| ≤ aperture_radius_cm) ──────────────────────
    a = aperture_radius_cm

    P, _ = quad(lambda x: (1/math.pi) * (gamma / (x**2 + gamma**2)), -a, a)

    # dP/dγ (analytic)
    dP_dgamma = (2/math.pi) * (a / (a**2 + gamma**2))
    P_err     = abs(dP_dgamma) * gamma_err

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        x = np.linspace(-4 * gamma, 4 * gamma, 1000)
        f = (1/math.pi) * (gamma / (x**2 + gamma**2))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, f, color='blue', linewidth=1.5, label='Beam profile')
        ax.axvspan(-a, a, alpha=0.2, color='green', label=f'Aperture ±{a} cm  (P = {P*100:.1f} ± {P_err*100:.1f}%)')
        ax.set_xlabel('Position [cm]', fontsize=14)
        ax.set_ylabel('Probability Density', fontsize=14)
        ax.tick_params(axis='both', labelsize=11)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {
        "g_FWHM":     g_FWHM,
        "g_FWHM_err": g_FWHM_err,
        "gamma":      gamma,
        "gamma_err":  gamma_err,
        "P":          P,
        "P_err":      P_err,
    }





#________________---------------------------------from master document-----------------__________________________________

def get_div_results(absorption_data_folder, results_folder):

    # ── CONFIGURATION 
    csv_folder   = absorption_data_folder
    plots_folder = os.path.join(results_folder, "fit_plots")
    output_file  = os.path.join(results_folder, "divergence_results.csv")
    save_plots   = True
    sides        = ["1", "2", "3", "4"]
    degree = 2

    # fit equation
    def lorentz(x, G):
        return 1.0 / (1.0 + (x / (G / 2.0))**2)

    def model_function(x, G, scale, dx, offset):
        return scale * lorentz(x - dx, G) + offset

    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    heights   = [os.path.splitext(f)[0] for f in csv_files]

    if save_plots:
        os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    print(f"Found {len(heights)} CSV files: {heights}")

    results = []

    for height in heights:
        filepath = os.path.join(csv_folder, f"{height}.csv")
        row = {"height": height}
        print(height)
        for side in sides:
            try:
                data = pd.read_csv(filepath)

                t_min, t_max = -np.inf, np.inf
                if side == "n":
                    t_min, t_max = -np.inf, np.inf
                elif side == "1":
                    t_min, t_max = -0.003, -0.00105
                elif side == "2":
                    t_min, t_max = -0.0009, 0.001
                elif side == "3":
                    t_min, t_max = 0.0011, 0.003
                elif side == "4":
                    t_min, t_max = 0.0031, 0.005

                data_filtered = data[(data['Time(s)'] >= t_min) & (data['Time(s)'] <= t_max)]
                t   = data_filtered['Time(s)']
                CH2 = data_filtered["CH2(V)"]

                if side == "1":
                    t_remove_min, t_remove_max = -0.0023, -0.00145
                elif side =="2":
                    t_remove_min, t_remove_max = -0.0005, 0.00025
                elif side =="3":
                    t_remove_min, t_remove_max = 0.00175, 0.00275
                elif side == "4":
                    t_remove_min, t_remove_max = 0.0035, 0.0045

                data_subtrac_ramp = data_filtered[
                    (data_filtered['Time(s)'] < t_remove_min) |
                    (data_filtered['Time(s)'] > t_remove_max)
                ]
                t_ramp   = data_subtrac_ramp['Time(s)']
                CH2_ramp = data_subtrac_ramp["CH2(V)"]

                coeffs = np.polyfit(t_ramp, CH2_ramp, deg=degree)
                ramp   = np.polyval(coeffs, t)

                resid   = CH2_ramp - np.polyval(coeffs, t_ramp)
                dof_r   = len(t_ramp) - 3
                sigma_y = np.sqrt(np.sum(resid**2) / dof_r)

                Rb_spectrum = -(CH2 - ramp)

                if side == "1":
                    dx_guess = -0.0019
                elif side == "2":
                    dx_guess = -0.0001
                elif side == "3":
                    dx_guess = 0.0021
                elif side == "4":
                    dx_guess = 0.0041

                H_guess = np.nanmax(Rb_spectrum) - np.nanmedian(Rb_spectrum)
                p0 = [2.0e-4, H_guess, dx_guess, 0.0]
                lb = [1e-12, 0.0, -np.inf, -np.inf]
                ub = [9.5e-3,  np.inf,  np.inf,  np.inf]

                popt, pcov = curve_fit(
                    model_function, t, Rb_spectrum,
                    p0=p0, sigma=sigma_y, absolute_sigma=True,
                    bounds=(lb, ub), method="trf", maxfev=1000
                )

                G, scale, dx, offset = popt
                eG, escale, edx, eoffset = np.sqrt(np.diag(pcov))

                scale  = 0 if np.isnan(scale)  else scale
                escale = 0 if np.isnan(escale) else escale

                if save_plots:
                    y_fit = model_function(t, *popt)
                    t_ms = t * 1000

                    x_pad = 0.05 * (t_ms.max() - t_ms.min())
                    y_all = np.concatenate([np.asarray(Rb_spectrum), y_fit])
                    y_pad = 0.15 * (np.nanmax(y_all) - np.nanmin(y_all))

                    fig, ax = plt.subplots()
                    ax.plot(t_ms, Rb_spectrum, markersize=0.05, linestyle="None", marker="x", color="orange", label="data points")
                    ax.plot(t_ms, y_fit, label="fit", color="blue")
                    ax.set_xlim(t_ms.min() - x_pad, t_ms.max() + x_pad)
                    ax.set_ylim(np.nanmin(y_all) - y_pad, np.nanmax(y_all) + y_pad)
                    ax.set_xlabel("Laser Scan Time (ms)")
                    ax.set_ylabel("Relative Voltage Changes (A.U.)")
                    ax.set_title(filepath)
                    ax.legend(loc='best')
                    fig.savefig(os.path.join(plots_folder, f"{height}_side{side}_fit.png"), dpi=100, bbox_inches='tight')
                    plt.close(fig)

                row[f"Scale{side}"]  = 0 if np.isnan(scale) else abs(model_function(dx, *popt))
                row[f"eScale{side}"] = escale

            except Exception as e:
                print(f"FAILED: {filepath} side={side} → {e}")
                row[f"Scale{side}"]  = 0
                row[f"eScale{side}"] = 0

        results.append(row)

    df_out = pd.DataFrame(results, columns=[
        "height",
        "Scale1", "eScale1",
        "Scale2", "eScale2",
        "Scale3", "eScale3",
        "Scale4", "eScale4",
    ])
    df_out.to_csv(output_file, index=False)
    print(f"\nSaved → {output_file}")
    return df_out

def get_flux_results(absorption_data_folder, results_folder):

    # ── CONFIGURATION ─────────────────────────────────────────────────────────────
    csv_folder   = absorption_data_folder
    plots_folder = os.path.join(results_folder, "fit_plots")
    output_file  = os.path.join(results_folder, "flux_results.csv")
    save_plots   = True
    sides        = ["1", "2", "3", "4"]
    degree = 2

    # fit equation
    def lorentz(x, G):
        return 1.0 / (1.0 + (x / (G / 2.0))**2)

    def model_function(x, G, scale, dx, offset):
        return scale * lorentz(x - dx, G) + offset

    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])
    heights   = [os.path.splitext(f)[0] for f in csv_files]

    if save_plots:
        os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    print(f"Found {len(heights)} CSV files: {heights}")

    results = []

    for height in heights:
        filepath = os.path.join(csv_folder, f"{height}.csv")
        row = {"height": height}

        for side in sides:
            data = pd.read_csv(filepath)

            t_min, t_max = -np.inf, np.inf
            if side == "n":
                t_min, t_max = -np.inf, np.inf
            elif side == "1":
                t_min, t_max = -0.003, -0.00105
            elif side == "2":
                t_min, t_max = -0.0009, 0.001
            elif side == "3":
                t_min, t_max = 0.0011, 0.003
            elif side == "4":
                t_min, t_max = 0.0031, 0.005

            data_filtered = data[(data['Time(s)'] >= t_min) & (data['Time(s)'] <= t_max)]
            t   = data_filtered['Time(s)']
            CH2 = data_filtered["CH2(V)"]

            if side == "1":
                t_remove_min, t_remove_max = -0.0023, -0.00145
            elif side =="2":
                t_remove_min, t_remove_max = -0.0005, 0.00025
            elif side =="3":
                t_remove_min, t_remove_max = 0.00175, 0.00275
            elif side == "4":
                t_remove_min, t_remove_max = 0.0035, 0.0045

            data_subtrac_ramp = data_filtered[
                (data_filtered['Time(s)'] < t_remove_min) |
                (data_filtered['Time(s)'] > t_remove_max)
            ]
            t_ramp   = data_subtrac_ramp['Time(s)']
            CH2_ramp = data_subtrac_ramp["CH2(V)"]

            coeffs = np.polyfit(t_ramp, CH2_ramp, deg=degree)
            ramp   = np.polyval(coeffs, t)

            resid   = CH2_ramp - np.polyval(coeffs, t_ramp)
            dof_r   = len(t_ramp) - 3
            sigma_y = np.sqrt(np.sum(resid**2) / dof_r)

            Rb_spectrum = -(CH2 - ramp)

            V = np.asarray(CH2, dtype=float)
            V0 = np.asarray(ramp, dtype=float)

            valid = V0 > 0
            A_pct = np.full_like(V0, np.nan, dtype=float)
            A_pct[valid] = 100.0 * (1.0 - V[valid] / V0[valid])

            Rb_spectrum = A_pct

            if side == "1":
                dx_guess = -0.00185
            elif side == "2":
                dx_guess = -0.0002
            elif side == "3":
                dx_guess = 0.00225
            elif side == "4":
                dx_guess = 0.0039

            H_guess = np.nanmax(Rb_spectrum) - np.nanmedian(Rb_spectrum)
            p0 = [2.0e-4, H_guess, dx_guess, 0.0]
            lb = [1e-12, 0.0, -np.inf, -np.inf]
            ub = [9.5e-3,  np.inf,  np.inf,  np.inf]

            popt, pcov = curve_fit(
                model_function, t, Rb_spectrum,
                p0=p0, sigma=sigma_y, absolute_sigma=True,
                bounds=(lb, ub), method="trf", maxfev=1000
            )

            G, scale, dx, offset = popt
            eG, escale, edx, eoffset = np.sqrt(np.diag(pcov))

            if save_plots:
                y_fit = model_function(t, *popt)
                t_ms = t * 1000

                x_pad = 0.05 * (t_ms.max() - t_ms.min())
                y_all = np.concatenate([np.asarray(Rb_spectrum), y_fit])
                y_pad = 0.15 * (np.nanmax(y_all) - np.nanmin(y_all))

                fig, ax = plt.subplots()
                ax.plot(t_ms, Rb_spectrum, markersize=0.05, linestyle="None", marker="x", color="orange", label="data points")
                ax.plot(t_ms, y_fit, label="fit", color="blue")
                ax.set_xlim(t_ms.min() - x_pad, t_ms.max() + x_pad)
                ax.set_ylim(np.nanmin(y_all) - y_pad, np.nanmax(y_all) + y_pad)
                ax.set_xlabel("Laser Scan Time (ms)")
                ax.set_ylabel("Absorption (%)")
                ax.set_title(filepath)
                ax.legend(loc='best')
                fig.savefig(os.path.join(plots_folder, f"{height}_side{side}_fit.png"), dpi=100, bbox_inches='tight')
                plt.close(fig)

            row[f"Scale{side}"]  =  scale
            row[f"eScale{side}"] = escale

        results.append(row)

    df_out = pd.DataFrame(results, columns=[
        "height",
        "Scale1", "eScale1",
        "Scale2", "eScale2",
        "Scale3", "eScale3",
        "Scale4", "eScale4",
    ])
    df_out.to_csv(output_file, index=False)
    print(f"\nSaved → {output_file}")
    return df_out
