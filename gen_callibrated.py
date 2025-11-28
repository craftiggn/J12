import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# -----------  USER-DEFINED CALIBRATION PARAMETERS -----------
# ============================================================
# Original calibration: channel = m * (energy_keV) + b
# Therefore: energy_keV = (channel - b) / m

m = 1.587522       # slope
b = 16.156407      # intercept
m_err = 0.000105
b_err = 0.109665

# ============================================================
# ------------------------ FUNCTIONS --------------------------
# ============================================================

def read_lst_data(filename):
    x_vals = []
    y_vals = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("$DATA"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError("No $DATA section found in file.")

    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        ch, counts = map(int, parts)
        x_vals.append(ch)
        y_vals.append(counts)

    return np.array(x_vals), np.array(y_vals)


# Gaussian model in CHANNEL space
def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C


def fit_gaussian(x, y, a_ch, b_ch):
    mask = (x >= a_ch) & (x <= b_ch)
    x_fit = x[mask]
    y_fit = y[mask]

    A_guess = np.max(y_fit) - np.min(y_fit)
    mu_guess = x_fit[np.argmax(y_fit)]
    sigma_guess = (b_ch - a_ch) / 6
    C_guess = np.min(y_fit)

    try:
        popt, pcov = curve_fit(
            gaussian, x_fit, y_fit,
            p0=[A_guess, mu_guess, sigma_guess, C_guess],
            bounds=([0, -np.inf, 0, 0],
                    [np.inf, np.inf, np.inf, np.inf]),
            maxfev=10000
        )

        uncertainties = np.sqrt(np.diag(pcov))
        return x_fit, y_fit, popt, uncertainties

    except RuntimeError:
        print("Gaussian fit failed.")
        return x_fit, y_fit, None, None


# ============================================================
# ------------------------- MAIN ------------------------------
# ============================================================

filename = "J12_Deuter.lst.lst"
Title = "Deuteru"
Title = "Test"

# ------------------------------------------------------------
# point, delta, a_keV, b_keV are in TRUE ENERGY (keV)
# ------------------------------------------------------------
point_keV = 400
delta_keV = 100

a_keV = point_keV - delta_keV
b_keV = point_keV + delta_keV

# Convert keV → channels for fitting window
a_ch = m * a_keV + b
b_ch = m * b_keV + b

do_fit = 1
save_fig = 1

x, y = read_lst_data(filename)

# ============================================================
# ------------------- FULL SPECTRUM PLOT ---------------------
# ============================================================

# Convert channel → keV correctly
# energy_keV = (ch - b) / m
x_cal_full = (x - b) / m

plt.figure(figsize=(10, 5))
plt.plot(x_cal_full, y, linewidth=1.2)
plt.xlabel("Energy (keV)")
plt.ylabel("Counts")
plt.title("Full Spectrum " + Title + " (calibrated)")
plt.tight_layout()
plt.savefig("output_full.png", dpi=300)
plt.close()

# ============================================================
# ------------------- RANGE PLOT IN keV -----------------------
# ============================================================

mask = (x >= a_ch) & (x <= b_ch)
x_plot = x[mask]
y_plot = y[mask]

# Convert channel → keV properly
x_plot_keV = (x_plot - b) / m

plt.figure(figsize=(10, 5))
plt.plot(x_plot_keV, y_plot, linewidth=2,
         alpha=0.85, label="Dane eksperymentalne")

# ---------------------- Gaussian fit ----------------------
if do_fit:
    x_fit, y_fit, params, errs = fit_gaussian(x, y, a_ch, b_ch)

    if params is not None:
        A, mu_ch, sigma_ch, C = params
        A_err, mu_err_ch, sigma_err_ch, C_err = errs

        # Convert center & width from channel → keV
        mu_keV = (mu_ch - b) / m
        sigma_keV = sigma_ch / m

        # Uncertainty of energy centroid
        mu_keV_err = np.sqrt(
            (mu_err_ch / m)**2 +
            ((mu_ch - b) * m_err / m**2)**2 +
            (b_err / m)**2
        )

        # FWHM in keV
        FWHM_keV = 2.354820045 * sigma_keV
        FWHM_keV_err = 2.354820045 * (sigma_err_ch / m)

        # Smooth Gaussian in channel space
        x_smooth_ch = np.linspace(a_ch, b_ch, 400)
        y_smooth = gaussian(x_smooth_ch, *params)

        # Convert smooth curve → keV
        x_smooth_keV = (x_smooth_ch - b) / m
        plt.plot(x_smooth_keV, y_smooth, 'k--', label="Dopasowany rozkład Gaussa")

        # ---------------- DISPLAY FIT RESULTS ----------------
        fit_text = (
            f"μ = {mu_keV:.2f} ± {mu_keV_err:.2f} keV\n"
            f"σ = {sigma_keV:.2f} ± {sigma_err_ch/m:.2f} keV\n"
            f"FWHM = {FWHM_keV:.2f} ± {FWHM_keV_err:.2f} keV"
        )

        plt.gca().text(
            0.02, 0.98, fit_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.25)
        )

plt.xlabel("Energia (keV)")
plt.ylabel("Zliczenia")
plt.title(f"Widmo {Title} od {a_keV} do {b_keV} keV")
plt.legend()
plt.tight_layout()

if save_fig:
    plt.savefig(f"{Title}_{a_keV}_{b_keV}.png", dpi=300)
    print("Figure saved.")

plt.show()
