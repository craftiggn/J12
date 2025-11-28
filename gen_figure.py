import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = "/home/marcel/Desktop/J12/J12_137Cs.lst.lst"     # change to your file
Title = ""
Title = "137Cs"
if len(Title) == 0:
    Title = 'output'
do_fit = 1
save_fig = 1
# point = 2050
# delta = 50
# a = point - delta
# b = point + delta

a = 1000
b = 1100



def read_lst_data(filename):
    x_vals = []
    y_vals = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find where $DATA: section starts
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("$DATA"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError("No $DATA section found in file.")

    # parse data lines
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        x, y = map(int, parts)
        x_vals.append(x)
        y_vals.append(y)

    return np.array(x_vals), np.array(y_vals)


# Gaussian model
def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C


def fit_gaussian(x, y, a, b):
    mask = (x >= a) & (x <= b)
    x_fit = x[mask]
    y_fit = y[mask]

    # Initial guesses
    A_guess = np.max(y_fit) - np.min(y_fit)
    mu_guess = x_fit[np.argmax(y_fit)]
    sigma_guess = (b - a) / 6
    C_guess = np.min(y_fit)

    try:
        # Fit with bounds and more max function evaluations
        popt, pcov = curve_fit(
            gaussian, x_fit, y_fit,
            p0=[A_guess, mu_guess, sigma_guess, C_guess],
            bounds=(
                [0, -np.inf, 0, 0],      # lower bounds
                [np.inf, np.inf, np.inf, np.inf]  # upper bounds
            ),
            maxfev=10000  # allow more iterations
        )

        # Uncertainties from covariance matrix
        uncertainties = np.sqrt(np.diag(pcov))
        mu_err = uncertainties[1]
        sigma_err = uncertainties[2]

        return x_fit, y_fit, popt, mu_err, sigma_err

    except RuntimeError:
        # Fit failed: return data only, no fit
        print("Gaussian fit failed; displaying data only.")
        return x_fit, y_fit, None, None, None


# ---------------- MAIN SCRIPT ----------------

x, y = read_lst_data(filename)

# Always show full spectrum
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="data")
plt.xlabel("Channel (X)")
plt.ylabel("Counts (Y)")
plt.title("Full Spectrum " + Title)
plt.tight_layout()
plt.savefig("output_full.png", dpi=300)
plt.close()

# ---------------- RANGE PLOT ----------------
x_range = (x >= a) & (x <= b)
x_plot = x[x_range]
y_plot = y[x_range]

plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_plot, linewidth=2, alpha=0.8, label="Dane eksperymentalne")

# Only do Gaussian fit if requested
if do_fit:
    x_fit, y_fit, params, mu_err, sigma_err = fit_gaussian(x, y, a, b)
    if params is not None:
        A, mu, sigma, C = params
        FWHM = 2.354820045 * abs(sigma)
        FWHM_err = 2.354820045 * sigma_err

        print(f"Gaussian mean mu          : {mu:.2f} ± {mu_err:.3f}")
        print(f"Gaussian sigma            : {sigma:.2f} ± {sigma_err:.3f}")
        print(f"Gaussian FWHM             : {FWHM:.2f} ± {FWHM_err:.3f}")

        # Smooth Gaussian line
        x_smooth = np.linspace(a, b, 400)
        y_smooth = gaussian(x_smooth, *params)
        plt.plot(x_smooth, y_smooth, 'k:', label="Dopasowany rozkład Gaussa")

        # Text box
        fit_text = (
            f"μ = {mu:.2f} ± {mu_err:.3f}\n"
            f"σ = {sigma:.2f} ± {sigma_err:.3f}\n"
            f"FWHM = {FWHM:.2f} ± {FWHM_err:.3f}"
        )
        plt.gca().text(
            0.02, 0.98, fit_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2)
        )

plt.xlabel("Numer Kanału")
plt.ylabel("Liczba zliczeń")
plt.title("Widmo " + Title + f" w zakresie kanałów {a}–{b}")
plt.legend()
plt.tight_layout()

# Save if requested
if save_fig:
    plt.savefig(Title + f"{a}_{b}.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved as {Title}{a}_{b}.png")

plt.show()
