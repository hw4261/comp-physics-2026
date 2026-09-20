import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Correlation function xi(r) from the power spectrum P(k)
# xi(r) = 1/(2 pi^2) * integral dk k^2 P(k) sin(kr)/(kr)

# Read the tabulated power spectrum
# first column is k [h/Mpc], second column is P(k) [(Mpc/h)^3]
k_tab, P_tab = np.loadtxt("lcdm_z0.matter_pk", usecols=(0, 1), unpack=True)

# Cubic spline interpolation of P(k), the table is log-spaced and P(k) changes by many orders
# of magnitude, so we interpolate ln P as a function of ln k. "natural" means the second
# derivatives are zero at the ends of the table as mentioned in class.
spline = CubicSpline(np.log(k_tab), np.log(P_tab), bc_type="natural")

def P_of_k(k):
    return np.exp(spline(np.log(k)))

# Simpson's rule in ln k. Changing variables from k to ln k, dk k^2 P = d(ln k) k^3 P, so
# xi(r) = 1/(2 pi^2) * integral d(ln k) k^3 P(k) sin(kr)/(kr)
def xi_of_r(r, kmin, kmax, N):
    """xi(r) for an array of r, using Simpson's rule with N bins (N even) in ln k
    between kmin and kmax"""
    u = np.linspace(np.log(kmin), np.log(kmax), N + 1)
    h = u[1] - u[0]
    k = np.exp(u)
    # Simpson weights 1, 4, 2, 4, ..., 2, 4, 1
    w = np.ones(N + 1)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    base = w * k**3 * P_of_k(k)
    xi = np.array([h / 3 * np.sum(base * np.sin(k * ri) / (k * ri)) for ri in r])
    return xi / (2 * np.pi**2)

def find_peak(r, y, r_lo=90):
    """Position and height of the highest point of y(r) for r > r_lo (the BAO bump).
    A parabola through the maximum point and its two neighbors gives the position between grid points."""
    i = np.where(r > r_lo)[0][np.argmax(y[r > r_lo])]
    a, b, c = y[i - 1], y[i], y[i + 1]
    dr = r[1] - r[0]
    return r[i] + 0.5 * dr * (a - c) / (a - 2 * b + c), b

# Choices for the integral: lower limit, upper limit, and number of bins
KMIN, KMAX, N = 1e-4, 100.0, 2**18

# Compute xi(r) for r = 50 to 120 Mpc/h
r = np.arange(50, 120.001, 0.1)
xi = xi_of_r(r, KMIN, KMAX, N)
y = r**2 * xi

# Position of the BAO peak
r_peak, y_peak = find_peak(r, y)
print(f"BAO peak: r = {r_peak:.2f} Mpc/h,  r^2 xi = {y_peak:.2f}")

# Test if the result is robust by varing the upper limit and the number of bins (only near the peak, to save time)
r_fine = np.arange(90, 120.001, 0.1)
print("\nvary kmax (N = 2^18)")
for kmax in [3, 10, 30, 100]:
    yy = r_fine**2 * xi_of_r(r_fine, KMIN, kmax, 2**18)
    rp, yp = find_peak(r_fine, yy)
    print(f"  kmax = {kmax:5.0f}   peak r = {rp:.2f}   r^2 xi = {yp:.3f}")
print("vary N (kmax = 100)")
for n in [2**14, 2**16, 2**18, 2**19]:
    yy = r_fine**2 * xi_of_r(r_fine, KMIN, 100.0, n)
    rp, yp = find_peak(r_fine, yy)
    print(f"  N = 2^{int(np.log2(n)):2d}    peak r = {rp:.2f}   r^2 xi = {yp:.3f}")

# Create the plot
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.labelsize": 20, "axes.titlesize": 18, "axes.linewidth": 1.3,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 8, "ytick.major.size": 8,
    "xtick.minor.size": 4, "ytick.minor.size": 4,
    "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(r, y, "-", color="#0072B2", lw=1.8)
ax.axvline(r_peak, color="0.55", ls="--", lw=1.5, zorder=1)
ax.plot(r_peak, y_peak, "o", color="#D55E00", ms=8, zorder=3)
ax.text(r_peak - 2, y_peak + 1.5, rf"$r = {r_peak:.1f}$ Mpc/$h$", ha="right", fontsize=16)
ax.set_xlim(50, 120)
ax.set_xlabel(r"$r$ [Mpc/$h$]")
ax.set_ylabel(r"$r^2\xi(r)$ [(Mpc/$h)^2$]")
fig.tight_layout()
plt.show()