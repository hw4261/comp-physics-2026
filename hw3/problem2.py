import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt("dow.txt")
N = len(y)

# Real FFT
c = np.fft.rfft(y)
Nc = len(c)


def smooth(c, frac):
    """Keep only the first `frac` fraction of the (low-k) Fourier coefficients,
    zero the rest, and transform back."""
    c_filtered = c.copy()
    n_keep = int(frac * Nc)
    c_filtered[n_keep:] = 0.0
    return np.fft.irfft(c_filtered, n=N)


y_10pct = smooth(c, 0.10)
y_2pct = smooth(c, 0.02)

print(f"N = {N} samples, {Nc} non-redundant Fourier coefficients")
print(f"10% cutoff: keep first {int(0.10 * Nc)} coefficients")
print(f"2% cutoff:  keep first {int(0.02 * Nc)} coefficients")

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

days = np.arange(N)
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(days, y, "-", color="0.7", lw=1.0, label="original", zorder=1)
ax.plot(days, y_10pct, "-", color="#0072B2", lw=1.8, label="first 10% kept", zorder=3)
ax.plot(days, y_2pct, "-", color="#D55E00", lw=1.8, label="first 2% kept", zorder=2)
ax.set_xlabel("trading day")
ax.set_ylabel("Dow Jones Industrial Average")
ax.set_xlim(0, N)
ax.legend(fontsize=14, framealpha=0.9)
fig.tight_layout()
plt.savefig("problem2.png", dpi=300)
plt.show()