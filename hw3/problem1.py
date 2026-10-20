import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("sunspots.txt")
months, spots = data[:, 0], data[:, 1]
N = len(spots)

# DFT of the signal
c = np.fft.rfft(spots)
power = np.abs(c) ** 2
k = np.arange(len(c))

# The peak of the power spectrum gives the dominant frequency
k_peak = 1 + np.argmax(power[1:])
period_months = N / k_peak
print(f"N = {N} samples")
print(f"peak at k = {k_peak}, power = {power[k_peak]:.4e}")
print(f"period = N/k = {period_months:.2f} months = {period_months/12:.2f} years")

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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(months, spots, "-", color="#0072B2", lw=0.9)
axes[0].set_xlabel("month")
axes[0].set_ylabel("sunspot number")
axes[0].set_xlim(months.min(), months.max())
axes[0].set_title("(a) time series", loc="left")

scale = 1e10
axes[1].plot(k, power / scale, "-", color="#D55E00", lw=1.2)
axes[1].plot(k_peak, power[k_peak] / scale, "o", color="#0072B2", ms=7, zorder=3)
axes[1].text(k_peak + 3, power[k_peak] / scale, rf"$k={k_peak}$", fontsize=14, va="center")
axes[1].set_xlabel(r"$k$")
axes[1].set_ylabel(r"$|c_k|^2\ (10^{10})$")
axes[1].set_xlim(0, 50)
axes[1].set_title("(b) power spectrum", loc="left")

fig.tight_layout()
plt.savefig("problem1.png", dpi=300)
plt.show()