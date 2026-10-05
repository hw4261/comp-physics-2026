import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const

ACCURACY = 1e-6

def f(x):
    return 5 * np.exp(-x) + x - 5

def find_root(f, x1, x2, accuracy):
    """Find a root of f(x) = 0 in (x1, x2) by binary search. f(x1) and f(x2) must
    have opposite signs. Stop once the bracket is narrower than accuracy."""
    if f(x1) * f(x2) >= 0:
        raise ValueError("f(x1) and f(x2) must have opposite signs")
    iterations = 0
    while abs(x2 - x1) > accuracy:
        xm = 0.5 * (x1 + x2)
        if f(xm) == 0:
            return xm, iterations
        if f(x1) * f(xm) < 0:
            x2 = xm
        else:
            x1 = xm
        iterations += 1
    return 0.5 * (x1 + x2), iterations

x_root, iterations = find_root(f, 4.0, 6.0, ACCURACY)
print(f"x = {x_root:.6f}  ({iterations} iterations)")

h, c, kB = const.h, const.c, const.k
b = h * c / (kB * x_root)
print(f"b = {b:.7e} m K  ({b*1e3:.4f} mm K)")
print(f"accepted value: b = 2.8977719e-3 m K")

lam_sun = 502e-9
T_sun = b / lam_sun
print(f"\nT_sun = {T_sun:.1f} K  (accepted value: ~5772 K)")

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

xx = np.linspace(0.3, 8, 400)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(xx, f(xx), "-", color="#0072B2", lw=1.8)
ax.axhline(0, color="0.6", lw=1)
ax.plot(x_root, 0, "o", color="#D55E00", ms=8, zorder=3)
ax.text(x_root + 0.2, 1.5, rf"$x = {x_root:.3f}$", fontsize=16)
ax.set_xlabel("$x$")
ax.set_ylabel(r"$5e^{-x}+x-5$")
fig.tight_layout()
plt.savefig("problem2.png", dpi=300)
plt.show()