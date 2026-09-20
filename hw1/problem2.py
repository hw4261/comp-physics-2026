import numpy as np
import matplotlib.pyplot as plt

# NumPy defaults to double precision (float64).
# To ensure that arithmetics are single precision, every input to the derivative
# calculation is cast with f32(...).
f32 = np.float32

# Integration limits
a, b = f32(0.0), f32(1.0)

# Exact value of the integral, computed in double precision
I_EXACT = 1.0 - np.exp(-1.0)

# The integrand (float32 in, float32 out)
def f(t):
    return np.exp(-t)

def fsum(arr):
    """Add up the numbers one at a time, like a running total in a loop.
    np.sum adds in pairs (pairwise summation), which hides most of the roundoff error."""
    return np.cumsum(arr, dtype=f32)[-1]

# Define functions for the 3 integration methods
def midpoint(N):
    """I ~ h * sum f(x_i),  x_i = a + (i + 1/2) h,  i = 0..N-1
    truncation error ~ h^2"""
    h = (b - a) / f32(N)
    i = np.arange(N, dtype=f32)
    x = a + (i + f32(0.5)) * h
    return h * fsum(f(x))

def trapezoid(N):
    """I ~ h * [f(a)/2 + f(b)/2 + sum f(x_i)],  x_i = a + i h,  i = 1..N-1
    truncation error ~ h^2"""
    h = (b - a) / f32(N)
    i = np.arange(1, N, dtype=f32)
    x = a + i * h
    return h * (f32(0.5) * (f(a) + f(b)) + fsum(f(x)))

def simpson(N):
    """I ~ (h/3) * [f(a) + f(b) + 4 sum_(i odd) f(x_i) + 2 sum_(i even) f(x_i)]
    N must be even (Simpson's rule works on pairs of bins)
    truncation error ~ h^4"""
    h = (b - a) / f32(N)
    x_odd = a + np.arange(1, N, 2, dtype=f32) * h
    x_even = a + np.arange(2, N - 1, 2, dtype=f32) * h
    total = f(a) + f(b) + f32(4) * fsum(f(x_odd))
    if len(x_even) > 0:
        total = total + f32(2) * fsum(f(x_even))
    return (h / f32(3)) * total

# (function, truncation power p)
METHODS = {
    "midpoint":  (midpoint,  2),
    "trapezoid": (trapezoid, 2),
    "Simpson":   (simpson,   4),
}

# Number of bins log-spaced. They are even so that Simpson's rule can be used, and stop below
# 2^23 so that the bin index i is still exactly representable in float32.
Ns = np.unique(2 * np.round(np.logspace(0.3, 6.9, 100) / 2)).astype(int)

# Compute relative errors
errors = {}
for mname, (rule, p) in METHODS.items():
    I_num = np.array([rule(int(N)) for N in Ns], dtype=np.float64)
    errors[mname] = np.maximum(np.abs(I_num - I_EXACT) / I_EXACT, 1e-12)

# Print the errors at N ~ 10, 100, ...
print(f"{'N':>9s}" + "".join(f"{m:>14s}" for m in METHODS))
for target in [10, 100, 1e3, 1e4, 1e5, 1e6, 5e6]:
    j = np.argmin(np.abs(np.log10(Ns) - np.log10(target)))
    print(f"{Ns[j]:9d}" + "".join(f"{errors[m][j]:14.2e}" for m in METHODS))

# Create log-log error plot
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

style = {"midpoint":  "#0072B2",
         "trapezoid": "#D55E00",
         "Simpson":   "#009E73"}
NN = np.logspace(0.3, 6.9, 200)

fig, ax = plt.subplots(figsize=(8, 6))
for mname, (_, p) in METHODS.items():
    # guide lines: N^-2 and N^-4 (unit prefactor)
    for p in (2, 4):
        ax.loglog(NN, 1 / NN ** p, "--", color="0.55", lw=1.5, zorder=1)
    ax.loglog(Ns, errors[mname], "-", color=style[mname], lw=1.6, zorder=3, label=mname)
ax.set_xlim(1.5, 1.2e7)
ax.set_ylim(1e-9, 1)
ax.set_xticks([1e0, 1e2, 1e4, 1e6])
ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
ax.set_xlabel("number of bins $N$")
ax.set_ylabel(r"relative error $\epsilon$")

# Legend
ax.legend(loc="upper center", fontsize=13, handlelength=1.6, borderpad=0.5,
          labelspacing=0.3, framealpha=0.9)
fig.tight_layout()
plt.show()

