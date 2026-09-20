import numpy as np
import matplotlib.pyplot as plt

# NumPy defaults to double precision (float64).
# To ensure that arithmetics are single precision, every input to the derivative
# calculation is cast with f32(...).

f32 = np.float32
EPS_M = 1e-7

# Define functions for the 3 differentiation method
def forward_diff(f, x, h):
    """f'(x) ~ [f(x+h) - f(x)] / h
    truncation error ~ h"""
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    """f'(x) ~ [f(x+h) - f(x-h)] / 2h
    truncation error ~ h^2"""
    return (f(x + h) - f(x - h)) / (f32(2) * h)

def extrap_diff(f, x, h):
    """f'(x) ~ [-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)] / 12h = (4*D(h) - D(2h)) / 3 with D = central difference
    truncation error ~ h^4"""
    two = f32(2)
    num = (-f(x + two * h) + f32(8) * f(x + h) - f32(8) * f(x - h) + f(x - two * h))
    return num / (f32(12) * h)

METHODS = {
    "forward":      (forward_diff, 1),
    "central":      (central_diff, 2),
    "extrapolated": (extrap_diff,  4),
}

# Define the functions that we what to differentiate
FUNCS = {
    "cos(x)": (np.cos, lambda x: -np.sin(x)),
    "exp(x)": (np.exp, np.exp),
}

# Take derivatives at these x's
XS = [0.1, 10.0]

# Step sizes, log-spaced
H = np.logspace(-6, 0, 100).astype(f32)

# Compute relative errors
results = {}
for fname, (f, fprime_exact) in FUNCS.items():
    for x0 in XS:
        x = f32(x0)
        truth = fprime_exact(float(x))
        for mname, (dfun, p) in METHODS.items():
            h = H
            h = h[h > 0]
            d = dfun(f, x, h)
            rel_err = np.abs(d.astype(np.float64) - truth) / abs(truth)
            results[(fname, x0, mname)] = (h.astype(np.float64), np.maximum(rel_err, 1e-12))


# Measured vs predicted
def smoothed_min(h, e, w=7):
    """Roundoff error is noisy (it randomly passes near zero), so a raw error is not reliable.
    Take a running median of log10(error) first then find its minimum."""
    le = np.log10(e)
    med = np.array([np.median(le[max(0, i - w // 2): i + w // 2 + 1]) for i in range(len(le))])
    i = np.argmin(med)
    return h[i], 10 ** med[i]

print(f"{'func':7s}{'x':>5s} {'method':13s}{'h_opt(meas)':>12s}{'h_opt(pred)':>12s}"
      f"{'eps_opt(meas)':>15s}{'eps_opt(pred)':>15s}{'sig. digits':>13s}")
for (fname, x0, mname), (h, e) in results.items():
    p = METHODS[mname][1]
    h_meas, e_meas = smoothed_min(h, e)
    h_pred, e_pred = EPS_M ** (1 / (p + 1)), EPS_M ** (p / (p + 1))
    print(f"{fname:7s}{x0:5.1f} {mname:13s}{h_meas:12.2e}{h_pred:12.2e}"
          f"{e_meas:15.2e}{e_pred:15.2e}{-np.log10(e_meas):13.1f}")

# Create log-log error plots
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

style = {"forward":      ("#0072B2", "o"),
         "central":      ("#D55E00", "s"),
         "extrapolated": ("#009E73", "^")}
hh = np.logspace(-6.5, 0.5, 200)
panel_letters = ["(a)", "(b)", "(c)", "(d)"]
titles = {"cos(x)": r"$\cos x$", "exp(x)": r"$e^{x}$"}

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
for r, fname in enumerate(FUNCS):
    for c, x0 in enumerate(XS):
        ax = axes[r, c]
        for p, h_lo in ((1, 1e-4), (2, 1e-3), (4, 1e-2)):
            g = np.logspace(np.log10(h_lo), 0.5, 50)
            ax.loglog(g, g ** p, "--", color="0.55", lw=1.5, zorder=1)
        for mname, (_, p) in METHODS.items():
            col, mk = style[mname]
            h, e = results[(fname, x0, mname)]
            ax.loglog(h, e, "-", color=col, lw=1.6, zorder=3,
                      label=mname if (r, c) == (0, 0) else None)
        ax.set_title(f"{panel_letters[2 * r + c]} {titles[fname]}, $x = {x0:g}$", loc="left")
        ax.set_xlim(5e-7, 2)
        ax.set_ylim(1e-9, 10)
        ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
        ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
        if r == 1: ax.set_xlabel("step size $h$")
        if c == 0: ax.set_ylabel(r"relative error $\epsilon$")

# Legend
axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.02, 0.19), fontsize=13,
                  handlelength=1.6, borderpad=0.5, labelspacing=0.3,
                  framealpha=0.9).set_zorder(6)
fig.tight_layout()
plt.show()