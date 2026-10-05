import numpy as np
import matplotlib.pyplot as plt

def numerical_gradient(f, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def gradient_descent(f, x0, step0=1.0, tol=1e-10, max_iter=20000):
    x = np.asarray(x0, dtype=float)
    fx = f(x)
    history = [fx]
    for _ in range(max_iter):
        grad = numerical_gradient(f, x)
        grad_sq = np.dot(grad, grad)
        if grad_sq == 0:
            break
        step = step0
        while True:
            x_try = x - step * grad
            with np.errstate(all="ignore"):
                f_try = f(x_try)
            if np.isfinite(f_try) and f_try <= fx - 1e-4 * step * grad_sq:
                break
            step *= 0.5
            if step < 1e-16:
                break
        x_new = x - step * grad
        f_new = f(x_new)
        history.append(f_new)
        if abs(fx - f_new) < tol:
            x, fx = x_new, f_new
            break
        x, fx = x_new, f_new
    return x, np.array(history)

def test_func(p):
    x, y = p
    return (x - 2)**2 + (y - 2)**2

x0_test = np.array([-2.0, -1.0])
x_test, hist_test = gradient_descent(test_func, x0_test, step0=0.1)
print(f"Test function: starting at {x0_test}, found minimum at "
      f"({x_test[0]:.6f}, {x_test[1]:.6f}) in {len(hist_test)-1} iterations "
      f"(f = {hist_test[-1]:.2e})")

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

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

axes[0].semilogy(np.maximum(hist_test, 1e-18), "-", color="#0072B2", lw=1.8)
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("$f(x,y)$")

path = [x0_test.copy()]
x = x0_test.copy()
for _ in range(len(hist_test) - 1):
    grad = numerical_gradient(test_func, x)
    step = 0.5
    while True:
        x_try = x - step * grad
        if test_func(x_try) <= test_func(x) - 1e-4 * step * np.dot(grad, grad):
            break
        step *= 0.5
    x = x - step * grad
    path.append(x.copy())
path = np.array(path)

margin = 0.5
x_lo, x_hi = min(path[:, 0].min(), 2) - margin, max(path[:, 0].max(), 2) + margin
y_lo, y_hi = min(path[:, 1].min(), 2) - margin, max(path[:, 1].max(), 2) + margin
xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 200), np.linspace(y_lo, y_hi, 200))
zz = (xx - 2)**2 + (yy - 2)**2
r_max = np.hypot(max(x_hi - 2, 2 - x_lo), max(y_hi - 2, 2 - y_lo))
levels = np.linspace(0.3, r_max, 12)**2
axes[1].contour(xx, yy, zz, levels=levels, colors="0.7", linewidths=0.8)
axes[1].plot(path[:, 0], path[:, 1], "o-", color="#D55E00", ms=4, lw=1.2)
axes[1].plot(2, 2, "*", color="#0072B2", ms=16, zorder=3)
axes[1].set_xlim(x_lo, x_hi)
axes[1].set_ylim(y_lo, y_hi)
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$y$")
fig.tight_layout()
plt.savefig("problem3_test.png", dpi=300)
plt.show()

logM, n_data, n_err = np.loadtxt("smf_cosmos.dat", unpack=True)

def schechter(logM, p):
    """Schechter function, n(M) = ln(10) * phi* * (M/M*)^(alpha+1) * exp(-M/M*),
    written in terms of x = log10(M) and the fit parameters
    p = [log10(phi*), log10(M*), alpha]. We fit the logs of phi* and M* instead
    of phi* and M* themselves, because they are both strictly positive and span
    many orders of magnitude, which makes the chi^2 surface much better behaved
    for gradient descent than fitting phi* and M* directly."""
    log_phi, log_Mstar, alpha = p
    ratio = 10**(logM - log_Mstar)
    return np.log(10) * 10**log_phi * ratio**(alpha + 1) * np.exp(-ratio)

def chi2(p):
    model = schechter(logM, p)
    return np.sum(((n_data - model) / n_err)**2)

starting_points = {
    "start 1": [-3.5, 10.5, -0.5],
    "start 2": [-2.8, 11.1, -0.8],
    "start 3": [-3.0, 11.3, -1.3],
}

results = {}
for name, p0 in starting_points.items():
    with np.errstate(all="ignore"):
        p_fit, hist = gradient_descent(chi2, p0, step0=1.0, max_iter=20000)
    results[name] = (p_fit, hist)
    print(f"{name}: p0={p0} -> log_phi={p_fit[0]:.4f}, log_Mstar={p_fit[1]:.4f}, "
          f"alpha={p_fit[2]:.4f}, chi2={hist[-1]:.4f}, iterations={len(hist)-1}")

p_best = results["start 1"][0]
log_phi_best, log_Mstar_best, alpha_best = p_best
print(f"\nBest fit: phi* = {10**log_phi_best:.4e},  M* = {10**log_Mstar_best:.4e},  "
      f"alpha = {alpha_best:.4f}")

fig, ax = plt.subplots(figsize=(7, 5.5))
colors = {"start 1": "#0072B2", "start 2": "#D55E00", "start 3": "#009E73"}
for name, (p_fit, hist) in results.items():
    ax.semilogy(hist, "-", color=colors[name], lw=1.6, label=name)
ax.set_xlabel("step $i$")
ax.set_ylabel(r"$\chi^2$")
ax.legend(fontsize=14)
fig.tight_layout()
plt.savefig("problem3a.png", dpi=300)
plt.show()

M_gal = 10**logM
M_fine = np.logspace(logM.min(), logM.max(), 300)
n_model_fine = schechter(np.log10(M_fine), p_best)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.errorbar(M_gal, n_data, yerr=n_err, fmt="o", color="#0072B2", ms=6,
            capsize=3, label="COSMOS data")
ax.plot(M_fine, n_model_fine, "-", color="#D55E00", lw=1.8, label="best-fit Schechter")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_{\rm gal}$")
ax.set_ylabel(r"$n(M_{\rm gal})$  [dex$^{-1}$ (Mpc/$h)^{-3}$]")
ax.legend(fontsize=14)
fig.tight_layout()
plt.savefig("problem3b.png", dpi=300)
plt.show()