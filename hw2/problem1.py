import numpy as np
import matplotlib.pyplot as plt

ACCURACY = 1e-6

# The equation to solve, x = f(x), and its derivative (needed for the error estimate)
def f(x, c):
    return 1 - np.exp(-c * x)

def fprime(x, c):
    return c * np.exp(-c * x)

# plain relaxation x' = f(x)
def relax(c, x0=0.5, accuracy=ACCURACY, max_iter=10000):
    x = x0
    for i in range(1, max_iter + 1):
        x_new = f(x, c)
        error = (x - x_new) / (1 - 1 / fprime(x, c))
        x = x_new
        if abs(error) < accuracy:
            return x, i
    raise RuntimeError("did not converge")

# overrelaxation x' = (1+omega) f(x) - omega x
def overrelax(c, omega, x0=0.5, accuracy=ACCURACY, max_iter=10000):
    x = x0
    for i in range(1, max_iter + 1):
        x_new = (1 + omega) * f(x, c) - omega * x
        error = (x - x_new) / (1 - 1 / ((1 + omega) * fprime(x, c) - omega))
        x = x_new
        if abs(error) < accuracy:
            return x, i
    raise RuntimeError("did not converge")

# plain relaxation for c = 2
x_plain, it_plain = relax(2)
print(f"Plain relaxation:  x = {x_plain:.8f},  iterations = {it_plain}")

# overrelaxation for a range of omega
print(f"\n{'omega':>8s}{'iterations':>12s}{'x':>14s}")
omegas = np.arange(-0.5, 0.95, 0.05)
iters = []
for omega in omegas:
    x, it = overrelax(2, omega)
    iters.append(it)
    print(f"{omega:8.2f}{it:12d}{x:14.8f}")
iters = np.array(iters)

i_best = np.argmin(iters)
print(f"\nFastest: omega = {omegas[i_best]:.2f}, {iters[i_best]} iterations "
      f"(plain relaxation took {it_plain})")

xstar = x_plain
omega_opt = fprime(xstar, 2) / (1 - fprime(xstar, 2))
print(f"Predicted optimal omega (zero amplification factor): {omega_opt:.3f}")

# Plot
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

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(omegas, iters, "o-", color="#0072B2", lw=1.6, ms=5)
ax.axhline(it_plain, color="0.55", ls="--", lw=1.5)
ax.text(omegas[0], it_plain + 0.4, "plain relaxation", fontsize=13, color="0.3")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("iterations to converge")
fig.tight_layout()
plt.savefig("problem1a.png", dpi=300)
plt.show()

def f_osc(x):
    return np.exp(1 - x**2)

def overrelax_osc(omega, x0=0.5, n=15):
    x = x0
    xs = [x]
    for _ in range(n):
        x = (1 + omega) * f_osc(x) - omega * x
        xs.append(x)
    return np.array(xs)

print("\nPart (d) demonstration: x = exp(1 - x^2), true solution x* = 1")
for omega in (0.0, -0.5):
    xs = overrelax_osc(omega)
    label = "plain relaxation (omega=0)" if omega == 0 else f"overrelaxation (omega={omega})"
    print(f"{label}: last 5 iterates = {np.round(xs[-5:], 6)}")

fig, ax = plt.subplots(figsize=(7, 5.5))
for omega, color, label in [(0.0, "#0072B2", r"$\omega=0$ (plain relaxation)"),
                              (-0.5, "#D55E00", r"$\omega=-0.5$")]:
    xs = overrelax_osc(omega)
    ax.plot(xs, "o-", color=color, lw=1.6, ms=5, label=label)
ax.axhline(1.0, color="0.55", ls="--", lw=1.5)
ax.set_xlabel("iteration")
ax.set_ylabel("$x$")
ax.legend(fontsize=13)
fig.tight_layout()
plt.savefig("problem1d.png", dpi=300)
plt.show()