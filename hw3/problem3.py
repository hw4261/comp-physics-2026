import numpy as np
import matplotlib.pyplot as plt

blurred = np.loadtxt("blur.txt")
M, N = blurred.shape
sigma = 25.0

print(f"image shape: {blurred.shape}")


def gaussian_psf(M, N, sigma):
    """Point-spread function f(x,y) = exp[-(x^2+y^2)/(2 sigma^2)]"""
    i = np.arange(M)
    j = np.arange(N)
    x = np.where(i < M / 2, i, i - M)
    y = np.where(j < N / 2, j, j - N)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


psf = gaussian_psf(M, N, sigma)

F = np.fft.rfft2(blurred)
P = np.fft.rfft2(psf)

eps = 1e-3
P_safe = np.where(np.abs(P) < eps, 1.0, P)
G = np.where(np.abs(P) < eps, F, F / P_safe)

unblurred = np.fft.irfft2(G, s=(M, N))

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.labelsize": 18, "axes.titlesize": 17, "axes.linewidth": 1.3,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

im0 = axes[0].imshow(blurred, cmap="gray")
axes[0].set_title("(a) blurred", loc="left")

im1 = axes[1].imshow(np.fft.fftshift(psf), cmap="gray")
axes[1].set_title(rf"(b) point spread function, $\sigma={sigma:g}$", loc="left")

im2 = axes[2].imshow(unblurred, cmap="gray", vmin=np.percentile(unblurred, 1),
                      vmax=np.percentile(unblurred, 99))
axes[2].set_title("(c) deconvolved", loc="left")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.savefig("problem3.png", dpi=300)
plt.show()