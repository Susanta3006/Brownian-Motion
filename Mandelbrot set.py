import matplotlib.pyplot as plt
import numpy as np


def mandelbrot(width=800, height=600, max_iter=100):
  # Define the complex plane bounds (Real: -2.0 to 1.0, Imaginary: -1.2 to 1.2)
  x = np.linspace(-2.0, 1.0, width)
  y = np.linspace(-1.2, 1.2, height)
  c = x[np.newaxis, :] + 1j * y[:, np.newaxis]

  z = np.zeros_like(c)
  escape_time = np.zeros(c.shape, dtype=int)

  for i in range(max_iter):
    # Track points that haven't escaped yet (|z| <= 2)
    mask = np.abs(z) <= 2
    z[mask] = z[mask] ** 2 + c[mask]
    escape_time[mask] = i

  return escape_time


# Generate and plot the fractal
fractal = mandelbrot(width=1000, height=800, max_iter=150)

plt.figure(figsize=(10, 8))
plt.imshow(fractal, extent=[-2.0, 1.0, -1.2, 1.2], cmap="hot", origin="lower")
plt.colorbar(label="Iterations before escape")
plt.title("Mandelbrot Set")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.show()