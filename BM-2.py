import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series

import pims
import trackpy as tp

video = pims.open ("/home/susanta/Downloads/PICT0097.AVI")

print("Number of frames:", len(video))
print ("Frames shape:", video.frame_shape)
fps = getattr(video, 'frame_rate', 15.0)
print(f"Frame rate    : {fps} fps")

plt.imshow(video[0])
plt.axis('off')
plt.show()

@pims.pipeline
def gray(image):
    if image.ndim == 3:
        return image[:, :, 1]
    return image

frames = gray(video)

plt.imshow(frames[0], cmap='gray')
plt.axis('off')
plt.show()

features = tp.locate(frames[0], diameter=11, minmass=1500, invert=True)
features.head()

tp.annotate(features, frames[0])
plt.show()

fig, ax = plt.subplots()
ax.hist(features['mass'], bins=20)
ax.set(xlabel='mass', ylabel='count')
plt.show()

features = tp.batch(
    frames,
    diameter=11,
    minmass=1500,
    invert=True,
    processes=1
)

features.head()

t = tp.link(features, search_range=5, memory=2)
t.head()

t1 = tp.filter_stubs(t, 25)

# Compare the number of particles before and after filtering
print(f"Number of particles before filtering: {t['particle'].nunique()}")
print(f"Number of particles after filtering: {t1['particle'].nunique()}")

plt.figure()
tp.mass_size(t1.groupby('particle').mean())
plt.show()

# Filter clean spherical colloidal particles
# t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) 
#          & (t1['ecc'] < 0.3))]
t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) 
         & (t1['ecc'] < 0.3))]
print(f"Clean, validated trajectories (t2): {t2['particle'].nunique()}")

# Annotate Frame 0 with clean filtered particles
plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0])
plt.show()

plt.figure()
tp.plot_traj(t2)
plt.show()

d = tp.compute_drift(t2)
d.plot()
plt.show()

tm = tp.subtract_drift(t2.copy(), d)

ax = tp.plot_traj(tm)
ax.set_title('Drift-subtracted trajectories')
plt.show()

# im = tp.imsd(tm, 100/285., 24)

# Spatial and temporal calibration
fps = 15.0  # Video frame acquisition rate (frames per second)
mpp = 0.1 # Microns per pixel calibration (set to 1.0 for pixel units, or specify calibration e.g., 0.15 um/pixel)
unit_label = r'\mu\mathrm{m}' if mpp != 1.0 else r'\mathrm{pixels}'

# Compute Individual Particle MSDs (IMSD)
im = tp.imsd(tm, mpp=mpp, fps=fps, max_lagtime=100)

fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set_xscale('log')
ax.set_yscale('log')

plt.show()

# em = tp.emsd(tm, 100/285., 24)

# Compute Ensemble Mean Squared Displacement (EMSD)
em = tp.emsd(tm, mpp=mpp, fps=fps, max_lagtime=100)

fig, ax = plt.subplots()
ax.plot(em.index, em, 'o')
ax.set_xscale('log')
ax.set_yscale('log')

ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set(ylim=(1e-2, 100))

plt.figure()
plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
plt.xlabel('lag time $t$')
tp.utils.fit_powerlaw(em)



# Calculate D, k_B and Avogadro's number


# Physical parameters
T = 298.15                  # Temperature in K (25 °C)
eta = 0.89e-3               # Viscosity of water at 25 °C (Pa.s)
r_um = 2.0                  # Particle radius in micrometers
R = 8.314462618             # Gas constant (J mol^-1 K^-1)


# Select the diffusive (linear) region of the EMSD


t_min = 0.2                 # lower fitting limit (seconds)
t_max = 2.0                 # upper fitting limit (seconds)

time = em.index.values
msd = em.values * mpp**2    # Convert pixel^2 to micrometer^2


mask = (time >= t_min) & (time <= t_max)

t_fit = time[mask]
msd_fit = msd[mask]


# Linear fit: MSD = 4Dt + intercept


slope, intercept = np.polyfit(t_fit, msd_fit, 1)

# For 2D Brownian motion:
D_um2_s = slope / 4.0

# Convert D from micrometer^2/s to m^2/s
D = D_um2_s * 1e-12
print(D)


# Stokes-Einstein relation
#
# D = k_B T / (6*pi*eta*r)
#
# Therefore:
# k_B = 6*pi*eta*r*D/T


r = r_um * 1e-6              # radius in meters

k_B = (6 * np.pi * eta * r * D) / T


# Avogadro's number
#
# R = N_A * k_B


N_A = R / k_B


# Print results


print(f"Temperature, T       = {T:.2f} K")
print(f"Water viscosity, eta = {eta:.3e} Pa.s")
print(f"Particle radius, r   = {r_um:.2f} µm")
print(f"mpp                  = {mpp:.4f} µm/pixel")

print("----------------------------------------")
print(f"MSD slope            = {slope:.6f} µm²/s")
print(f"Diffusion coefficient D = {D_um2_s:.6f} µm²/s")
print(f"Diffusion coefficient D = {D:.6e} m²/s")

print("----------------------------------------")
print(f"Boltzmann constant k_B = {k_B:.6e} J/K")
print(f"Avogadro number N_A    = {N_A:.6e} mol⁻¹")
print("========================================")