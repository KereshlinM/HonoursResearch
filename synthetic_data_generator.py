import numpy as np
import matplotlib.pyplot as plt


def folding_function(x, y, z, a0, bks, cks, dks, sigmaks, z_max):
    fold = a0 + (2.0 * z / z_max) * sum(
        bk * np.exp(-((x - ck) ** 2 + (y - dk) ** 2) / (2 * sigmak ** 2))
        for bk, ck, dk, sigmak in zip(bks, cks, dks, sigmaks)
    )
    return fold


grid_size = 128
x_range = y_range = z_range = np.linspace(0, 1000, grid_size)
X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
z_max = x_range[-1]

reflection_coefficients = np.random.uniform(-1, 1, (grid_size, grid_size, grid_size))
a0 = 0.5
N = 10
bks = np.random.uniform(0, 1, N)
cks = np.random.uniform(0, 1000, N)
dks = np.random.uniform(0, 1000, N)
sigmaks = np.random.uniform(50, 200, N)

seismic_volume = reflection_coefficients.copy()

for i in range(grid_size):
    seismic_volume[:, :, i] += folding_function(X[:, :, i], Y[:, :, i], Z[:, :, i], a0, bks, cks, dks, sigmaks, z_max)

num_faults = 10
fault_volume = np.zeros_like(seismic_volume)

for _ in range(num_faults):
    # Random fault parameters
    mu_f = np.random.uniform(-1, 1)  # fault dip vector
    nu_f = np.random.uniform(-1, 1)  # fault strike vector
    omega_f = np.random.uniform(-1, 1)  # fault normal vector
    R = np.array([mu_f, nu_f, omega_f])  # fault vector
    S = np.diag(R)
    x0, y0, z0 = np.random.randint(0, grid_size, 3)

    for x in range(max(0, x0 - 5), min(grid_size, x0 + 5)):
        for y in range(max(0, y0 - 5), min(grid_size, y0 + 5)):
            for z in range(max(0, z0 - 5), min(grid_size, z0 + 5)):
                distance_vector = np.array([x - x0, y - y0, z - z0])
                exponent = -0.5 * np.dot(distance_vector.T, S @ distance_vector)
                exponent = np.clip(exponent, -50, 50)
                fault_effect = np.exp(exponent)
                seismic_volume[x, y, z] += fault_effect
                if fault_effect > 15:
                    fault_volume[x, y, z] = 1


def ricker_wavelet(f, length=0.128, dt=0.001):
    t = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1 - 2 * (np.pi ** 2) * (f ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t ** 2))
    return y


freq = 25
ricker = ricker_wavelet(freq, length=0.128, dt=0.008)


for x in range(grid_size):
    for y in range(grid_size):
        seismic_volume[x, y, :] = np.convolve(seismic_volume[x, y, :], ricker, mode='same')

noise_level = 0.1
noise = noise_level * np.max(seismic_volume) * np.random.normal(size=seismic_volume.shape)
seismic_volume += noise
seismic_volume = np.nan_to_num(seismic_volume)

depth_index = 64
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(seismic_volume[:, :, depth_index], cmap="seismic", extent=[0, 1000, 0, 1000])
plt.colorbar(label="Amplitude")
plt.title("Seismic Volume with Faults and Noise")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.subplot(1, 2, 2)
plt.imshow(fault_volume[:, :, depth_index], cmap="gray", extent=[0, 1000, 0, 1000])
plt.colorbar(label="Fault Presence (1 for fault, 0 for no fault)")
plt.title("Fault Volume")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()
