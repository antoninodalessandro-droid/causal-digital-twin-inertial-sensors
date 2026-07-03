import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------
# Parameters
# -----------------------------
ENOB = np.arange(12, 21)
f = np.logspace(-4, 2, 600)

# Reference noise levels (nominal, consistent with previous figures)
S_th = 2e-11                 # thermal (Brownian) ASD [m s^-2 / sqrt(Hz)]
S_act = 1e-9                 # actuator / feedback ASD floor

def S_adc(f, enob):
    # ADC + quantization noise, ENOB-dependent
    # NOTE: use float base to allow negative exponents safely
    return 3e-11 * np.power(2.0, 16 - enob) * (1.0 + (f / 5.0)**2)

def S_readout(f):
    # readout noise with low-frequency floor and high-frequency rise
    return 5e-13 * (1.0 + (f / 0.2)**2)

# -----------------------------
# (a) Regime map
# -----------------------------
regime = np.zeros((len(ENOB), len(f)))

for i, enob in enumerate(ENOB):
    Sadc = S_adc(f, enob)
    Sro = S_readout(f)
    Stack = np.vstack([
        Sro,                     # readout
        S_act * np.ones_like(f), # actuator
        Sadc                     # ADC
    ])
    regime[i, :] = np.argmax(Stack, axis=0)

cmap = ListedColormap([
    "#fdae61",  # readout-dominated
    "#abd9e9",  # actuator-dominated
    "#2c7bb6"   # ADC-dominated
])

# -----------------------------
# (b) Near-plateau bandwidth (MODIFIED PART)
# -----------------------------
# Practical metric: bandwidth over which the total self-noise stays within
# a factor beta of the minimum achievable plateau (not of the thermal limit).
beta = 1.3
bandwidth = []

for enob in ENOB:
    Stot = np.sqrt(
        S_th**2 +
        S_act**2 +
        S_readout(f)**2 +
        S_adc(f, enob)**2
    )
    S_plateau = np.min(Stot)
    idx = np.where(Stot <= beta * S_plateau)[0]
    bandwidth.append(f[idx[-1]] if len(idx) > 0 else np.nan)

bandwidth = np.array(bandwidth)

# -----------------------------
# Plot
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# (a) Regime map
im = axs[0].pcolormesh(
    f, ENOB, regime,
    shading="auto",
    cmap=cmap
)
axs[0].set_xscale("log")
axs[0].set_ylabel("ENOB [bits]")
axs[0].set_title("(a)", loc="left")
axs[0].set_yticks(ENOB)

cbar = fig.colorbar(im, ax=axs[0], pad=0.01)
cbar.set_ticks([0.33, 1.0, 1.66])
cbar.set_ticklabels([
    "Readout-dominated",
    "Actuator-dominated",
    "ADC-dominated"
])

axs[0].grid(True, which="both", ls=":", lw=0.8)

# (b) Near-plateau bandwidth
axs[1].plot(ENOB, bandwidth, marker="o")
axs[1].set_xlabel("ENOB [bits]")
axs[1].set_ylabel("Near-plateau bandwidth [Hz]")
axs[1].set_yscale("log")
axs[1].set_title("(b)", loc="left")
axs[1].grid(True, which="both", ls=":", lw=0.8)

plt.savefig("Figure7_design_regime_map.png", dpi=300)
plt.show()
