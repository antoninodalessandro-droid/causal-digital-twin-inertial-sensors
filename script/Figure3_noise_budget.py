import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Frequency axis
# -----------------------------
f = np.logspace(-4, 2, 2000)

# -----------------------------
# Noise components
# -----------------------------
brownian = 2.0e-11 * np.ones_like(f)
readout = 5.0e-13 * np.sqrt(1 + (f / 0.1)**4)
actuator = 1.0e-9 * np.ones_like(f)
quant_clock = 3.0e-11 * np.ones_like(f)

total_noise = np.sqrt(
    brownian**2 +
    readout**2 +
    actuator**2 +
    quant_clock**2
)

thermal_limit = 2.0e-11 * np.ones_like(f)

# -----------------------------
# Figure
# -----------------------------
fig, axs = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw=dict(hspace=0.12)
)

# =============================
# (a) Noise budget
# =============================
ax = axs[0]

ax.loglog(f, brownian, label='Brownian (thermal)', lw=2)
ax.loglog(f, readout, label='Readout', lw=2)
ax.loglog(f, actuator, label='Actuator / feedback', lw=2)
ax.loglog(f, quant_clock, label='Quantization + clock', lw=2)
ax.loglog(f, total_noise, label='Total self-noise', lw=2.5)

ax.set_ylabel(r'Acceleration noise  [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]')
ax.set_xlim(1e-4, 1e2)
ax.set_ylim(5e-13, 5e-7)

ax.grid(True, which='both', ls=':', lw=0.8)
ax.legend(loc='upper left', frameon=False)

# =============================
# (b) Self-noise vs thermal limit
# =============================
ax = axs[1]

ax.loglog(f, total_noise, lw=2.5, label='Total self-noise')
ax.loglog(f, thermal_limit, '--', lw=2, label='Thermal limit')

ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'Acceleration noise  [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]')

ax.set_xlim(1e-4, 1e2)
ax.set_ylim(1e-11, 5e-7)

ax.grid(True, which='both', ls=':', lw=0.8)
ax.legend(loc='upper left', frameon=False)

# -----------------------------
# Panel labels OUTSIDE axes
# -----------------------------
fig.text(0.065, 0.94, '(a)', fontsize=13)
fig.text(0.065, 0.47, '(b)', fontsize=13)

# -----------------------------
# Save figure
# -----------------------------
plt.tight_layout(rect=[0.06, 0.04, 0.99, 0.96])
plt.savefig('Figure3_noise_budget.png', dpi=300, bbox_inches='tight')
plt.show()
