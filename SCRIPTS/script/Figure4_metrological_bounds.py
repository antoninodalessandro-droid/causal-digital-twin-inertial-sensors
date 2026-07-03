import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Physical constants & params
# -----------------------------
kB = 1.380649e-23
T = 300.0          # K
m = 1.0            # kg
zeta = 0.20
T0 = 120.0         # s
omega0 = 2*np.pi / T0
c = 2 * zeta * m * omega0

# -----------------------------
# Frequency axis
# -----------------------------
f = np.logspace(-4, 2, 2000)

# -----------------------------
# Fundamental thermal limit (ASD)
# Correct unit: [m s^-2 / sqrt(Hz)]
# S_a (PSD) = 4 kB T c / m^2  -> ASD = sqrt(S_a)
# -----------------------------
Sa_th_PSD = (4 * kB * T * c) / m**2
thermal_limit = np.sqrt(Sa_th_PSD) * np.ones_like(f)

# -----------------------------
# Ideal force-feedback bound
# In an ideal noiseless force-balance system, the lower bound coincides with the
# thermal limit. For visibility only, we plot it slightly above (+2%).
# -----------------------------
ideal_bound = 1.02 * thermal_limit

# -----------------------------
# Implementation-dependent bound (ASD)
# Model: flat floor + high-frequency rise (closed-loop bandwidth/causality)
# -----------------------------
adc_floor = 8.0e-10    # m s^-2 / sqrt(Hz)
f_c = 5.0              # Hz
impl_bound = adc_floor * np.sqrt(1 + (f / f_c)**4)

# -----------------------------
# Achieved performance (consistent with Fig. 3)
# -----------------------------
brownian = 2.0e-11 * np.ones_like(f)
readout = 5.0e-13 * np.sqrt(1 + (f / 0.1)**4)
actuator = 1.0e-9 * np.ones_like(f)
quant_clock = 3.0e-11 * np.ones_like(f)

achieved_total = np.sqrt(
    brownian**2 +
    readout**2 +
    actuator**2 +
    quant_clock**2
)

# -----------------------------
# Figure
# -----------------------------
fig, axs = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw=dict(hspace=0.12)
)

# =============================
# (a) Bounds + achieved reference
# =============================
ax = axs[0]
ax.loglog(f, thermal_limit, '--', lw=2, label='Thermal (Brownian) limit')
ax.loglog(f, ideal_bound, ':', lw=2.5, label='Ideal force-feedback bound')
ax.loglog(f, impl_bound, lw=2, label='Implementation-dependent bound')
ax.loglog(f, achieved_total, lw=2.5, label='Achieved total self-noise')

ax.set_ylabel(r'Acceleration noise  [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]')
ax.set_xlim(1e-4, 1e2)

# Y-limits chosen to show all bounds clearly, including thermal (~1e-11)
ax.set_ylim(5e-12, 5e-7)

ax.grid(True, which='both', ls=':', lw=0.8)
ax.legend(loc='upper left', frameon=False)

# =============================
# (b) Achieved vs bounds (clean comparison)
# =============================
ax = axs[1]
ax.loglog(f, achieved_total, lw=2.5, label='Achieved total self-noise')
ax.loglog(f, thermal_limit, '--', lw=2, label='Thermal limit')
ax.loglog(f, impl_bound, ':', lw=2, label='Implementation bound')

ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'Acceleration noise  [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]')
ax.set_xlim(1e-4, 1e2)
ax.set_ylim(5e-12, 5e-7)

ax.grid(True, which='both', ls=':', lw=0.8)
ax.legend(loc='upper left', frameon=False)

# -----------------------------
# Panel labels OUTSIDE axes (non-bold)
# -----------------------------
fig.text(0.065, 0.94, '(a)', fontsize=13)
fig.text(0.065, 0.47, '(b)', fontsize=13)

# -----------------------------
# Save
# -----------------------------
plt.tight_layout(rect=[0.06, 0.04, 0.99, 0.96])
plt.savefig('Figure4_metrological_bounds.png', dpi=300, bbox_inches='tight')
plt.show()
