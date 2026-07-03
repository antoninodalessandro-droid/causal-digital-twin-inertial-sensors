import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Constants and nominal settings
# -----------------------------
kB = 1.380649e-23
T = 300.0          # K
m = 1.0            # kg
T0 = 120.0         # s
omega0 = 2*np.pi / T0

# Frequency axis (same as previous figures)
f = np.logspace(-4, 2, 2000)

# -----------------------------
# Baseline noise models (ASD, in m s^-2 / sqrt(Hz))
# Keep consistent with the previous illustrative framework
# -----------------------------
# Readout-equivalent acceleration noise: low at ULP, rising at higher f
def Sa_readout(f):
    return 5.0e-13 * np.sqrt(1.0 + (f / 0.1)**4)

# Quantization+clock / digital shaping: mild rise beyond a corner
def Sa_digital_shape(f, fc=5.0):
    return np.sqrt(1.0 + (f / fc)**4)

# Thermal (Brownian) acceleration ASD for viscous damping c = 2 zeta m omega0
def Sa_thermal(zeta):
    c = 2.0 * zeta * m * omega0
    Sa_th_psd = (4.0 * kB * T * c) / (m**2)          # (m s^-2)^2 / Hz
    return np.sqrt(Sa_th_psd)                        # m s^-2 / sqrt(Hz)

# -----------------------------
# Parameter-dependent components
# -----------------------------
# (a) ENOB sensitivity: ADC-related floor decreases with ENOB
# Simple, monotone scaling: +1 bit -> ~6 dB improvement
def Sa_adc_from_enob(ENOB, base_at_16bit=3.0e-11):
    # scale factor relative to 16-bit
    return base_at_16bit * 2.0**(-(ENOB - 16.0))

# (b) Zeta sensitivity: introduce a realistic trade-off
# - thermal increases with sqrt(zeta) (via c)
# - actuator/feedback noise tends to increase for very low damping (more aggressive control)
def Sa_actuator(zeta, base_at_zeta0=1.0e-9, zeta0=0.20):
    # illustrative trade-off model: lower damping -> larger actuator noise
    return base_at_zeta0 * np.sqrt(zeta0 / zeta)

# -----------------------------
# Figure 5: two spectral envelopes
# -----------------------------
ENOB_list = [12, 14, 16, 18, 20]
zeta_list = [0.05, 0.10, 0.20, 0.30]

# Nominal values used when varying the other parameter
zeta_nom = 0.20
ENOB_nom = 16

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                        gridspec_kw=dict(hspace=0.12))

# =============================
# (a) ENOB sensitivity (spectral)
# =============================
ax = axs[0]

Sa_th_nom = Sa_thermal(zeta_nom) * np.ones_like(f)
Sa_ro = Sa_readout(f)
Sa_act_nom = Sa_actuator(zeta_nom) * np.ones_like(f)

for en in ENOB_list:
    Sa_adc = Sa_adc_from_enob(en) * Sa_digital_shape(f, fc=5.0)
    Sa_tot = np.sqrt(Sa_th_nom**2 + Sa_ro**2 + Sa_act_nom**2 + Sa_adc**2)
    ax.loglog(f, Sa_tot, lw=2, label=f"ENOB = {en}")

# Add thermal limit as reference (single line)
ax.loglog(f, Sa_th_nom, '--', lw=2, label="Thermal limit")

ax.set_ylabel(r'Acceleration noise  [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]')
ax.set_xlim(1e-4, 1e2)
ax.grid(True, which='both', ls=':', lw=0.8)
ax.legend(loc='upper left', frameon=False)

# =============================
# (b) Damping sensitivity (spectral)
# =============================
ax = axs[1]

Sa_adc_nom = Sa_adc_from_enob(ENOB_nom) * Sa_digital_shape(f, fc=5.0)
Sa_ro = Sa_readout(f)

for z in zeta_list:
    Sa_th = Sa_thermal(z) * np.ones_like(f)
    Sa_act = Sa_actuator(z) * np.ones_like(f)
    Sa_tot = np.sqrt(Sa_th**2 + Sa_ro**2 + Sa_act**2 + Sa_adc_nom**2)
    ax.loglog(f, Sa_tot, lw=2, label=rf"$\zeta$ = {z:.2f}")

# Add thermal limit reference for nominal zeta (for context)
ax.loglog(f, Sa_thermal(zeta_nom) * np.ones_like(f), '--', lw=2, label="Thermal limit (nominal)")

ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'Acceleration noise  [m s$^{-2}$/$\sqrt{\mathrm{Hz}}$]')
ax.set_xlim(1e-4, 1e2)
ax.grid(True, which='both', ls=':', lw=0.8)
ax.legend(loc='upper left', frameon=False)

# -----------------------------
# Panel labels OUTSIDE axes (non-bold)
# -----------------------------
fig.text(0.065, 0.94, '(a)', fontsize=13)
fig.text(0.065, 0.47, '(b)', fontsize=13)

# Keep room for panel labels
plt.tight_layout(rect=[0.06, 0.04, 0.99, 0.96])

# Save
plt.savefig('Figure5_spectral_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()
